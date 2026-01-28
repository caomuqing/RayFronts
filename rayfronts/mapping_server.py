"""Server to run and visualize semantic mapping from a posed RGBD data source

The map server can be queried with images or text with a messaging_service
online or with a predefined query file.

Loads the configs/default.yaml as the root config but all options can be
overwridden by other configs or through the command line. Check hydra-configs
for more details.

"""

import random
import os
import logging
import time
import atexit
import inspect
import threading
import signal
from enum import Enum
from functools import partial
from typing_extensions import List
import json

import torch
import torchvision
import numpy as np
import hydra

from rayfronts import datasets, visualizers, image_encoders, mapping, utils
from rayfronts.utils import compute_cos_sim

logger = logging.getLogger(__name__)

class MappingServer:
  """Server performing mapping on a stream of posed RGBD data. Can be queried.
  
  Attributes:
    status: Status enum signaling the current state of the server.
    cfg: Stores the mapping system configuration as is.
    dataset: Stores the dataset/datasource object.
    vis: Stores the visualizer used. May be None.
    encoder: Stores the encoder model used by the mapper.
    mapper: Stores the mapper used.
    messaging_service: Stores the messaging service used for online querying.
  """

  class Status(Enum):
    INIT = 0 # Server is initializing
    MAPPING = 1 # Server is actively mapping
    IDLE = 2 # Server has stopped mapping and awaits any new queries.
    CLOSING = 3 # Server is in the process of shutting down.
    CLOSED = 4 # Server has shutdown.

  @torch.inference_mode()
  def __init__(self, cfg):
    self.status = MappingServer.Status.INIT
    self._status_lock = threading.RLock()

    self.cfg = cfg
    self.dataset: datasets.PosedRgbdDataset = \
      hydra.utils.instantiate(cfg.dataset)

    intrinsics_3x3 = self.dataset.intrinsics_3x3
    if "vox_size" in cfg.mapping:
      base_point_size = cfg.mapping.vox_size / 2
    else:
      base_point_size = None

    self.vis: visualizers.Mapping3DVisualizer = None
    if "vis" in cfg and cfg.vis is not None:
      self.vis = hydra.utils.instantiate(cfg.vis, intrinsics_3x3=intrinsics_3x3,
                                         base_point_size=base_point_size)

    # Ugly way to check if the chosen mapper constructor needs an encoder.
    c = getattr(mapping, cfg.mapping._target_.split(".")[-1])
    init_encoder = "encoder" in inspect.signature(c.__init__).parameters.keys()
    init_encoder = init_encoder and "encoder" in cfg
    mapper_kwargs = dict()

    self.encoder: image_encoders.ImageEncoder = None
    self.feat_compressor = None
    if ("feat_compressor" in self.cfg.mapping and
        self.cfg.mapping.feat_compressor is not None):
      self.feat_compressor = hydra.utils.instantiate(
        self.cfg.mapping.feat_compressor)

    if init_encoder:
      encoder_kwargs = dict()
      if (cfg.querying.text_query_mode is not None and
          "RadioEncoder" in cfg.encoder and cfg.encoder.lang_model is None):
        raise ValueError("Radio encoder must have a language model if text "
                        "querying is enabled.")
      if "NARadioEncoder" in cfg.encoder._target_:
        encoder_kwargs["input_resolution"] = [self.dataset.rgb_h,
                                              self.dataset.rgb_w]
      if (hasattr(self.dataset, "cat_name_to_index") and
          "classes" in cfg.encoder and cfg.encoder.classes is None):
        encoder_kwargs["classes"] = self.dataset.cat_index_to_name[1:]

      self.encoder = hydra.utils.instantiate(cfg.encoder, **encoder_kwargs)
      mapper_kwargs["encoder"] = self.encoder
      mapper_kwargs["feat_compressor"] = self.feat_compressor

    self.mapper: mapping.RGBDMapping = hydra.utils.instantiate(
      cfg.mapping, intrinsics_3x3=intrinsics_3x3, visualizer=self.vis,
      **mapper_kwargs)

    # Dictionary mapping a label group name to a list of string labels.
    # In the case of a text query, the label is the query. In case of image
    # querying, the label is the image file name.
    self._queries_labels = None

    # Dictionary mapping a label group name to an NxD torch tensor where
    # N corresponds to the number of queries in that group and D is the
    # feature dimension. N must equal len(self.queries_labels[k]) for a group.
    self._queries_feats = None

    # History is used when mapper is idling such that only new queries are
    # visualized as opposed to visualizing the full query set everytime.
    # This also depends on compute_prob value.
    self._queries_labels_history = set()

    # Flag to track if the set of queries has been updated or not.
    self._queries_updated = False

    # Color map mapping query label to color.
    self._query_cmap = dict()

    self._query_lock = threading.RLock()

    if cfg.querying.query_file is not None:
      with open(cfg.querying.query_file, "r", encoding="UTF-8") as f:
        if cfg.querying.query_file.endswith(".json"):
          cmap_queries = json.load(f)
          queries = list(cmap_queries.keys())
          self._query_cmap = {k: utils.hex_to_rgb(v) for
                              k, v in cmap_queries.items()}
        else:
          queries = [l.strip() for l in f.readlines()]
        self.add_queries(queries)

    self.messaging_service = None
    if "messaging_service" in cfg and cfg.messaging_service is not None:
      self.messaging_service = hydra.utils.instantiate(
        cfg.messaging_service,
        text_query_callback = self.add_queries if init_encoder else None)

  @torch.inference_mode()
  def add_queries(self, queries: List[str]):
    """Adds a list of queries to query the map with at fixed intervals.
    
    Args:
      queries: List of string where each string is either a text query or a 
        path to a local image file for image querying.
    """
    if self.encoder is None or not hasattr(self.encoder, "encode_labels"):
      raise Exception("Trying to query without a capable text encoder.")

    if isinstance(queries, str):
      queries = [queries]

    with self._query_lock:
      queries = set(queries).difference(self._queries_labels_history)

      queries = list(queries)
      if len(queries) == 0:
        return

      self._queries_labels_history.update(queries)

    logger.info("Received queries: %s", str(queries))
    img_queries = [x for x in queries if os.path.exists(x)]
    text_queries = [x for x in queries if not os.path.exists(x)]
    text_queries_feats = None
    if len(text_queries) > 0:
      if self.cfg.querying.text_query_mode == "labels":
        text_queries_feats = self.encoder.encode_labels(text_queries)
      elif self.cfg.querying.text_query_mode == "prompts":
        text_queries_feats = self.encoder.encode_prompts(text_queries)
      else:
        raise ValueError("Invalid query type")

    img_queries_feats = None
    if len(img_queries) > 0:
      imgs = list()
      for q in img_queries:
        imgs.append(torch.nn.functional.interpolate(
          torchvision.io.read_image(q).unsqueeze(0).float().cuda()/255,
          size=(self.dataset.rgb_h, self.dataset.rgb_w),
          mode="bilinear", antialias=True))
      imgs = torch.cat(imgs, dim=0)
      img_queries_feats = self.encoder.align_global_features_with_language(
        self.encoder.encode_image_to_vector(imgs))

    queries_labels = dict(text=text_queries, img=img_queries)
    queries_feats = dict(text=text_queries_feats, img=img_queries_feats)
    if (self.feat_compressor is not None and
        self.cfg.querying.compressed):
      if not self.feat_compressor.is_fitted():
        logger.warning("The feature compressor was not fitted. "
                       "Will try to fit to query features which may fail.")
        l = [x for x in queries_feats.values() if x is not None]
        self.feat_compressor.fit(torch.cat(l, dim=0))
      for k,v in queries_feats.items():
        if v is None:
          continue
        queries_feats[k] = self.feat_compressor.compress(v)

    with self._query_lock:
      if self._queries_feats is None:
        self._queries_labels = queries_labels
        self._queries_feats = queries_feats
      else:
        for k, v in queries_feats.items():
          if v is None:
            continue
          if k not in self._queries_feats:
            self._queries_feats[k] = v
            self._queries_labels[k] = queries_labels
          else:
            self._queries_feats[k] = torch.concat(
              (self._queries_feats[k], queries_feats[k]), dim=0)
            self._queries_labels[k].extend(queries_labels[k])

      self._queries_updated = True

  def clear_queries(self):
    with self._query_lock:
      self._queries_labels = None
      self._queries_feats = None
      self._queries_updated = False

  def save_voxels_ply(self, vox_xyz: torch.FloatTensor,
                     vox_labels: torch.LongTensor,
                     class_name: str,
                     output_path: str):
    """Save voxels of a specific class as a PLY point cloud file.
    
    Args:
      vox_xyz: Nx3 float tensor of voxel positions
      vox_labels: N long tensor of class labels
      class_name: Name of the class to save
      output_path: Path to save the PLY file
    """
    # Filter for this class
    class_mask = vox_labels == self._class_name_to_index[class_name]
    if not torch.any(class_mask):
      logger.debug(f"No voxels found for class '{class_name}', skipping PLY export")
      return
    
    class_xyz = vox_xyz[class_mask].cpu().numpy()
    num_points = class_xyz.shape[0]
    
    logger.info(f"Saving {num_points} voxels for class '{class_name}' to {output_path}")
    
    # Write PLY file
    with open(output_path, 'w') as f:
      # PLY header
      f.write("ply\n")
      f.write("format ascii 1.0\n")
      f.write(f"element vertex {num_points}\n")
      f.write("property float x\n")
      f.write("property float y\n")
      f.write("property float z\n")
      f.write("property uchar red\n")
      f.write("property uchar green\n")
      f.write("property uchar blue\n")
      f.write("end_header\n")
      
      # Write vertices with color based on class
      # Use a simple color scheme - you can customize this
      class_colors = {
        "wall": [200, 200, 200],
        "soil pile": [139, 69, 19],
        "beam": [160, 82, 45],
        "excavator": [255, 0, 0],
        "background": [0, 0, 0],
        "unlabelled": [128, 128, 128],
      }
      color = class_colors.get(class_name.lower(), [128, 128, 128])
      
      for i in range(num_points):
        x, y, z = class_xyz[i]
        f.write(f"{x:.6f} {y:.6f} {z:.6f} {color[0]} {color[1]} {color[2]}\n")
    
    logger.info(f"Successfully saved {class_name} voxels to {output_path}")

  @torch.inference_mode()
  def save_class_pointclouds(self):
    """Save point clouds for all classes after mapping is complete."""
    # Check if mapper has voxel data and encoder supports semantic segmentation
    if (self.mapper.is_empty() or 
        self.encoder is None or 
        not hasattr(self.encoder, "encode_labels") or
        not hasattr(self.mapper, "global_vox_xyz") or
        not hasattr(self.mapper, "global_vox_feat")):
      logger.info("Mapper is empty or encoder doesn't support semantic segmentation. "
                  "Skipping class point cloud export.")
      return
    
    # Ensure final accumulation
    if hasattr(self.mapper, "accum_semantic_voxels"):
      self.mapper.accum_semantic_voxels()
    
    vox_xyz = self.mapper.global_vox_xyz
    if vox_xyz is None or vox_xyz.shape[0] == 0:
      logger.info("No voxels in mapper. Skipping class point cloud export.")
      return
    
    vox_feat = self.mapper.global_vox_feat
    if vox_feat is None:
      logger.info("No features in mapper. Skipping class point cloud export.")
      return
    
    # Try to get class names from multiple sources
    class_names = None
    
    # 1. Try dataset first
    if hasattr(self.dataset, "cat_index_to_name") and hasattr(self.dataset, "num_classes"):
      class_names = self.dataset.cat_index_to_name[1:]  # Skip index 0 (ignore class)
      logger.info(f"Using classes from dataset: {class_names}")
    
    # 2. Try encoder config classes
    if (class_names is None or len(class_names) == 0) and hasattr(self.cfg.encoder, "classes"):
      encoder_classes = self.cfg.encoder.classes
      if encoder_classes is not None and len(encoder_classes) > 0:
        class_names = list(encoder_classes)
        # Remove empty string if present (ignore class)
        if len(class_names) > 0 and class_names[0] == "":
          class_names = class_names[1:]
        logger.info(f"Using classes from encoder config: {class_names}")
    
    # 3. Try encoder's own classes (for encoders like GTEncoder, SemSegWrapEncoder)
    if (class_names is None or len(class_names) == 0) and hasattr(self.encoder, "cat_index_to_name"):
      encoder_cat_to_name = self.encoder.cat_index_to_name
      if encoder_cat_to_name is not None and len(encoder_cat_to_name) > 0:
        # Get all names except index 0 (ignore class)
        class_names = [encoder_cat_to_name[i] for i in sorted(encoder_cat_to_name.keys()) if i > 0]
        logger.info(f"Using classes from encoder: {class_names}")
    
    # 4. Try queries (prompt classes)
    if (class_names is None or len(class_names) == 0) and self._queries_labels is not None:
      with self._query_lock:
        # Collect all text queries
        text_queries = []
        if "text" in self._queries_labels:
          text_queries.extend(self._queries_labels["text"])
        if "img" in self._queries_labels:
          # Skip image queries
          pass
        if len(text_queries) > 0:
          class_names = text_queries
          logger.info(f"Using classes from queries: {class_names}")
    
    if class_names is None or len(class_names) == 0:
      logger.info("No classes found from dataset, encoder config, encoder, or queries. "
                  "Skipping class point cloud export.")
      return
    
    logger.info(f"Generating semantic predictions for {len(class_names)} classes...")
    
    # Generate text embeddings for all classes
    text_query_mode = getattr(self.cfg.querying, "text_query_mode", "labels")
    if text_query_mode == "labels":
      text_embeds = self.encoder.encode_labels(class_names)
    elif text_query_mode == "prompts":
      text_embeds = self.encoder.encode_prompts(class_names)
    else:
      logger.warning(f"Unknown text_query_mode '{text_query_mode}'. Using 'labels'.")
      text_embeds = self.encoder.encode_labels(class_names)
    
    # Decompress features if needed
    compressed = getattr(self.cfg.querying, "compressed", False)
    if (self.feat_compressor is not None and compressed):
      vox_feat = self.feat_compressor.decompress(vox_feat)
    
    # Align features with language
    vox_feat_lang = self.encoder.align_spatial_features_with_language(
      vox_feat.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)
    
    # Compute predictions using the same logic as eval_utils.compute_semseg_preds
    # Note: compute_cos_sim(text_embeds, vox_feat) returns (num_voxels, num_classes)
    chunk_size = 10000
    prompt_denoising_thresh = 0.5
    prediction_thresh = 0.1
    num_chunks = int(np.ceil(vox_feat_lang.shape[0] / chunk_size))
    preds = list()
    for c in range(num_chunks):
      sim_vx = compute_cos_sim(
        text_embeds, vox_feat_lang[c*chunk_size: (c+1)*chunk_size], softmax=True)
      
      # Prompt denoising: find classes with low max similarity across all voxels
      # sim_vx is (num_voxels_in_chunk, num_classes)
      max_sim = torch.max(sim_vx, dim=0).values  # (num_classes,)
      low_conf_classes = torch.argwhere(max_sim < prompt_denoising_thresh)
      if low_conf_classes.shape[0] > 0:
        # low_conf_classes is (N, 1), squeeze to (N,) for indexing
        low_conf_indices = low_conf_classes.squeeze(-1)
        sim_vx[:, low_conf_indices] = -torch.inf
      
      # Find best class for each voxel
      sim_value, pred = torch.max(sim_vx, dim=-1)  # pred is (num_voxels_in_chunk,)
      # 0 is the ignore id / no pred, so we add 1
      pred += 1
      pred[sim_value < prediction_thresh] = 0
      preds.append(pred)
    
    vox_labels = torch.cat(preds, dim=0)  # (num_voxels,)
    
    # Create mapping from class name to index
    self._class_name_to_index = {name: idx+1 for idx, name in enumerate(class_names)}
    
    # Create output directory
    output_dir = "pointclouds"
    if hasattr(self.cfg, "output_dir") and self.cfg.output_dir is not None:
      output_dir = self.cfg.output_dir
    elif hasattr(self.dataset, "scene_name"):
      scene_name = self.dataset.scene_name.replace("/", "_")
      output_dir = os.path.join("pointclouds", scene_name)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save PLY file for each class
    for class_name in class_names:
      safe_class_name = class_name.replace(" ", "_").replace("/", "_")
      ply_path = os.path.join(output_dir, f"{safe_class_name}.ply")
      self.save_voxels_ply(vox_xyz, vox_labels, class_name, ply_path)
    
    logger.info(f"Saved point clouds for all classes to {output_dir}/")

  def run_queries(self):
    with self._query_lock:
      if (self._queries_feats is not None and len(self._queries_feats) > 0):
        kwargs = dict()
        if self._query_cmap is not None and len(self._query_cmap) > 0:
          kwargs["vis_colors"] = self._query_cmap

        for k,v in self._queries_labels.items():
          if v is None or len(v) < 1:
            continue
          r = self.mapper.feature_query(
            self._queries_feats[k], softmax=self.cfg.querying.compute_prob,
            compressed=self.cfg.querying.compressed)
          if self.vis is not None and r is not None:
            self.mapper.vis_query_result(r, vis_labels=v, **kwargs)

        self._queries_updated = False
      with self._status_lock:
        if (self.status == MappingServer.Status.IDLE and
           not self.cfg.querying.compute_prob):
          # No need to relog old queries so we clear them.
          self._queries_feats = None
          self._queries_labels.clear()

  @torch.inference_mode()
  def run(self):
    total_wall_t0 = time.time()
    total_map = 0
    total_frames_processed = 0
    wall_t0 = time.time()

    dataloader = list()
    with self._status_lock:
      if self.status == MappingServer.Status.INIT:
        self.status = MappingServer.Status.MAPPING
        dataloader = torch.utils.data.DataLoader(
          self.dataset, batch_size = self.cfg.batch_size)
        logger.info("Datastream opened. Starting mapping.")

    for i, batch in enumerate(dataloader):
      if batch is None:
        break
      rgb_img = batch["rgb_img"].cuda()
      depth_img = batch["depth_img"].cuda()
      pose_4x4 = batch["pose_4x4"].cuda()
      kwargs = dict()
      if "confidence_map" in batch.keys():
        kwargs["conf_map"] = batch["confidence_map"].cuda()

      if self.cfg.depth_limit >= 0:
        depth_img[torch.logical_and(
          torch.isfinite(depth_img),
          depth_img > self.cfg.depth_limit)] = torch.inf

      # Visualize inputs
      if self.vis is not None:
        if i % self.cfg.vis.pose_period == 0:
          self.vis.log_pose(batch["pose_4x4"][-1])
        if i % self.cfg.vis.input_period == 0:
          self.vis.log_img(batch["rgb_img"][-1].permute(1,2,0))
          self.vis.log_depth_img(depth_img.cpu()[-1].squeeze())
          if "confidence_map" in batch.keys():
            self.vis.log_img(batch["confidence_map"][-1])
          if "semseg_img" in batch.keys():
            self.vis.log_label_img(batch["semseg_img"][-1])

      map_t0 = time.time()
      r = self.mapper.process_posed_rgbd(rgb_img, depth_img, pose_4x4, **kwargs)
      map_t1 = time.time()

      if self.vis is not None:
        if i % self.cfg.vis.input_period == 0:
          self.mapper.vis_update(**r)
        if i % self.cfg.vis.map_period == 0:
          self.mapper.vis_map()

      if i % self.cfg.querying.period == 0:
        self.run_queries()

      if self.vis is not None:
        self.vis.step()

      # Stat calculation
      total_frames_processed += rgb_img.shape[0]
      map_p = map_t1-map_t0
      total_map += map_p
      map_thr = rgb_img.shape[0] / map_p
      wall_t1 = time.time()
      wall_p = wall_t1 - wall_t0
      wall_thr = rgb_img.shape[0] / wall_p
      wall_t0 = wall_t1
      logger.info("[#%4d#] Wall (#%6.4f# ms/batch - #%6.2f# frame/s), "
                  "Mapping (#%6.4f# ms/batch - #%6.2f# frame/s), "
                  "Mapping/Wall (#%6.4f%%)", 
                  i, wall_p*1e3, wall_thr, map_p*1e3, map_thr,
                  map_p/wall_p*100)

      with self._status_lock:
        if self.status != MappingServer.Status.MAPPING:
          logger.info("Mapping stopped.")
          break

    # Final stat calculation
    total_wall_t1 = time.time()
    total_wall = total_wall_t1 - total_wall_t0
    if total_map > 0 and total_wall > 0:
      logger.info("Total Wall (#%6.4f# s - #%6.2f# frame/s), "
                  "Total Mapping (#%6.4f# s - #%6.2f# frame/s), "
                  "Mapping/Wall (#%6.4f%%)", 
                  total_wall, total_frames_processed/total_wall,
                  total_map, total_frames_processed/total_map,
                  total_map/total_wall*100)

    # Save point clouds for all classes after data runs out
    self.save_class_pointclouds()

    # Shutting down or transitioning to idling
    self._status_lock.acquire()
    if self.status == MappingServer.Status.MAPPING:
      if self.messaging_service is not None:
        self.status = MappingServer.Status.IDLE
        try:
          self.dataset.shutdown()
        except AttributeError:
          pass # Its fine dataset doesn't have shutdown function
      else:
        self.shutdown()
        return

    # No new data is coming so we only need to add new queries and not
    # update old ones. Unless compute_prob is set to true b.c new queries
    # will not affect old results.
    if not self.cfg.querying.compute_prob:
      self._queries_feats = None
      self._queries_labels.clear()
    # Idling loop
    while self.status == MappingServer.Status.IDLE:
      self._status_lock.release()
      time.sleep(1)
      with self._query_lock:
        if self._queries_updated:
          self.run_queries()
      self._status_lock.acquire()

    self.status = MappingServer.Status.CLOSED
    self._status_lock.release()
    self.shutdown()

  def shutdown(self):
    with self._status_lock:
      self.status = MappingServer.Status.CLOSING
    if self.messaging_service is not None:
      self.messaging_service.shutdown()
    if self.dataset is not None:
      try:
        self.dataset.shutdown()
      except AttributeError:
        pass
    if self.vis is not None:
      try:
        self.vis.shutdown()
      except AttributeError:
        pass
    with self._status_lock:
      self.status = MappingServer.Status.CLOSED

def signal_handler(mapping_server: MappingServer, sig, frame):
  with mapping_server._status_lock:
    if mapping_server.status == MappingServer.Status.MAPPING:
      if mapping_server.messaging_service is not None:
        logger.info(
          "Received interrupt signal. Stopping mapping. Messaging service is "
          "still online. Interrupt again to shutdown.")

        mapping_server.status = MappingServer.Status.IDLE
      else:
        logger.info("Received interrupt signal. Shutting down.")
        mapping_server.status = MappingServer.Status.CLOSING

    elif mapping_server.status == MappingServer.Status.IDLE:
      logger.info("Received interrupt signal. Shutting down.")
      mapping_server.status = MappingServer.Status.CLOSING
  try:
    mapping_server.dataset.shutdown()
  except AttributeError:
    pass # Its fine dataset doesn't have shutdown function

@hydra.main(version_base=None, config_path="configs", config_name="default")
@torch.inference_mode()
def main(cfg = None):
  if cfg.seed >= 0:
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

  try:
    server = MappingServer(cfg)
  except KeyboardInterrupt:
    logger.info("Shutdown before initializing completed.")
    return

  signal.signal(signal.SIGINT, partial(signal_handler, server))
  try:
    server.run()
  except Exception as e:
    server.shutdown()
    raise e

if __name__ == "__main__":
  # Cleanup for nanobind. See https://github.com/wjakob/nanobind/issues/19
  def cleanup():
    import typing
    for cleanup in typing._cleanups:
      cleanup()
  atexit.register(cleanup)

  main()
