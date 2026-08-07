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
import struct

from rayfronts import datasets, visualizers, image_encoders, mapping, utils
from rayfronts.behavior_manager import BehaviorManager
#from rayfronts.mode_text_visualizer import ModeTextVisualizer

import rclpy
from rclpy.node import Node
import std_msgs.msg
from std_msgs.msg import String
from nav_msgs.msg import Path, OccupancyGrid, Odometry
try:
  from vision_msgs.msg import Detection2DArray, Detection3DArray
except ImportError:
  Detection2DArray = Detection3DArray = None
from geometry_msgs.msg import PoseStamped
import scipy.ndimage
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs.msg import Image
from std_msgs.msg import Header, ColorRGBA
from sensor_msgs_py import point_cloud2
from sensor_msgs.msg import NavSatFix
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, TransformStamped, PolygonStamped
from tf2_ros import StaticTransformBroadcaster
from collections import deque
from rayfronts import geometry3d as g3d
from rayfronts.geo_frame import FrontierFrame
from rayfronts.info_grid import InfoGrid2D
from rayfronts.person_tracker import PersonTracker
from rayfronts.task_planner import MaippTaskPlanner
from rayfronts.utils import compute_cos_sim
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    DurabilityPolicy,
    HistoryPolicy,
)

logger = logging.getLogger(__name__)

class MappingServer(Node):
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

  @torch.no_grad()
  def __init__(self, cfg):
    super().__init__('mapping_server')
    self.status = MappingServer.Status.INIT
    self._status_lock = threading.RLock()

    self.cfg = cfg
    self.dataset: datasets.PosedRgbdDataset = \
      hydra.utils.instantiate(cfg.dataset)

    # Frontier-mapping frame: a mission-level frame defined by a global origin
    # (lat/lon/alt) and a heading (deg CCW from East). Frontiers are only
    # eligible for the global plan if they fall inside the map_min/max_x/y box
    # expressed in this frame. Its origin in the local/odom frame ("home") is
    # estimated online from GPS + odometry (see gps_callback / the run loop).
    self.geo_frame = FrontierFrame(
      origin_lat=40.413131,
      origin_lon=-79.946393,
      origin_alt=220.80565047594274,
      heading_deg=-76.17)

    # Bag-replay / surveyed-takeoff override: fix home directly instead of
    # waiting for a live GPS fix (bags without the mavros GPS topic would
    # otherwise leave the frontier frame forever not-ready). Enable with:
    #   +fix_home_lat=40.41350 +fix_home_lon=-79.94658 +fix_home_alt=220.8
    fix_home_lat = cfg.get("fix_home_lat", None)
    if fix_home_lat is not None:
      self.geo_frame.set_home(
        float(fix_home_lat), float(cfg.fix_home_lon),
        float(cfg.get("fix_home_alt", 0.0)))
      logger.info(
        "geo_frame home FIXED from config: (%.6f, %.6f, %.1f); live GPS "
        "will not override it.",
        self.geo_frame._home_lat, self.geo_frame._home_lon,
        self.geo_frame._home_alt)

    # Inscribed box (fully inside the 4 surveyed boundary corners) in the
    # frontier frame at heading -76.17. See bound_coordinates.
    # self.map_min_x, self.map_max_x = -34.86, 37.33
    # self.map_min_y, self.map_max_y = -32.09, 29.82
    self.map_min_x, self.map_max_x = -34.86, 34.33
    self.map_min_y, self.map_max_y = -20.09, 7.82

    # Keepout zones (surveyed corners, WGS84; see bound_coordinates). No
    # frontier inside these polygons may be chosen for the global plan. They
    # need not lie fully within the map_min/max box. Converted once to
    # frontier-frame xy (depends only on origin/heading, not on home).
    keepout_zones_gps = [
      [(40.41340, -79.94609), (40.41333, -79.94631),
       (40.41367, -79.94647), (40.41374, -79.94624)],
      [(40.41316, -79.94650), (40.41311, -79.94686),
       (40.41356, -79.94694), (40.41361, -79.94663)],
    ]
    self.keepout_polygons = [
      np.stack([self.geo_frame.gps_to_frame(lat, lon)[:2] for lat, lon in zone])
      for zone in keepout_zones_gps]

    self.behavior_manager = BehaviorManager(
      get_clock=self.get_clock,
      geo_frame=self.geo_frame,
      map_min_x=self.map_min_x, map_max_x=self.map_max_x,
      map_min_y=self.map_min_y, map_max_y=self.map_max_y,
      keepout_polygons=self.keepout_polygons)

    # Latest GPS fix (NavSatFix), used to estimate the local->frontier-frame
    # transform. Same topic the compass planner uses.
    self._latest_gps = None
    gps_qos = QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
    )

    self.create_subscription(
        NavSatFix,
        '/robot_1/interface/mavros/global_position/global',
        self.gps_callback,
        gps_qos,
    )
    
    self.path_publisher = self.create_publisher(Path, '/robot_1/global_plan', 10)
    #self.pc2_publisher = self.create_publisher(PointCloud2, '/colored_pointcloud', 10)
    #self.rays_publisher = self.create_publisher(MarkerArray, '/rays', 10)
    self.voxel_bbox_publisher = self.create_publisher(MarkerArray, '/filtered_voxel_bbox', 10)

    self.filtered_rays_publisher = self.create_publisher(MarkerArray, '/filtered_rays', 10)

    #self.mode_text_publisher = self.create_publisher(Marker, '/mode_text', 10)
    #self.mode_text_visualizer = ModeTextVisualizer(get_clock=self.get_clock, mode_text_publisher = self.mode_text_publisher, node=self)

    self.viewpoint_publisher = self.create_publisher(PointCloud2, "/frontier_viewpoints", 10)

    # Visualizes the frontier selection boundary (in the odom/map frame) so it
    # can be overlaid with /robot_1/global_plan in RViz.
    self.boundary_publisher = self.create_publisher(Marker, "/frontier_boundary", 10)

    self.publisher_dict = {'path': self.path_publisher, 'voxel_bbox': self.voxel_bbox_publisher, 'viewpoint': self.viewpoint_publisher, 'filtered_rays': self.filtered_rays_publisher}

    self.subscriber_dict = {}

    self.waypoint_locked = False
    self.target_waypoint = None
    self.target_waypoint2 = None

    # Frontier/task behavior stays dormant until the robot's odometry
    # altitude first exceeds this (i.e. after takeoff), then latches on.
    # Mapping / home estimation / person tracking run regardless.
    self.behavior_start_altitude = 6.0
    self.behavior_active = False

    self.behavior_mode = 'Frontier-based'

    self.prev_filtered_marker_ids = 0

    self._target_objects = ['green tent']
    #for i in range(len(self._target_objects)):
        #self.add_queries(self._target_objects[i])

    self._background_objects = []
    self.create_subscription(String, '/input_prompt', self.target_object_callback, 10)

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
    if self.cfg.mapping.feat_compressor is not None:
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

      self.encoder = hydra.utils.instantiate(cfg.encoder, **encoder_kwargs)
      mapper_kwargs["encoder"] = self.encoder
      mapper_kwargs["feat_compressor"] = self.feat_compressor

    self.mapper: mapping.RGBDMapping = hydra.utils.instantiate(
      cfg.mapping, intrinsics_3x3=intrinsics_3x3, visualizer=self.vis,
      **mapper_kwargs)

    # 2D class-specific information grid shared with the low-flying drone
    # (MAIPP comms). Cells are keyed in the frontier frame; cell size and
    # vote semantics must match the peer's grid_cell_size (0.5) so coverage
    # cells align across robots. Published as an OccupancyGrid on
    # /robot_1/maipp/coverage_grid (this drone is robot_1).
    self.info_grid = InfoGrid2D(
      vox_size=self.mapper.vox_size,
      geo_frame=self.geo_frame,
      bounds=(self.map_min_x, self.map_max_x,
              self.map_min_y, self.map_max_y),
      cell_size=0.5,
      obstacle_max_height_voxels=10,
      obstacle_min_frac=0.5)
    self.coverage_grid_publisher = self.create_publisher(
      OccupancyGrid, '/robot_1/maipp/coverage_grid', 1)
    # Latched map -> frontier_frame transform so RViz can render the
    # coverage grid overlaid with the rest of the 'map'-frame topics.
    self._tf_static_broadcaster = StaticTransformBroadcaster(self)

    # Person tracking from the gimbal detector (MAIPP tracks contract).
    # Bboxes on /robot_1/gimbal/boundingbox are produced on the front-stereo
    # left frames but in the camera's NATIVE 1920x1080 pixel space; the
    # dataset intrinsics (scaled to depth resolution) are rescaled to match.
    self.person_tracker = None
    if Detection2DArray is None:
      logger.warning("vision_msgs unavailable; person tracking disabled.")
    else:
      det_w, det_h = 1920, 1080
      k_det = self.dataset.intrinsics_3x3.clone().float()
      k_det[0, :] *= det_w / float(self.dataset.depth_w)
      k_det[1, :] *= det_h / float(self.dataset.depth_h)
      self.person_tracker = PersonTracker(
        intrinsics_3x3=k_det,
        det_resolution=(det_w, det_h),
        src2rdf=self.dataset.src2rdf_transform,
        vox_size=self.mapper.vox_size,
        robot_id=1)
      det_qos = QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=10)
      self.create_subscription(
        Detection2DArray, '/robot_1/gimbal/boundingbox',
        self.person_tracker.on_detections, det_qos)
      # Raw odometry buffer for stamp-matched projection (detections arrive
      # at camera rate; the dataset's synced frames are far sparser).
      self.create_subscription(
        Odometry, cfg.dataset.pose_topic,
        self.person_tracker.on_odom, det_qos)
      self.tracks_publisher = self.create_publisher(
        Detection3DArray, '/robot_1/maipp/tracks', 1)
      self.people_markers_publisher = self.create_publisher(
        MarkerArray, '/people_tracks', 1)

    # MAIPP comms cadence: wall-clock, decoupled from the frame-driven
    # query cycle. The low flyer expires peer claims after ~10s
    # (claim_ttl_s), so tying publishes to every Nth frame would let our
    # claim flicker out whenever the frame rate dips.
    self.comms_publish_period_s = 3.0
    self._last_comms_pub = 0.0

    # MAIPP task layer: region-based coverage + person-track revisit task
    # selection over the frontier viewpoints, cooperating with the low
    # flyer (robot_2) through its coverage / tracks / task claims.
    self.task_planner = None
    self._peer_tracks_queue = deque(maxlen=8)
    if self.person_tracker is None:
      logger.warning("Person tracker unavailable; MAIPP task planner "
                     "disabled.")
    else:
      self.task_planner = MaippTaskPlanner(
        geo_frame=self.geo_frame,
        info_grid=self.info_grid,
        tracker=self.person_tracker,
        robot_id=1,
        hover_height=8.0,
        keepout_polygons=self.keepout_polygons,
        # Soft preference for tasks on the +x side of the frontier frame
        # (score units per meter of frame x; see task_w_xbias docstring).
        task_w_xbias=0.10)
      self.behavior_manager.set_task_planner(self.task_planner)
      peer_ns = '/robot_2/maipp'
      self.create_subscription(
        OccupancyGrid, peer_ns + '/coverage_grid',
        partial(self.task_planner.on_peer_cover, 2), 10)
      self.create_subscription(
        PolygonStamped, peer_ns + '/task_claim',
        partial(self.task_planner.on_peer_claim, 2), 10)
      # Peer tracks are queued and fused on the mapping loop thread (the
      # tracker's people list is not lock-protected).
      self.create_subscription(
        Detection3DArray, peer_ns + '/tracks',
        self._on_peer_tracks, 10)
      self.claim_publisher = self.create_publisher(
        PolygonStamped, '/robot_1/maipp/task_claim', 1)

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

    self.sky_feat = self.encoder.encode_labels(['sky'])

    if cfg.querying.query_file is not None:
      with open(cfg.querying.query_file, "r", encoding="UTF-8") as f:
        if cfg.querying.query_file.endswith(".json"):
          cmap_queries = json.load(f)
          queries = list(cmap_queries.keys())
          self._query_cmap = {k: utils.hex_to_rgb(v) for
                              k, v in cmap_queries.items()}
        else:
          queries = [l.strip() for l in f.readlines()]
          self._background_objects = queries
        self.add_queries(queries)

    for i in range(len(self._target_objects)):
        self.add_queries(self._target_objects[i])

    self.messaging_service = None
    if "messaging_service" in cfg and cfg.messaging_service is not None:
      self.messaging_service = hydra.utils.instantiate(
        cfg.messaging_service,
        text_query_callback = self.add_queries if init_encoder else None)

  @torch.no_grad()
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

        # Cache the text query set on the mapper so voxels can be classified
        # outside the query cadence (get_class_partition -> 2D info grid).
        if (hasattr(self.mapper, "update_query_cache")
            and self._queries_feats.get("text", None) is not None
            and len(self._queries_labels.get("text", [])) > 0):
          self.mapper.update_query_cache(
            self._queries_feats["text"], self._queries_labels["text"],
            compressed=self.cfg.querying.compressed)

        self._queries_updated = False
      with self._status_lock:
        if (self.status == MappingServer.Status.IDLE and
           not self.cfg.querying.compute_prob):
          # No need to relog old queries so we clear them.
          self._queries_feats = None
          self._queries_labels.clear()

  @torch.no_grad()
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

      pose_4x4_np = pose_4x4.cpu().numpy()
      cur_pose_np = np.array([float(pose_4x4_np[0][2,3]), float(-pose_4x4_np[0][0,3]), float(-pose_4x4_np[0][1,3])])

      # cur_pose_np is the robot position in the local/odom (ENU-at-home) frame.
      # Pair it with the latest GPS fix to estimate/lock the origin of that
      # frame ("home"), which fixes the local->frontier-mapping-frame transform.
      if self._latest_gps is not None and not self.geo_frame.home_fixed:
        self.geo_frame.update_home(
          self._latest_gps.latitude, self._latest_gps.longitude,
          self._latest_gps.altitude, cur_pose_np)

      # Overlay the frontier selection boundary in the map frame for RViz.
      self.publish_frontier_boundary()

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

      map_t0 = time.time()
      r = self.mapper.process_posed_rgbd(rgb_img, depth_img, pose_4x4, **kwargs)
      map_t1 = time.time()

      # Takeoff trigger: frontier/task behavior starts only once the robot
      # first climbs above behavior_start_altitude, then stays active.
      if not self.behavior_active and \
         cur_pose_np[2] > self.behavior_start_altitude:
        self.behavior_active = True
        logger.info(
          "Robot altitude %.1f m > %.1f m: starting frontier/task selection "
          "behavior.", cur_pose_np[2], self.behavior_start_altitude)

      if self.behavior_active:
        #behavior manager selects mode
        self.behavior_manager.mode_select(queries_labels=self._queries_labels,target_objects = self._target_objects, queries_feats = self._queries_feats, mapper=self.mapper, publisher_dict=self.publisher_dict, subscriber_dict=self.subscriber_dict)

        if self.behavior_mode != self.behavior_manager.behavior_mode:
            self.mode_switch_trigger()

        self.behavior_mode = self.behavior_manager.behavior_mode

        #RVIZ visualizer for /mode_text
        #self.modeTextVisualize(cur_pose_np, self._target_object, self.behavior_mode)

        point3d_dict = {'cur_pose': cur_pose_np, 'target1': self.target_waypoint, 'target2': self.target_waypoint2}

        self.waypoint_locked, self.target_waypoint, self.target_waypoint2 = self.behavior_manager.behavior_execute(self.behavior_mode, self.mapper, point3d_dict, self.waypoint_locked, self.publisher_dict, self.subscriber_dict)

      # Drain queued person detections into tracks (cheap when empty),
      # fusing any freshly received peer tracks first.
      if self.person_tracker is not None:
        while self._peer_tracks_queue:
          self.person_tracker.fuse_peer_tracks(
            self._peer_tracks_queue.popleft(), self.geo_frame)
        self.person_tracker.update(self.mapper)

      if self.vis is not None:
        if i % self.cfg.vis.input_period == 0:
          self.mapper.vis_update(**r)
        if i % self.cfg.vis.map_period == 0:
          self.mapper.vis_map()

      if i % self.cfg.querying.period == 0:
        self.run_queries()
        # Fold new keyframe snapshots into the 2D info grid (right after
        # run_queries so the mapper's query cache is fresh). No-op until
        # the geo_frame home estimate is locked in.
        # Flush pending semantic points into the global store first:
        # update() drains snapshot seqs permanently, and frames processed
        # since the last vox_accum_period flush would otherwise be
        # unlabeled at vote time — their cells would stay unknown unless
        # the camera revisits them.
        self.mapper.accum_semantic_voxels()
        self.info_grid.update(self.mapper)

        if self.person_tracker is not None:
          # Ground plane for detection rays: median height of the info
          # grid's ground cells (falls back to nothing until coverage
          # exists, which also gates projection).
          gys = [c[2] for c in self.info_grid.info_grid.values()
                 if c[0] == 1]
          if gys:
            self.person_tracker.set_ground_height(float(np.median(gys)))

      # MAIPP comms run on their own wall-clock cadence (every loop
      # iteration checks; publishes every comms_publish_period_s).
      self.publish_maipp_comms()

      # Memory housekeeping: on Jetson unified memory, torch's caching
      # allocator holds freed blocks indefinitely and fragments across the
      # accumulation cycles' large transient tensors, ratcheting RSS upward.
      # empty_cache() actually returns those blocks to the OS here.
      if i % 50 == 0 and i > 0:
        torch.cuda.empty_cache()
      if i % 100 == 0:
        try:
          with open("/proc/self/statm") as f:
            rss_gb = int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1e9
          n_vox = (0 if self.mapper.global_vox_xyz is None
                   else self.mapper.global_vox_xyz.shape[0])
          n_rays = (0 if getattr(self.mapper, "global_rays_orig_angles", None)
                    is None else self.mapper.global_rays_orig_angles.shape[0])
          logger.info(
            "[mem] RSS %.1f GB | cuda alloc %.1f / reserved %.1f GB | "
            "sem vox %d | rays %d", rss_gb,
            torch.cuda.memory_allocated() / 1e9,
            torch.cuda.memory_reserved() / 1e9, n_vox, n_rays)
        except Exception:
          pass

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
      logger.debug("[#%4d#] Wall (#%6.4f# ms/batch - #%6.2f# frame/s), "
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
      logger.info("Total Wall (#%6.4f# ms/batch - #%6.2f# frame/s), "
                  "Mapping (#%6.4f# ms/batch - #%6.2f# frame/s), "
                  "Mapping/Wall (#%6.4f%%)", 
                  total_wall*1e3, total_frames_processed/total_wall,
                  total_map*1e3, total_frames_processed/total_map,
                  total_map/total_wall*100)

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

  def target_object_callback(self, msg):
    targets = [t.strip().lower() for t in msg.data.split(",") if t.strip()]
    if not targets:
      self._target_objects = []
    else:
      self._target_objects = targets
    for target in self._target_objects:
      if target not in self._queries_labels['text']:
        self.add_queries(target)
  
  def gps_callback(self, msg):
    self._latest_gps = msg

  def publish_maipp_comms(self):
    """Publishes coverage / tracks / claim + the frontier-frame TF.

    Wall-clock rate-limited to comms_publish_period_s, independent of the
    frame-driven mapping/query cycle: peer claim TTLs (~10s on the low
    flyer) must be outrun even when the frame rate dips.
    """
    now = time.time()
    if now - self._last_comms_pub < self.comms_publish_period_s:
      return
    self._last_comms_pub = now
    stamp = self.get_clock().now().to_msg()
    self.coverage_grid_publisher.publish(
      self.info_grid.build_coverage_msg(OccupancyGrid(), stamp))
    self.publish_frontier_frame_tf()
    if self.person_tracker is not None:
      tracks_msg = self.person_tracker.build_tracks_msg(
        stamp, self.geo_frame)
      if tracks_msg is not None:
        self.tracks_publisher.publish(tracks_msg)
      self.people_markers_publisher.publish(
        self.person_tracker.build_markers_msg(stamp, Marker, MarkerArray))
    if self.task_planner is not None:
      # Advertise the committed task so the low flyer plans around it
      # (empty polygon = idle; same wire format it publishes back).
      self.claim_publisher.publish(
        self.task_planner.build_claim_msg(PolygonStamped(), stamp))

  def _on_peer_tracks(self, msg):
    """Reduces a peer Detection3DArray to plain tuples and queues them."""
    payload = []
    for det in msg.detections:
      if len(det.results) == 0:
        continue
      hyp = det.results[0]
      c = hyp.pose.covariance
      payload.append((
        str(det.id),
        float(hyp.pose.pose.position.x), float(hyp.pose.pose.position.y),
        (float(c[0]), float(c[1]), float(c[6]), float(c[7])),
        float(hyp.hypothesis.score),
        det.header.stamp.sec + det.header.stamp.nanosec * 1e-9))
    if payload:
      self._peer_tracks_queue.append(payload)

  def publish_frontier_frame_tf(self):
    """Broadcasts the map -> frontier_frame transform for RViz.

    'map' is the local/odom ENU frame every other RViz topic here uses. The
    geo_frame affine maps local -> frontier frame (p_f = R p_l + t); a TF
    parent->child transform maps child coordinates into the parent, so we
    broadcast the inverse (p_l = R^-1 p_f - R^-1 t). The affine's geodetic
    tilt is ~1e-4 rad at mission scale, so the rotation is sent yaw-only.
    Re-sent (latched) each publish cycle since home refines until fixed.
    No-op until the frame's home has been estimated.
    """
    if not self.geo_frame.is_ready():
      return
    r_inv = np.linalg.inv(self.geo_frame._affine_R)
    t_inv = -r_inv @ self.geo_frame._affine_t
    yaw = np.arctan2(r_inv[1, 0], r_inv[0, 0])
    tf = TransformStamped()
    tf.header.stamp = self.get_clock().now().to_msg()
    tf.header.frame_id = 'map'
    tf.child_frame_id = 'frontier_frame'
    tf.transform.translation.x = float(t_inv[0])
    tf.transform.translation.y = float(t_inv[1])
    tf.transform.translation.z = float(t_inv[2])
    tf.transform.rotation.z = float(np.sin(yaw / 2.0))
    tf.transform.rotation.w = float(np.cos(yaw / 2.0))
    self._tf_static_broadcaster.sendTransform(tf)

  def publish_frontier_boundary(self, z_disp=8.0):
    """Publish the frontier selection box as a LINE_STRIP Marker.

    The box is defined in the frontier frame; here the 4 corners are
    transformed back into the local/odom frame so the marker overlays with
    /robot_1/global_plan (frame_id 'map'). Drawn at z=z_disp (goal height).
    No-op until the frame's home has been estimated.
    """
    if not self.geo_frame.is_ready():
      return
    corners_frame = [
      (self.map_min_x, self.map_min_y),
      (self.map_max_x, self.map_min_y),
      (self.map_max_x, self.map_max_y),
      (self.map_min_x, self.map_max_y),
      (self.map_min_x, self.map_min_y),  # close the loop
    ]
    marker = Marker()
    marker.header.frame_id = "map"
    marker.header.stamp = self.get_clock().now().to_msg()
    marker.ns = "frontier_boundary"
    marker.id = 0
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD
    marker.scale.x = 0.3  # line width (m)
    marker.color.r = 1.0
    marker.color.g = 0.0
    marker.color.b = 0.0
    marker.color.a = 1.0
    marker.pose.orientation.w = 1.0
    for fx, fy in corners_frame:
      local = self.geo_frame.frame_to_local(np.array([fx, fy, 0.0]))
      p = Point()
      p.x = float(local[0])
      p.y = float(local[1])
      p.z = float(z_disp)
      marker.points.append(p)
    self.boundary_publisher.publish(marker)

    # Keepout zones as orange closed loops (ids 1..N on the same topic).
    for i, poly in enumerate(self.keepout_polygons):
      km = Marker()
      km.header.frame_id = "map"
      km.header.stamp = self.get_clock().now().to_msg()
      km.ns = "frontier_boundary"
      km.id = 1 + i
      km.type = Marker.LINE_STRIP
      km.action = Marker.ADD
      km.scale.x = 0.3
      km.color.r = 1.0
      km.color.g = 0.5
      km.color.b = 0.0
      km.color.a = 1.0
      km.pose.orientation.w = 1.0
      closed = np.vstack([poly, poly[:1]])
      for fx, fy in closed:
        local = self.geo_frame.frame_to_local(np.array([fx, fy, 0.0]))
        p = Point()
        p.x = float(local[0])
        p.y = float(local[1])
        p.z = float(z_disp)
        km.points.append(p)
      self.boundary_publisher.publish(km)

  def mode_switch_trigger(self):
    self.waypoint_locked = False
    self.target_waypoint = None
    self.target_waypoint2 = None
  
  def clear_filtered_rays(self):
    if self.prev_filtered_marker_ids > 0:
      clear_marker_array = MarkerArray()
      for i in range(self.prev_filtered_marker_ids):
        clear_marker = Marker()
        clear_marker.header.frame_id = "map"
        clear_marker.header.stamp = self.get_clock().now().to_msg()
        clear_marker.ns = "arrows"
        clear_marker.id = 1
        clear_marker.action = Marker.DELETE
        clear_marker_array.markers.append(clear_marker)
      self.filtered_rays_publisher.publish(clear_marker_array)
  
  def create_colored_pointcloud_msg(self, xyz_tensor, rgb_tensor):
    xyz = xyz_tensor.cpu().numpy()
    rgb = (rgb_tensor*255).cpu().numpy()
    assert xyz.shape[0] == rgb.shape[0]

    def pack_rgb(r,g,b):
      rgb_int = (int(r) << 16) | (int(g) << 8) | int(b)
      return struct.unpack('f', struct.pack('I', rgb_int))[0]
    
    points = []
    for i in range(xyz.shape[0]):
      xo,yo,zo = xyz[i]
      x,y,z = zo,-xo,-yo
      r,g,b = rgb[i]
      rgb_packed= pack_rgb(r,g,b)
      points.append([x,y,z,rgb_packed])
    
    fields = [PointField(name='x',offset=0,datatype=PointField.FLOAT32, count=1), 
              PointField(name='y',offset=4,datatype=PointField.FLOAT32, count=1), 
              PointField(name='z',offset=8,datatype=PointField.FLOAT32, count=1), 
              PointField(name='rgb',offset=12,datatype=PointField.FLOAT32, count=1)]
    header = Header()
    header.stamp = self.get_clock().now().to_msg()
    header.frame_id = 'map'
    return point_cloud2.create_cloud(header, fields, points)

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
@torch.no_grad()
def main(cfg = None):
  if cfg.seed >= 0:
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
  
  rclpy.init()
  try:
    server = MappingServer(cfg)
  except KeyboardInterrupt:
    logger.info("Shutdown before initializing completed.")
    return

  spin_thread = threading.Thread(target=rclpy.spin, args=(server,), daemon=True)
  spin_thread.start()
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
