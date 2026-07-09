"""Module defining RayFronts ! The semantic ray frontier map.

Typical usage example:

  map = SemanticRayFrontiersMap(intrinsics_3x3, None, visualizer, encoder)
  for batch in dataloader:
    rgb_img = batch["rgb_img"].cuda()
    depth_img = batch["depth_img"].cuda()
    pose_4x4 = batch["pose_4x4"].cuda()
    map.process_posed_rgbd(rgb_img, depth_img, pose_4x4)
  map.vis_map()

  r = map.text_query(["man wearing a blue shirt"])
  map.vis_query_result(r)

  map.save("test.pt")
"""

from typing_extensions import override, List, Tuple, Dict
import sys
import os
import math
import logging

import torch
import openvdb

from rayfronts.mapping.base import SemanticRGBDMapping
from rayfronts import (geometry3d as g3d, visualizers, image_encoders,
                       feat_compressors)
from rayfronts.utils import compute_cos_sim

sys.path.insert(
  0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../csrc/build/"))
)
import rayfronts_cpp

logger = logging.getLogger(__name__)

class SemanticRayFrontiersMap(SemanticRGBDMapping):
  """RayFronts: Semantic Rays + Frontiers + Semantic Voxels + Occupancy map.

  Attributes:
    intrinsics_3x3: See base.
    device: See base.
    visualizer: See base.
    clip_bbox: See base
    encoder: See base.
    feat_compressor: See base.
    interp_mode: See base.

    max_pts_per_frame: See __init__.
    vox_size: See __init__.
    max_empty_pts_per_frame: See __init__.
    max_rays_per_frame: See __init__.

    max_depth_sensing: See __init__.
    max_empty_cnt: See __init__.
    max_occ_cnt: See __init__.
    occ_observ_weight: See __init__.
    occ_thickness: See __init__.
    vox_accum_period: See __init__.
    occ_pruning_tolerance: See __init__.
    occ_pruning_period: See __init__.
    sem_pruning_thresh: See __init__.
    sem_pruning_period: See __init__.

    fronti_neighborhood_r: See __init__.
    fronti_min_unobserved: See __init__.
    fronti_min_empty: See __init__.
    fronti_min_occupied: See __init__.
    fronti_subsampling: See __init__.
    fronti_subsampling_min_fronti: See __init__.

    ray_accum_period: See __init__.
    ray_accum_phase: See __init__.
    angle_bin_size: See __init__.
    ray_erosion: See __init__.
    zero_depth_mode: See __init__.
    ray_tracing: See __init__.
    global_encoding: See __init__.

    occ_map_vdb: An OpenVDB Int8 Grid storing log-odds occupancy.
      more information about available functions on the grid can be found below:
      https://www.openvdb.org/documentation/doxygen/classopenvdb_1_1v12__0_1_1Grid.html
    global_vox_xyz: Nx3 Float tensor describing voxel centroids in world
      coordinate frame.
    global_vox_rgb_feat_cnt: (Nx(3+C+1)) Float tensor; 3 for rgb, C for
      features, 1 for hit count. This tensor is aligned with
      global_vox_xyz.
    frontiers: (Nx3) Float tensor describing locations of map frontiers.
    frontiers_neighbor_cnts: (Nx3) Float tensor describing neighborhood counts
      (empty, unobserved, occupied) per frontier, or None.
    global_rays_orig_angles: Mx(3+2) 3 for origin xyz and 2 for spherical angles
      theta (azimuthal angle in the xy-plane from the x-axis with -pi<=theta<pi)
      and phi (polar/zenith angle from the positive z-axis with 0<=phi<=pi)
    global_rays_feats_cnt: Mx(C+1) C for features, 1 for confidence weight.
  """
  def __init__(self,
               intrinsics_3x3: torch.FloatTensor,
               device: str = None,
               visualizer: visualizers.Mapping3DVisualizer = None,
               clip_bbox: Tuple[Tuple] = None,
               encoder: image_encoders.ImageEncoder = None,
               feat_compressor: feat_compressors.FeatCompressor = None,
               interp_mode: str = "bilinear",

               max_pts_per_frame: int = -1,
               vox_size: int = 1,
               vox_accum_period: int = 1,
               max_empty_pts_per_frame: int = -1,
               max_rays_per_frame: int = -1,

               max_depth_sensing: float = -1,
               max_empty_cnt: int = 3,
               max_occ_cnt: int = 5,
               occ_observ_weight: int = 5,
               occ_thickness: int = 2,
               occ_pruning_tolerance: int = 2,
               occ_pruning_period: int = 1,
               sem_pruning_thresh: int = 0,
               sem_pruning_period: int = 1,

               fronti_neighborhood_r: int = 1,
               fronti_min_unobserved: int = 4,
               fronti_min_empty: int = 2,
               fronti_min_occupied: int = 0,
               fronti_subsampling: int = 4,
               fronti_subsampling_min_fronti: int = 10,

               ray_accum_period: int = 2,
               ray_accum_phase: int = 1,
               angle_bin_size: float = 30,
               ray_erosion: int = 1,
               ray_tracing: bool = False,
               global_encoding: bool = False,
               zero_depth_mode: bool = False,
               infer_direction: bool = False,
               keep_frontier_neighbor_cnts: bool = False,
               visibility_tolerance: float = 1.0,
               coverage_min_unlabeled_frac: float = 0.5,
               class_frontier_classes: List[str] = None,
               class_frontier_min_prob: float = 0.4,
               class_frontier_subsampling: int = 3,
               class_frontier_min_cnt: int = 4,
               class_frontier_top_surface_only: bool = True,
               debug_log_period: int = 0):
    """
    Args:
      intrinsics_3x3: See base.
      device: See base.
      visualizer: See base.
      clip_bbox: See base.
      encoder: See base.
      feat_compressor: See base.
      interp_mode: See base.

      max_pts_per_frame: How many points to project per frame. Set to -1 to 
        project all valid depth points.
      vox_size: Length of a side of a voxel in meters.
      vox_accum_period: How often do we aggregate voxels into the global 
        representation. Setting to 10, will accumulate point clouds from 10
        frames before voxelization. Should be tuned to balance memory,
        throughput, and min latency.

      max_empty_pts_per_frame: How many empty points to project per frame.
        Set to -1 to project all valid depth points.
      max_depth_sensing: Depending on the max sensing range, we project empty
        voxels up to that range if that pixel had +inf depth or out of range.
        Set to -1 to use the max depth in that frame as the max sensor range.
      max_empty_cnt: The maximum log odds value for empty voxels. 3 means the
        cell will be capped at -3 which corresponds to a
        probability of e^-3 / ( e^-3 + 1 ) ~= 0.05 Lower values help compression
        and responsivness to dynamic objects whereas higher values help
        stability and retention of more evidence.
      max_occ_cnt: The maximum log odds value for occupied voxels. Same
        discussion of max_empty_cnt applies here.
      occ_observ_weight: How much weight does an occupied observation hold
        over an empty observation.
      occ_thickness: When projecting occupied points, how many points do we
        project as occupied? e.g. Set to 3 to project 3 points centered around
        the original depth value with vox_size/2 spacing between them. This
        helps reduce holes in surfaces.
      occ_pruning_tolerance: Tolerance when merging voxels into bigger nodes.
      occ_pruning_period: How often do we prune occupancy into bigger voxels.
        Set to -1 to disable.
      sem_pruning_thresh: Voxels with log-odds occupancy below this value are
        removed.
      sem_pruning_period: How often do we prune semantic voxels to reflect
        occupancy (That is erase semantic voxels that are no longer occupied).
        Set to -1 to disable.

      fronti_neighborhood_r: 3D neighborhood radius to compute if a voxel is a
        frontier or not.
      fronti_min_unobserved: Minimum number of unobserved cells in the
        neighborhood of a cell for it to be considered a frontier.
      fronti_min_empty: Minimum number of empty/free cells in the
        neighborhood of a cell for it to be considered a frontier.
      fronti_min_occupied: Minimum number of occupied cells in the
        neighborhood of a cell for it to be considered a frontier.
      fronti_subsampling: After computing frontiers, we subsample using the
        below factor.
      fronti_subsampling_min_fronti: When subsampling (Clustering frontiers into
        bigger cells), how many frontiers should lie in the big cell to consider
        it as a frontier. This is heavilly tied to the subsampling factor. Ex. A
        subsampling factor of 4 means 4^3 cells will cluster into one cell.

      ray_accum_period: How often do we accumulate/bin the rays.
      ray_accum_phase: A phase term to offset the ray accumulation such that it
        does not happen with voxel accumulation.
      angle_bin_size: Bin size when discretizing the angles of rays in degrees.
      ray_erosion: Should we erode the out of range depth mask before shooting
        the rays ? Set to 0 to disable erosion. 1 means an erosion kernel that 
        is 3x3.
      ray_tracing: Enables ray tracing when projecting rays to frontiers. Slows
        things down but gives more accurate ray to frontier placement.
      global_encoding: Instead of using the spatial/dense features for rays, 
        encode the whole image into one feature vector.
      zero_depth_mode: Pose graph mode when depth is not available/unreliable, 
        or there is no desire for dense voxel mapping. All rays attach to the
        current location.
      infer_direction: Whether to infer frontier directions based on occupancy.
      keep_frontier_neighbor_cnts: Whether to store per-frontier neighborhood
        counts (empty, unobserved, occupied) in self.frontiers_neighbor_cnts.
      visibility_tolerance: In the decoupled pipeline
        (process_semantic_frame), a voxel is considered visible if its camera
        depth is within visibility_tolerance*vox_size of the closest voxel
        projecting to the same pixel.
      coverage_min_unlabeled_frac: A coverage-frontier cluster is flagged only
        if the unlabeled fraction of its occupied voxels is at least this
        value (in addition to the fronti_subsampling_min_fronti count), so
        partially labeled surfaces are not flagged.
      class_frontier_classes: List of class labels (must be part of the active
        query set) whose boundary is computed as class frontiers: voxels
        classified as these classes that are adjacent to occupied-but-unlabeled
        voxels or to raw geometric frontier cells. Set to None to disable.
      class_frontier_min_prob: Minimum softmax probability of the argmax class
        for a voxel to be considered part of a target class.
      class_frontier_subsampling: Clustering factor (x vox_size) for class
        frontier boundary voxels.
      class_frontier_min_cnt: Minimum boundary voxels per cluster cell for the
        cell to be kept as a class frontier point.
      class_frontier_top_surface_only: Keep, per lateral (x,z) column, only
        boundary voxels at the class's top surface, removing boundary points
        beneath it (e.g. sub-surface floor voxels adjacent to the unlabeled
        interior of the ground). Right for horizontal-surface classes such as
        ground/floor; set to False for vertical classes such as walls where
        boundaries below the top edge are meaningful.
      debug_log_period: If > 0, print mapper point/voxel stats every N calls.
    """
    super().__init__(intrinsics_3x3, device, visualizer, clip_bbox, encoder,
                     feat_compressor, interp_mode)

    self.max_pts_per_frame = max_pts_per_frame
    self.max_dirs_per_frame = max_rays_per_frame

    self.occ_pruning_period = occ_pruning_period
    self.occ_pruning_tolerance = occ_pruning_tolerance
    self._occ_pruning_cnt = 0

    self.sem_pruning_period = sem_pruning_period
    self.sem_pruning_thresh = sem_pruning_thresh
    self._sem_pruning_cnt = 0

    self.fronti_neighborhood_r = fronti_neighborhood_r
    self.fronti_min_unobserved = fronti_min_unobserved
    self.fronti_min_empty = fronti_min_empty
    self.fronti_min_occupied = fronti_min_occupied
    self.fronti_subsampling = fronti_subsampling
    self.fronti_subsampling_min_fronti = fronti_subsampling_min_fronti
    self.angle_bin_size = angle_bin_size

    self.vox_size = vox_size

    self.max_empty_pts_per_frame = max_empty_pts_per_frame
    self.max_depth_sensing = max_depth_sensing
    self.max_empty_cnt = max_empty_cnt
    self.max_occ_cnt = max_occ_cnt
    self.occ_observ_weight = occ_observ_weight
    self.occ_thickness = occ_thickness

    self.ray_erosion = ray_erosion
    self.ray_tracing = ray_tracing
    self.global_encoding = global_encoding
    self.zero_depth_mode = zero_depth_mode
    if infer_direction and angle_bin_size < 360:
      logger.warning("Setting infer direction to true while not in semantic "
                     "frontier mode may not make sense !")
    self.infer_direction = infer_direction
    self.keep_frontier_neighbor_cnts = keep_frontier_neighbor_cnts
    self.visibility_tolerance = float(visibility_tolerance)
    self.coverage_min_unlabeled_frac = float(coverage_min_unlabeled_frac)
    self.class_frontier_classes = list(class_frontier_classes) \
      if class_frontier_classes is not None else []
    self.class_frontier_min_prob = float(class_frontier_min_prob)
    self.class_frontier_subsampling = int(class_frontier_subsampling)
    self.class_frontier_min_cnt = int(class_frontier_min_cnt)
    self.class_frontier_top_surface_only = bool(class_frontier_top_surface_only)
    self._class_frontier_warned = False
    # TEMP(viz-only): height band (min, max) in metres for all voxel/frontier
    # visualization layers. Up = -y in world RDF, relative to world origin.
    # Set to None to visualize everything again. Does NOT affect the map,
    # queries, or messaging — display only.
    self.vis_height_range = None
    self.debug_log_period = int(debug_log_period)
    self._debug_frame_idx = 0

    v = self.vox_size

    ### Core data structures

    self.occ_map_vdb = openvdb.Int8Grid()
    # FIXME: This following line causes "nanobind: leaked 1 instances!" upon
    # exiting.
    self.occ_map_vdb.transform = openvdb.createLinearTransform(voxelSize=v)

    # (Nx3)
    self.global_vox_xyz = None
    # (Nx(3+C+1)) 3 for rgb, C for features, 1 for observation count
    # We keep this in the same tensor to avoid having to constantly continue
    # concatenating and slicing when performing voxelization.
    # TODO: cap count such that it can be updated if with dynamic env.
    self.global_vox_rgb_feat_cnt = None

    # Fx3
    self.frontiers = None
    # Fx3 (empty_cnt, unobserved_cnt, occupied_cnt) per frontier, or None
    self.frontiers_neighbor_cnts = None
    # Kx3 occupied-but-unlabeled voxel clusters (decoupled pipeline);
    # boundaries of camera semantic coverage. See
    # update_semantic_coverage_frontiers.
    self.semantic_coverage_frontiers = None
    # Kx3 clustered boundary points of the selected semantic classes.
    # See update_class_frontiers.
    self.class_frontiers = None
    # Nx3 voxel centers classified as one of class_frontier_classes at the
    # last update_class_frontiers call (used e.g. as exploration anchors).
    self.class_voxels_xyz = None
    # Raw (pre-subsampling) geometric frontier cells at vox_size resolution.
    # Maintained by update_frontiers; used for class-frontier adjacency.
    self._frontiers_raw = None
    # Nx3 occupied-but-unlabeled voxel centers cached by
    # update_semantic_coverage_frontiers; used for class-frontier adjacency.
    self._unlabeled_occ_xyz = None
    # Mx(3+2) 3 for origin and 2 for angle
    self.global_rays_orig_angles = None
    # Mx(C+1) C for features, 1 for count.
    self.global_rays_feats_cnt = None

    ### Temporary accumulation variables

    # Point cloud and voxel accumulation before global update
    self.vox_accum_period = vox_accum_period
    self._vox_accum_cnt = 0

    self._tmp_vox_xyz = list()
    self._tmp_vox_occ = list()
    self._tmp_vox_xyz_since_prune = list()

    self._tmp_pc_xyz = list()
    self._tmp_pc_rgb_feat_cnt = list()

    # Semantic ray accumulation before global update
    self.ray_accum_period = ray_accum_period
    self.ray_accum_phase = ray_accum_phase
    self._ray_accum_delay = self.ray_accum_phase
    self._ray_accum_cnt = 0
    self._tmp_ray_orig = list()
    self._tmp_ray_dir = list()
    self._tmp_ray_feat = list()

  @property
  def global_vox_rgb(self):
    if self.global_vox_rgb_feat_cnt is None:
      return None
    else:
      return self.global_vox_rgb_feat_cnt[:, :3]

  @property
  def global_vox_feat(self):
    if self.global_vox_rgb_feat_cnt is None:
      return None
    else:
      return self.global_vox_rgb_feat_cnt[:, 3:-1]

  @property
  def global_vox_cnt(self):
    if self.global_vox_rgb_feat_cnt is None:
      return None
    else:
      return self.global_vox_rgb_feat_cnt[:, -1:]

  @property
  def global_rays_feat(self):
    if self.global_rays_feats_cnt is None:
      return None
    else:
      return self.global_rays_feats_cnt[:, :-1]

  @property
  def global_rays_cnt(self):
    if self.global_rays_feats_cnt is None:
      return None
    else:
      return self.global_rays_feats_cnt[:, -1:]

  @override
  def save(self, file_path):
    raise NotImplementedError()

  @override
  def load(self, file_path):
    raise NotImplementedError()

  @override
  def is_empty(self) -> bool:
    return (self.occ_map_vdb.empty() and
            (self.global_vox_xyz is None or
             self.global_vox_xyz.shape[0] == 0) and
            (self.global_rays_orig_angles is None or
             self.global_rays_orig_angles.shape[0] == 0))

  def summarize_frontier_feats(self):
    """Summarize the frontier rays into frontiers by averaging features.
    
    Returns:
      xyz: Nx3 float tensor describing the locations of frontiers that have
        features.
      feats: NxC float tensor describing frontier features.
    """
    if self.global_rays_orig_angles is None:
      return None, None
    fronti_xyz = torch.clone(self.global_rays_orig_angles[:, :3])
    fronti_feats_cnt = torch.clone(self.global_rays_feats_cnt)
    fronti_xyz, fronti_feats_cnt = g3d.add_weighted_sparse_voxels(
      fronti_xyz, fronti_feats_cnt,
      torch.zeros_like(fronti_xyz[1:1]),
      torch.zeros_like(fronti_feats_cnt[1:1]),
      self.vox_size)

    return fronti_xyz, fronti_feats_cnt[:, :-1]

  @override
  def process_posed_rgbd(self,
                         rgb_img: torch.FloatTensor,
                         depth_img: torch.FloatTensor,
                         pose_4x4: torch.FloatTensor,
                         conf_map: torch.FloatTensor = None,
                         feat_img: torch.FloatTensor = None) -> dict:
    update_info = dict()

    # TODO: Decouple ray resolution from depth resolution. Would be beneficial
    # to project a smaller number of rays at lower resolution to reduce
    # object semantics leaking at the boundaries.

    r = g3d.depth_to_sparse_occupancy_voxels(
      depth_img, pose_4x4, self.intrinsics_3x3, self.vox_size, conf_map,
      max_num_pts = self.max_pts_per_frame,
      max_num_empty_pts = self.max_empty_pts_per_frame,
      max_num_dirs = self.max_dirs_per_frame,
      max_depth_sensing = self.max_depth_sensing,
      occ_thickness=self.occ_thickness,
      return_pc=True, return_dirs= not self.global_encoding,
      dirs_erosion=self.ray_erosion,
    )
    if not self.global_encoding:
      vox_xyz, vox_occ, pc_xyz, selected_pc_ind, \
        origs, dirs, selected_dir_ind = r
    else:
      vox_xyz, vox_occ, pc_xyz, selected_pc_ind = r
      origs = torch.zeros(pose_4x4.shape[0], 1, 3,
                          dtype=torch.float, device=self.device)
      dirs = torch.zeros(pose_4x4.shape[0], 1, 3,
                         dtype=torch.float, device=self.device)
      dirs[..., -1] = 1
      origs = g3d.transform_points(origs, pose_4x4).reshape(-1, 3)
      dirs = g3d.transform_points(dirs, pose_4x4).reshape(-1, 3) - origs

    vox_xyz, vox_occ = self._clip_pc(vox_xyz, vox_occ)
    pc_xyz, selected_pc_ind = self._clip_pc(
      pc_xyz, selected_pc_ind.unsqueeze(-1))
    selected_pc_ind = selected_pc_ind.squeeze(-1)

    if (self.debug_log_period > 0
        and self._debug_frame_idx % self.debug_log_period == 0):
      depth_valid = int(
        (torch.isfinite(depth_img) & (depth_img > 0)).sum().item())
      ray_count = int(origs.shape[0]) if origs is not None else 0
      logger.info(
        "Mapper debug frame=%d depth_valid=%d pc_xyz=%d vox_xyz=%d "
        "ray_origins=%d max_pts_per_frame=%d max_empty_pts_per_frame=%d",
        self._debug_frame_idx, depth_valid, pc_xyz.shape[0],
        vox_xyz.shape[0], ray_count, self.max_pts_per_frame,
        self.max_empty_pts_per_frame)
      print(
        "[Mapper debug] "
        f"frame={self._debug_frame_idx} depth_valid={depth_valid} "
        f"pc_xyz={pc_xyz.shape[0]} vox_xyz={vox_xyz.shape[0]} "
        f"ray_origins={ray_count} "
        f"max_pts_per_frame={self.max_pts_per_frame} "
        f"max_empty_pts_per_frame={self.max_empty_pts_per_frame}",
        flush=True)
    self._debug_frame_idx += 1

    B, _, rH, rW = rgb_img.shape
    B, _, dH, dW = depth_img.shape

    if rH != dH or rW != dW:
      pts_rgb = torch.nn.functional.interpolate(
        rgb_img,
        size=(dH, dW),
        mode=self.interp_mode,
        antialias=self.interp_mode in ["bilinear", "bicubic"])
    else:
      pts_rgb = rgb_img

    pts_rgb = pts_rgb.permute(0, 2, 3, 1).reshape(-1, 3)[selected_pc_ind]

    if not self.global_encoding:
      if feat_img is None:
        feat_img = self.encoder.encode_image_to_feat_map(rgb_img)
      feat_img = self._proj_resize_feat_map(feat_img, dH, dW)
      update_info["feat_img"] = feat_img

      feat_img_flat = feat_img.permute(0, 2, 3, 1).reshape(-1,
                                                           feat_img.shape[1])

      dirs_feat = feat_img_flat[selected_dir_ind]
      pts_feat = feat_img_flat[selected_pc_ind]
      del feat_img_flat

    else:
      feat_vec = self.encoder.encode_image_to_vector(rgb_img)
      dirs_feat = feat_vec
      pts_feat = feat_vec[selected_pc_ind // (dH*dW)]

    N = pts_rgb.shape[0]
    pts_rgb_feat_cnt = torch.cat(
      (pts_rgb, pts_feat, torch.ones((N, 1), device=self.device)), dim=-1)
    # [0, 1] to [-1, occ_observ_weight]
    vox_occ = vox_occ*self.occ_observ_weight-1

    self._vox_accum_cnt += B
    self._occ_pruning_cnt += B
    self._sem_pruning_cnt += B

    if self._ray_accum_delay > 0:
      self._ray_accum_delay -= B
    else:
      self._ray_accum_cnt += B

    if vox_xyz.shape[0] > 0:
      self._tmp_vox_occ.append(vox_occ)
      self._tmp_vox_xyz.append(vox_xyz)
    if pc_xyz.shape[0] > 0:
      self._tmp_pc_xyz.append(pc_xyz)
      self._tmp_pc_rgb_feat_cnt.append(pts_rgb_feat_cnt)

    if origs.shape[0] > 0:
      self._tmp_ray_feat.append(dirs_feat)
      self._tmp_ray_orig.append(origs)
      self._tmp_ray_dir.append(dirs)

    if self._vox_accum_cnt >= self.vox_accum_period:
      self._vox_accum_cnt = 0
      self._run_accumulation_step()

    if self._ray_accum_cnt >= self.ray_accum_period:
      self._ray_accum_cnt = 0

      ## 6. Prune semantic rays
      self.prune_semantic_rays()

      ## 7. Update semantic rays
      self.cast_semantic_rays()

      if self.infer_direction:
        self.compute_inferred_directions()

    return update_info

  def _run_accumulation_step(self) -> None:
    """Accumulate tmp buffers into the global map, prune, update frontiers."""
    ## 1. Accumulate occupancy voxels
    updated_vox_xyz = self.accum_occ_voxels()

    ## 2. Accumulate semantic voxels
    self.accum_semantic_voxels()

    ## 3. Prune occupancy map
    if (self.occ_pruning_period > -1 and
        self._occ_pruning_cnt >= self.occ_pruning_period):

      self._occ_pruning_cnt = 0
      self.occ_map_vdb.prune(self.occ_pruning_tolerance)

    ## 4. Prune semantic voxels
    if (self.sem_pruning_period > -1 and
        self.global_vox_xyz is not None and
        self.global_vox_xyz.shape[0] > 0 and
        self._sem_pruning_cnt >= self.sem_pruning_period and
        len(self._tmp_vox_xyz_since_prune) > 0):

      self._sem_pruning_cnt = 0
      self.prune_semantic_voxels(
        torch.cat(self._tmp_vox_xyz_since_prune, dim=0))
      self._tmp_vox_xyz_since_prune.clear()

    ## 5. Update Frontiers

    # Compute active window/bbox.
    # TODO: Test if its faster to project boundary points and pose centers
    # instead of doing min max over all tmp voxels. Or maybe let occ_pc2vdb
    # return the bounding box since it will iterate over all voxels already.
    if updated_vox_xyz is not None and updated_vox_xyz.shape[0] > 0:
      active_bbox_min = torch.min(updated_vox_xyz, dim = 0).values
      active_bbox_max = torch.max(updated_vox_xyz, dim = 0).values
      self.update_frontiers(active_bbox_min, active_bbox_max)

  def process_pointcloud(self,
                         pc_xyz: torch.FloatTensor,
                         origin_xyz: torch.FloatTensor) -> dict:
    """Update map geometry (occupancy + frontiers) from a point cloud.

    Decoupled-pipeline geometry entry point: builds occupancy directly from a
    world-frame point cloud with free space carved from the sensor origin.
    Semantics are attached separately via process_semantic_frame.

    Args:
      pc_xyz: Nx3 float tensor of points in world RDF coordinates.
      origin_xyz: Float tensor of size 3; sensor origin in world RDF used as
        the carving start point (e.g. body/LiDAR position at scan time).

    Returns:
      An update info dict (currently empty; mirrors process_posed_rgbd).
    """
    update_info = dict()
    pc_xyz = pc_xyz.to(self.device)
    origin_xyz = origin_xyz.to(self.device)

    vox_xyz, vox_occ = g3d.pointcloud_to_sparse_occupancy_voxels(
      pc_xyz, origin_xyz, self.vox_size,
      max_num_pts=self.max_pts_per_frame,
      max_num_empty_pts=self.max_empty_pts_per_frame,
      occ_thickness=self.occ_thickness)

    vox_xyz, vox_occ = self._clip_pc(vox_xyz, vox_occ)

    # [0, 1] to [-1, occ_observ_weight]
    vox_occ = vox_occ*self.occ_observ_weight-1

    if (self.debug_log_period > 0
        and self._debug_frame_idx % self.debug_log_period == 0):
      logger.info(
        "Mapper debug (pointcloud) frame=%d pc_xyz=%d vox_xyz=%d",
        self._debug_frame_idx, pc_xyz.shape[0], vox_xyz.shape[0])
    self._debug_frame_idx += 1

    self._vox_accum_cnt += 1
    self._occ_pruning_cnt += 1
    self._sem_pruning_cnt += 1

    if vox_xyz.shape[0] > 0:
      self._tmp_vox_occ.append(vox_occ)
      self._tmp_vox_xyz.append(vox_xyz)

    if self._vox_accum_cnt >= self.vox_accum_period:
      self._vox_accum_cnt = 0
      self._run_accumulation_step()

    return update_info

  def _get_occupied_voxel_centers(self) -> torch.FloatTensor:
    """Returns Nx3 occupied voxel centers from the occupancy VDB (or None).

    Note: pruned/merged tiles bigger than vox_size are represented by their
    center only (v1 approximation).
    """
    if self.occ_map_vdb.empty():
      return None
    pc_xyz_occ_size = rayfronts_cpp.occ_vdb2sizedpc(self.occ_map_vdb)
    occupied = pc_xyz_occ_size[pc_xyz_occ_size[:, -2] > 0]
    if occupied.shape[0] == 0:
      return None
    return occupied[:, :3].to(self.device)

  def process_semantic_frame(self,
                             rgb_img: torch.FloatTensor,
                             pose_4x4: torch.FloatTensor) -> dict:
    """Attach semantic features from an RGB keyframe to visible occupied voxels.

    Decoupled-pipeline semantics entry point. Occupied voxel centers are
    projected into the camera; a z-buffer built from the same centers filters
    occluded voxels; the surviving voxels get the encoder features (and rgb)
    sampled at their projected pixel, fused via the standard weighted
    aggregation.

    Args:
      rgb_img: 1x3xHxW float tensor with values in [0, 1].
      pose_4x4: 4x4 (or 1x4x4) float tensor; camera-to-world (RDF) pose.

    Returns:
      update info dict with "feat_img" when features were computed.
    """
    update_info = dict()
    assert rgb_img.shape[0] == 1, \
      "process_semantic_frame expects batch size 1"
    rgb_img = rgb_img.to(self.device)
    pose_4x4 = pose_4x4.reshape(4, 4).to(self.device)
    _, _, rH, rW = rgb_img.shape

    cand = self._get_occupied_voxel_centers()
    if cand is None:
      return update_info

    # Project candidates into the camera.
    cam_pts = g3d.transform_points(cand, torch.linalg.inv(pose_4x4))
    z = cam_pts[:, 2]
    front = (z > 0) & torch.isfinite(z)
    if not front.any():
      return update_info
    uv_h = (self.intrinsics_3x3 @ cam_pts[front].T).T
    u = (uv_h[:, 0] / uv_h[:, 2]).long()
    v = (uv_h[:, 1] / uv_h[:, 2]).long()
    in_bounds = (u >= 0) & (u < rW) & (v >= 0) & (v < rH)
    if not in_bounds.any():
      return update_info

    cand_f = cand[front][in_bounds]
    z_f = z[front][in_bounds]
    u = u[in_bounds]
    v = v[in_bounds]

    # Visibility: z-buffer the candidates themselves with their projected
    # voxel footprints as splats (isolated center pixels would let hidden
    # voxels peek through the gaps); keep voxels within tolerance of the
    # closest splat on their pixel.
    fx = self.intrinsics_3x3[0, 0]
    r_px = torch.ceil(0.5 * fx * self.vox_size / z_f).long()
    zbuf = g3d.splat_min_zbuffer(u, v, z_f, r_px, (rH, rW))
    tol = self.visibility_tolerance * self.vox_size
    visible = z_f <= (zbuf[v, u] + tol)
    if not visible.any():
      return update_info

    vis_vox = cand_f[visible]
    pix_flat = (v[visible] * rW + u[visible])

    # Encode features and sample at the projected pixels (same path as
    # process_posed_rgbd).
    feat_img = self.encoder.encode_image_to_feat_map(rgb_img)
    feat_img = self._proj_resize_feat_map(feat_img, rH, rW)
    update_info["feat_img"] = feat_img

    feat_img_flat = feat_img.permute(0, 2, 3, 1).reshape(-1, feat_img.shape[1])
    pts_feat = feat_img_flat[pix_flat]
    del feat_img_flat
    pts_rgb = rgb_img.permute(0, 2, 3, 1).reshape(-1, 3)[pix_flat]

    N = vis_vox.shape[0]
    pts_rgb_feat_cnt = torch.cat(
      (pts_rgb, pts_feat, torch.ones((N, 1), device=self.device)), dim=-1)

    self._tmp_pc_xyz.append(vis_vox)
    self._tmp_pc_rgb_feat_cnt.append(pts_rgb_feat_cnt)
    # Fuse immediately at keyframe rate; geometry accumulation is driven by
    # process_pointcloud independently.
    self.accum_semantic_voxels()

    self.update_semantic_coverage_frontiers(cand)

    if (self.debug_log_period > 0
        and self._debug_frame_idx % self.debug_log_period == 0):
      cov = self.semantic_coverage_frontiers
      logger.info(
        "Mapper debug (semantic frame) candidates=%d visible=%d "
        "sem_vox=%d coverage_frontiers=%d",
        cand.shape[0], N,
        0 if self.global_vox_xyz is None else self.global_vox_xyz.shape[0],
        0 if cov is None else cov.shape[0])

    return update_info

  def update_semantic_coverage_frontiers(
      self, occupied_centers: torch.FloatTensor) -> None:
    """Compute occupied-but-unlabeled voxels clustered as coverage frontiers.

    These mark surfaces the camera has not yet attached semantics to
    (geometrically known, semantically unknown).
    """
    if occupied_centers is None or occupied_centers.shape[0] == 0:
      self.semantic_coverage_frontiers = None
      self._unlabeled_occ_xyz = None
      return
    if self.global_vox_xyz is None or self.global_vox_xyz.shape[0] == 0:
      occ_cells = occupied_centers
      unlabeled_flag = torch.ones_like(occ_cells[:, 0:1])
    else:
      union, flag = g3d.intersect_voxels(
        occupied_centers, self.global_vox_xyz, self.vox_size)
      keep = flag >= 0  # 1 = occupied only (unlabeled), 0 = occupied+labeled
      occ_cells = union[keep]
      unlabeled_flag = (flag[keep] == 1).float().unsqueeze(-1)

    # Cache for class-frontier adjacency (condition 1).
    self._unlabeled_occ_xyz = occ_cells[unlabeled_flag.squeeze(-1) == 1]

    if occ_cells.shape[0] == 0 or unlabeled_flag.sum() == 0:
      self.semantic_coverage_frontiers = None
      return

    if self.fronti_subsampling > 1:
      # Cluster with per-cell total and unlabeled counts: flag a cell only if
      # unlabeled voxels are numerous AND the dominant share, so partially
      # labeled surfaces (e.g. obliquely viewed ground) are not flagged
      # wholesale.
      feat = torch.cat(
        [torch.ones_like(unlabeled_flag), unlabeled_flag], dim=-1)
      clustered, cnts = g3d.pointcloud_to_sparse_voxels(
        occ_cells, vox_size=self.vox_size*self.fronti_subsampling,
        feat_pc=feat, aggregation="sum")
      total = cnts[:, 0]
      unlabeled_cnt = cnts[:, 1]
      mask = ((unlabeled_cnt > self.fronti_subsampling_min_fronti) &
              (unlabeled_cnt / total.clamp(min=1) >=
               self.coverage_min_unlabeled_frac))
      clustered = clustered[mask]
      self.semantic_coverage_frontiers = \
        clustered if clustered.shape[0] > 0 else None
    else:
      self.semantic_coverage_frontiers = \
        occ_cells[unlabeled_flag.squeeze(-1) == 1]

  def update_class_frontiers(self,
                             query_feats: torch.FloatTensor,
                             query_labels: List[str],
                             compressed: bool = False) -> None:
    """Compute boundary frontiers of the selected semantic classes.

    A boundary voxel is a semantic voxel classified (argmax over the active
    query set, with a minimum probability) as one of class_frontier_classes
    that is adjacent (26-neighborhood) to either
    (1) an occupied-but-unlabeled voxel (cached by
        update_semantic_coverage_frontiers), or
    (2) a raw geometric frontier cell (empty bordering unobserved; maintained
        by update_frontiers). Frontier-adjacency naturally excludes sealed
        unknown space such as under-floor cells.
    Boundary voxels are clustered into self.class_frontiers (single merged
    layer).

    Args:
      query_feats: QxD float tensor of the active query features.
      query_labels: List of Q labels aligned with query_feats.
      compressed: Whether query_feats (and comparison) are in compressed
        feature space. Mirrors feature_query.
    """
    if len(self.class_frontier_classes) == 0:
      return
    if (self.global_vox_xyz is None or self.global_vox_xyz.shape[0] == 0 or
        query_feats is None or len(query_labels) == 0):
      self.class_frontiers = None
      return

    targets = set(self.class_frontier_classes)
    target_idx = [i for i, l in enumerate(query_labels) if l in targets]
    if len(target_idx) == 0:
      if not self._class_frontier_warned:
        logger.warning(
          "class_frontier_classes %s not found in query set %s; class "
          "frontiers disabled until those queries are added.",
          self.class_frontier_classes, list(query_labels))
        self._class_frontier_warned = True
      return

    # Classify semantic voxels by argmax over the query set (same feature
    # handling as feature_query).
    vox_feat = self.global_vox_feat
    if self.feat_compressor is not None and not compressed:
      vox_feat = self.feat_compressor.decompress(vox_feat)
    vox_feat = self.encoder.align_spatial_features_with_language(
      vox_feat.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)
    prob = compute_cos_sim(query_feats, vox_feat, softmax=True)  # Nv x Q
    best_prob, best_cls = prob.max(dim=-1)
    target_mask = torch.isin(
      best_cls, torch.tensor(target_idx, device=best_cls.device))
    if self.class_frontier_min_prob > 0:
      target_mask &= best_prob >= self.class_frontier_min_prob
    cls_xyz = self.global_vox_xyz[target_mask]
    self.class_voxels_xyz = cls_xyz if cls_xyz.shape[0] > 0 else None
    if cls_xyz.shape[0] == 0:
      self.class_frontiers = None
      return

    # Adjacency sets: (1) occupied-but-unlabeled, (2) raw frontier cells.
    adj_sets = []
    if (self._unlabeled_occ_xyz is not None and
        self._unlabeled_occ_xyz.shape[0] > 0):
      adj_sets.append(self._unlabeled_occ_xyz.to(cls_xyz.device))
    if self._frontiers_raw is not None and self._frontiers_raw.shape[0] > 0:
      adj_sets.append(self._frontiers_raw.to(cls_xyz.device))
    if len(adj_sets) == 0:
      self.class_frontiers = None
      return
    adj = g3d.pointcloud_to_sparse_voxels(
      torch.cat(adj_sets, dim=0), self.vox_size)
    if self.class_frontier_top_surface_only:
      # Surface-class mode: cells directly BENEATH a class voxel must not
      # flag it (the interior of a thick floor is unphotographable, not a
      # frontier). Dilate the adjacency set with lateral and downward offsets
      # only (up = -y), so only lateral/above unknown-ness creates boundary.
      o = torch.arange(-1, 2, device=adj.device,
                       dtype=adj.dtype) * self.vox_size
      ox, oy, oz = torch.meshgrid(o, o, o, indexing="xy")
      offs = torch.stack([ox, oy, oz], dim=-1).reshape(-1, 3)
      offs = offs[offs[:, 1] >= 0]  # drop upward (-y) offsets
      adj_dilated = g3d.pointcloud_to_sparse_voxels(
        (adj.reshape(-1, 1, 3) + offs.reshape(1, -1, 3)).reshape(-1, 3),
        self.vox_size)
    else:
      adj_dilated = g3d.dilate_sparse_voxels(adj, self.vox_size, r=1)

    # Class voxels that fall inside the dilated adjacency set are boundary.
    union, flag = g3d.intersect_voxels(cls_xyz, adj_dilated, self.vox_size)
    boundary = union[flag == 0]
    if boundary.shape[0] == 0:
      self.class_frontiers = None
      return

    if self.class_frontier_top_surface_only:
      # Keep only boundary voxels at the class top surface of their lateral
      # (x, z) column; drops boundary points beneath the class (e.g. voxels
      # inside a thick floor adjacent to its unphotographable interior).
      # World RDF: up = -y, so the column top is the minimum y.
      m = 2**20  # column hash multiplier (lateral index range guard)
      cls_col = torch.round(cls_xyz[:, [0, 2]] / self.vox_size).long()
      cls_h = cls_col[:, 0] * m + cls_col[:, 1]
      uniq_h, inv = torch.unique(cls_h, return_inverse=True)
      top_y = torch.full((uniq_h.shape[0],), torch.inf,
                         device=cls_xyz.device, dtype=cls_xyz.dtype)
      top_y.scatter_reduce_(0, inv, cls_xyz[:, 1], reduce="amin",
                            include_self=False)
      b_col = torch.round(boundary[:, [0, 2]] / self.vox_size).long()
      b_h = b_col[:, 0] * m + b_col[:, 1]
      pos = torch.searchsorted(uniq_h, b_h).clamp(max=uniq_h.shape[0] - 1)
      col_found = uniq_h[pos] == b_h
      # Allow only the top voxel layer (0.6 * vox margin).
      boundary = boundary[col_found &
                          (boundary[:, 1] <= top_y[pos] + 0.6*self.vox_size)]
      if boundary.shape[0] == 0:
        self.class_frontiers = None
        return

    # Cluster boundary voxels into frontier points.
    if self.class_frontier_subsampling > 1:
      feat = torch.ones_like(boundary[:, 0:1])
      clustered, cnt = g3d.pointcloud_to_sparse_voxels(
        boundary, vox_size=self.vox_size*self.class_frontier_subsampling,
        feat_pc=feat, aggregation="sum")
      clustered = clustered[cnt[:, 0] >= self.class_frontier_min_cnt]
      self.class_frontiers = clustered if clustered.shape[0] > 0 else None
    else:
      self.class_frontiers = boundary

  def cast_semantic_rays(self) -> None:
    """Cast semantic rays accumulated in the temporary buffers onto frontiers.
    """
    for l in self._tmp_ray_dir:
      if len(l) > 0:
        break
    else:
      # No rays to cast
      return
    ray_dir = torch.cat(self._tmp_ray_dir, dim = 0) # Nx3
    self._tmp_ray_dir.clear()
    ray_orig = torch.cat(self._tmp_ray_orig, dim = 0) # Nx3
    self._tmp_ray_orig.clear()
    ray_feat = torch.cat(self._tmp_ray_feat, dim = 0) # Nx3
    self._tmp_ray_feat.clear()
    N = ray_orig.shape[0]
    M = self.frontiers.shape[0] if self.frontiers is not None else 0
    if N > 0 and M > 0:
      frontier_vox_size = self.vox_size*self.fronti_subsampling
      frontier_dir = self.frontiers.reshape(M,1,3) - ray_orig.reshape(1,N,3)
      dot_prod = frontier_dir.reshape(M, N, 1, 3) @ \
        ray_dir.reshape(1, N, 3, 1)
      dot_prod = dot_prod.squeeze(-1).squeeze(-1)
      closest_pts = dot_prod.reshape(M, N, 1) * ray_dir.reshape(1, N, 3) + \
        ray_orig.reshape(1, N, 3)

      # MxN distance matrix where [i,j] represents the shortest distance
      # between frontier i and ray-origin j.
      dist = torch.norm(frontier_dir, dim=-1)

      if not self.zero_depth_mode:
        # MxN distance matrix where [i,j] represents the shortest distance
        # between frontier i and ray j.
        ortho_dist = torch.norm(
          closest_pts - self.frontiers.reshape(M, 1, 3), dim=-1)

        # In [0-1]
        cost_matrix = (ortho_dist/ortho_dist.max() + dist/dist.max()) / 2

        # Only consider distances where frontier is in front of ray,
        # frontier is at a minimum distance from ray origin, and orthogonal
        # distance is at a maximum distance.
        cost_matrix[(dot_prod <= 0) |
                    (dist < frontier_vox_size*2) |
                    (ortho_dist > frontier_vox_size)] = torch.inf

        if self.max_depth_sensing > 0:
          cost_matrix[dist > self.max_depth_sensing*3] = torch.inf

      else:
        # If global_vox_xyz is None then we are in 0 depth mode
        cost_matrix = dist

      # Match rays with frontiers
      # TODO: Add option for interpolation instead of assigning each ray to a
      # single frontier.
      min_cost, min_cost_ind = torch.min(cost_matrix, dim=0)
      mask = min_cost.isfinite()
      min_cost = min_cost[mask]
      min_cost_ind = min_cost_ind[mask]
      ray_orig = ray_orig[mask, :]
      ray_dir = ray_dir[mask, :]
      ray_feat = ray_feat[mask, :]

      if self.ray_tracing and not self.zero_depth_mode and \
        ray_orig.shape[0] > 0:
        # Now we have assigned a ray to the best frontier candidate, let us
        # make sure that ray is not occluded by observed occupied voxels or
        # potentially occupied unobserved voxels.

        # We select the end point for each ray which is the closest point to
        # the selected frontier on that ray.
        closest_pts = closest_pts[:, mask, :]
        closest_pts = closest_pts[min_cost_ind, torch.arange(
          min_cost_ind.shape[0], device=self.device)]

        if self.max_depth_sensing > 0:
          max_dist = self.max_depth_sensing * 3
        else:
          max_dist = torch.norm(closest_pts - ray_orig, dim=-1).max().item()

        max_steps = int(math.ceil(max_dist/self.vox_size))
        marching_rays = torch.clone(ray_orig)
        # Marching mask
        m = torch.ones_like(ray_orig[:, 0], dtype=torch.bool)

        for s in range(max_steps):
          marching_rays[m] = marching_rays[m] + self.vox_size*ray_dir[m]

          occ = rayfronts_cpp.query_occ(
            self.occ_map_vdb, marching_rays[m].cpu()).to(self.device)
          m[m.nonzero()[occ >= 0]] = False
          m &= (torch.norm(marching_rays - closest_pts, dim=-1) >
                self.vox_size)

        # Valid ray mask
        vrm = torch.norm(marching_rays - closest_pts, dim=-1) <= self.vox_size
        min_cost = min_cost[vrm]
        min_cost_ind = min_cost_ind[vrm]
        ray_orig = ray_orig[vrm, :]
        ray_dir = ray_dir[vrm, :]
        ray_feat = ray_feat[vrm, :]

        ## Below is more parallel but extremely expensive for memory
        # coefs = torch.linspace(0, 1, max_steps, device=self.device)
        # coefs = coefs.reshape(1, -1, 1)
        # ray_trace_xyz = coefs * ray_orig.unsqueeze(1) + \
        #   (1-coefs) * closest_pts.unsqueeze(1)

      # Change the ray's origin to its matching frontier.
      ray_orig = self.frontiers[min_cost_ind]

      # TODO: Partition angle space in a better way
      _, theta, phi = g3d.cartesian_to_spherical(
        ray_dir[:, 0], ray_dir[:, 1], ray_dir[:, 2])

      ray_orig_angle = torch.cat(
        [ray_orig, torch.rad2deg(theta).unsqueeze(-1),
        torch.rad2deg(phi).unsqueeze(-1)], dim=-1)

      ray_weights = (1-min_cost).unsqueeze(-1)
      ray_weights /= torch.sum(ray_weights)
      if self.global_rays_orig_angles is None:
        ray_orig_angle, ray_feat_cnt = g3d.bin_rays(
          ray_orig_angle, self.vox_size, self.angle_bin_size,
          torch.cat((ray_feat, ray_weights), dim=-1),
          aggregation="weighted_mean")

        self.global_rays_orig_angles = ray_orig_angle
        self.global_rays_feats_cnt = ray_feat_cnt
      else:
        self.global_rays_orig_angles, self.global_rays_feats_cnt = \
          g3d.add_weighted_binned_rays(
            self.global_rays_orig_angles,
            self.global_rays_feats_cnt,
            ray_orig_angle,
            torch.cat((ray_feat, ray_weights), dim=-1),
            vox_size=self.vox_size,
            bin_size=self.angle_bin_size,
          )

  def prune_semantic_rays(self) -> None:
    """Remove semantic rays that are no longer on a frontier.
    
    If ray_tracing is enabled, then the removed rays are added back to the
    accumulation buffer to be recast in the next iteration.
    """
    if self.global_rays_orig_angles is None:
      return
    # if a ray origin does not lie on a frontier then that ray will
    # be removed.
    M = self.frontiers.shape[0]
    N = self.global_rays_orig_angles.shape[0]
    dist = torch.norm(
      self.frontiers.reshape(M, 1, 3) -
      self.global_rays_orig_angles[:, :3].reshape(1, N, 3), p=1, dim=-1)
    mask = torch.any(dist < 1e-6, dim=0)
    if self.ray_tracing:
      # If ray tracing is enabled then we push these rays instead of
      # destroying them
      re_shoot_orig_angles =  self.global_rays_orig_angles[~mask]
      if re_shoot_orig_angles.shape[0] > 0:
        re_shoot_feats_cnt =  self.global_rays_feats_cnt[~mask]
        ray_orig = re_shoot_orig_angles[:, :3]
        angles = torch.deg2rad(re_shoot_orig_angles[:, 3:])
        ray_dir = torch.stack(
          g3d.spherical_to_cartesian(1, angles[:, 0], angles[:, 1]), dim=-1)
        self._tmp_ray_dir.append(ray_dir)
        self._tmp_ray_orig.append(ray_orig)
        self._tmp_ray_feat.append(re_shoot_feats_cnt[:, :-1])


    self.global_rays_orig_angles = self.global_rays_orig_angles[mask]
    self.global_rays_feats_cnt = self.global_rays_feats_cnt[mask]

  def accum_occ_voxels(self) -> torch.FloatTensor:
    """Accumulate the temporarilly gathered occupancy voxels.
    
    Returns:
      Voxels that were updated (With repetitions)
    """
    if len(self._tmp_vox_xyz) == 0:
      return
    vox_xyz = torch.cat(self._tmp_vox_xyz, dim = 0)
    self._tmp_vox_xyz.clear()
    vox_occ = torch.cat(self._tmp_vox_occ, dim = 0)
    self._tmp_vox_occ.clear()
    if self.sem_pruning_period > 0:
      self._tmp_vox_xyz_since_prune.append(vox_xyz)

    rayfronts_cpp.occ_pc2vdb(
      self.occ_map_vdb, vox_xyz.cpu(), vox_occ.cpu().squeeze(-1),
      self.max_empty_cnt, self.max_occ_cnt)
    return vox_xyz

  def accum_semantic_voxels(self) -> None:
    """Accumulate the temporarilly gathered semantic points into voxels."""
    if len(self._tmp_pc_xyz) == 0:
      return
    pc_xyz = torch.cat(self._tmp_pc_xyz, dim = 0)
    self._tmp_pc_xyz.clear()
    pc_rgb_feat_cnt = torch.cat(self._tmp_pc_rgb_feat_cnt, dim = 0)
    self._tmp_pc_rgb_feat_cnt.clear()

    if self.global_vox_xyz is None:
      vox_xyz, vox_rgb_feat_cnt, vox_cnt = g3d.pointcloud_to_sparse_voxels(
        pc_xyz, feat_pc=pc_rgb_feat_cnt, vox_size=self.vox_size,
        return_counts=True)

      vox_rgb_feat_cnt[:, -1] = vox_cnt.squeeze()
      self.global_vox_xyz = vox_xyz
      self.global_vox_rgb_feat_cnt = vox_rgb_feat_cnt
    else:
      self.global_vox_xyz, self.global_vox_rgb_feat_cnt = \
        g3d.add_weighted_sparse_voxels(
          self.global_vox_xyz,
          self.global_vox_rgb_feat_cnt,
          pc_xyz,
          pc_rgb_feat_cnt,
          vox_size=self.vox_size
        )

  def prune_semantic_voxels(self, updated_pts_xyz) -> None:
    """Remove semantic voxels that are no longer occupied.
    
    Args:
      updated_pts_xyz: (Nx3) Float tensor describing the voxels/points that
        have been updated. Only these points will be considered for removal.
    """
    if self.global_vox_xyz is None or self.global_vox_xyz.shape[0] == 0:
      return

    updated_vox_xyz = g3d.pointcloud_to_sparse_voxels(
      updated_pts_xyz, vox_size=self.vox_size)
    updated_vox_occ = rayfronts_cpp.query_occ(
      self.occ_map_vdb, updated_vox_xyz.cpu()).to(self.device)

    vox_xyz_to_remove = updated_vox_xyz[updated_vox_occ.squeeze(-1) <=
                                        self.sem_pruning_thresh]

    self.global_vox_xyz, flag = g3d.intersect_voxels(
      self.global_vox_xyz, vox_xyz_to_remove, self.vox_size)

    self.global_vox_xyz = self.global_vox_xyz[flag == 1]

    # Strong assumption here that the original global_vox_xyz is sorted !
    # and that the produced global_vox_xyz is also sorted.
    # If both the first input and the output are sorted then the filtered flag
    # will be aligned with the first input.
    # TODO: Double check and have stronger guarantees / fail-safes
    self.global_vox_rgb_feat_cnt = \
      self.global_vox_rgb_feat_cnt[flag[flag >= 0] == 1]

  def update_frontiers(self, active_bbox_min, active_bbox_max) -> None:
    frontiers_update = rayfronts_cpp.parallel_filter_cells_in_bbox(
      self.occ_map_vdb,
      cell_type_to_iterate = rayfronts_cpp.CellType.Empty,
      world_bbox_min = rayfronts_cpp.Vec3d(*active_bbox_min),
      world_bbox_max = rayfronts_cpp.Vec3d(*active_bbox_max),
      neighborhood_r = self.fronti_neighborhood_r,
      min_unobserved = self.fronti_min_unobserved,
      min_empty = self.fronti_min_empty,
      min_occupied = self.fronti_min_occupied,
      return_cnts = self.keep_frontier_neighbor_cnts,
    ).to(self.device)

    # Keep raw (pre-subsampling) frontier cells for class-frontier adjacency,
    # with the same replace-inside-active-window update as below.
    raw_update = frontiers_update[:, :3].clone()
    if self._frontiers_raw is None:
      self._frontiers_raw = raw_update
    else:
      outside_mask = torch.logical_or(
        torch.any(self._frontiers_raw < active_bbox_min, dim=-1),
        torch.any(self._frontiers_raw > active_bbox_max, dim=-1))
      self._frontiers_raw = torch.cat(
        [self._frontiers_raw[outside_mask], raw_update], dim=0)

    # Subsample frontiers using voxel grid
    if self.fronti_subsampling > 1:
      feat = torch.ones_like(frontiers_update[:, 0:1])
      if self.keep_frontier_neighbor_cnts:
        feat = torch.cat([feat, frontiers_update[:, 3:]], dim=-1)
      frontiers_update, cnt = g3d.pointcloud_to_sparse_voxels(
        frontiers_update[:, :3], vox_size=self.vox_size*self.fronti_subsampling,
        feat_pc=feat, aggregation="sum")
      mask = cnt[:, 0] > self.fronti_subsampling_min_fronti
      frontiers_update = frontiers_update[mask]
      cnt = cnt[mask]
      cnts_update = cnt[:, 1:] / cnt[:, :1] if self.keep_frontier_neighbor_cnts \
        else None
    else:
      cnts_update = frontiers_update[:, 3:] if self.keep_frontier_neighbor_cnts \
        else None
      frontiers_update = frontiers_update[:, :3]

    # Update global frontiers
    if self.frontiers is None:
      self.frontiers = frontiers_update
      self.frontiers_neighbor_cnts = cnts_update
    else:
      # Replace old frontiers in active window with updated frontiers
      outside_mask = torch.logical_or(
        torch.any(self.frontiers < active_bbox_min, dim=-1),
        torch.any(self.frontiers > active_bbox_max, dim=-1))
      self.frontiers = self.frontiers[outside_mask]
      self.frontiers = torch.cat([self.frontiers, frontiers_update], dim=0)
      if self.keep_frontier_neighbor_cnts:
        self.frontiers_neighbor_cnts = torch.cat(
          [self.frontiers_neighbor_cnts[outside_mask], cnts_update], dim=0)

  def compute_inferred_directions(self):
    """Infer frontier directions based on occupancy map."""
    frontiers = self.global_rays_orig_angles[:, :3]
    a = torch.arange(-1, 2, 1, device=self.device, dtype=torch.float)
    a *= self.vox_size
    window = torch.stack(torch.meshgrid(a, a, a, indexing="xy"),
                          dim=-1).reshape(-1, 3)
    window = window[torch.arange(0, 27) != 13] # Remove center
    query_pts = (frontiers.unsqueeze(1) + window.unsqueeze(0))
    query_pts = query_pts.reshape(-1, 3)
    occ = rayfronts_cpp.query_occ(self.occ_map_vdb, query_pts.cpu())
    occ = occ.to(self.device).reshape(-1, 26).float()
    weight = -torch.ones_like(occ)
    weight[occ==0] = 1
    dirs = weight @ window
    dirs = torch.nn.functional.normalize(dirs, dim=-1)
    _, theta, phi = g3d.cartesian_to_spherical(
      dirs[:, 0], dirs[:, 1], dirs[:, 2])
    rays_orig_angles = torch.cat(
      [frontiers, torch.rad2deg(theta).unsqueeze(-1),
       torch.rad2deg(phi).unsqueeze(-1)], dim=-1)

    self.global_rays_orig_angles = rays_orig_angles

  @override
  def feature_query(self,
                    feat_query: torch.FloatTensor,
                    softmax: bool = False,
                    compressed: bool = True)-> dict:
    if self.is_empty():
      return

    r = dict()
    # Query semantic voxels
    if self.global_vox_xyz is not None:
      vox_feat = self.global_vox_feat
      if self.feat_compressor is not None and not compressed:
        vox_feat = self.feat_compressor.decompress(vox_feat)
      vox_feat = self.encoder.align_spatial_features_with_language(
        vox_feat.unsqueeze(-1).unsqueeze(-1)
      ).squeeze(-1).squeeze(-1)
      r["vox_xyz"] = self.global_vox_xyz
      r["vox_sim"] = compute_cos_sim(feat_query, vox_feat, softmax=softmax).T

    # Query semantic ray frontiers
    if self.global_rays_orig_angles is not None and \
       self.global_rays_orig_angles.shape[0] > 0:

      rays_feat = self.global_rays_feat
      if self.feat_compressor is not None and not compressed:
        rays_feat = self.feat_compressor.decompress(rays_feat)
      rays_feat = self.encoder.align_spatial_features_with_language(
        rays_feat.unsqueeze(-1).unsqueeze(-1)
      ).squeeze(-1).squeeze(-1)
      r["ray_orig_angles"] = self.global_rays_orig_angles
      r["ray_sim"] = compute_cos_sim(feat_query, rays_feat, softmax=softmax).T

    return r

  @override
  def _vis_height_mask(self, xyz: torch.FloatTensor):
    """TEMP(viz-only): boolean mask for self.vis_height_range (up = -y).

    Returns None when the filter is disabled (vis_height_range is None).
    """
    if self.vis_height_range is None:
      return None
    height = -xyz[:, 1]
    return ((height >= self.vis_height_range[0]) &
            (height <= self.vis_height_range[1]))

  def vis_map(self) -> None:
    if self.visualizer is None or self.is_empty():
      return

    # Vis semantic voxels
    if self.global_vox_xyz is not None and self.global_vox_xyz.shape[0] > 0:
      m = self._vis_height_mask(self.global_vox_xyz)
      vox_xyz = self.global_vox_xyz if m is None else self.global_vox_xyz[m]
      if vox_xyz.shape[0] > 0:
        vox_rgb = self.global_vox_rgb if m is None else self.global_vox_rgb[m]
        self.visualizer.log_pc(vox_xyz, vox_rgb, layer="voxel_rgb")
        if self.encoder is not None:
          vox_feat = self.global_vox_feat if m is None \
            else self.global_vox_feat[m]
          self.visualizer.log_feature_pc(
            vox_xyz, vox_feat, layer="voxel_feature")

        vox_cnt = self.global_vox_cnt if m is None else self.global_vox_cnt[m]
        log_hit_count = torch.log2(vox_cnt.squeeze(-1))
        self.visualizer.log_heat_pc(vox_xyz, log_hit_count,
                                    layer="voxel_log_hit_count")

    # Vis occupancy voxels
    if not self.occ_map_vdb.empty():
      pc_xyz_occ_size = rayfronts_cpp.occ_vdb2sizedpc(self.occ_map_vdb)
      m = self._vis_height_mask(pc_xyz_occ_size[:, :3])
      if m is not None:
        pc_xyz_occ_size = pc_xyz_occ_size[m]

      if pc_xyz_occ_size.shape[0] > 0:
        self.visualizer.log_occ_pc(
          pc_xyz_occ_size[:, :3],
          torch.clamp(pc_xyz_occ_size[:, -2:-1], min=-1, max=1),
          layer="voxel_occ"
        )

        tiles = pc_xyz_occ_size[pc_xyz_occ_size[:, -1] > self.vox_size, :]
        if tiles.shape[0] > 0:
          self.visualizer.log_occ_pc(
            tiles[:, :3],
            torch.clamp(tiles[:, -2:-1], min=-1, max=1),
            tiles[:, -1:],
            layer="voxel_occ_tiles"
          )

    # Vis frontiers
    if self.frontiers is not None and self.frontiers.shape[0] > 0:
      fronti_rgb = None
      if self.frontiers_neighbor_cnts is not None:
        fronti_rgb = self.frontiers_neighbor_cnts / \
          self.frontiers_neighbor_cnts.max(dim=0).values.clamp(min=1)
      self.visualizer.log_pc(self.frontiers, fronti_rgb, layer="frontiers")

    # Vis semantic coverage frontiers (occupied but unlabeled; decoupled
    # pipeline only). Fixed orange to distinguish from geometric frontiers.
    if (self.semantic_coverage_frontiers is not None and
        self.semantic_coverage_frontiers.shape[0] > 0):
      cov = self.semantic_coverage_frontiers
      m = self._vis_height_mask(cov)
      if m is not None:
        cov = cov[m]
      if cov.shape[0] > 0:
        cov_rgb = torch.tensor([[1.0, 0.55, 0.0]], device=cov.device).expand(
          cov.shape[0], 3)
        self.visualizer.log_pc(cov, cov_rgb,
                               layer="semantic_coverage_frontiers")

    # Vis class-specific frontiers (boundary of the selected semantic classes
    # against unlabeled/unknown space). Fixed magenta.
    if self.class_frontiers is not None and self.class_frontiers.shape[0] > 0:
      cf = self.class_frontiers
      m = self._vis_height_mask(cf)
      if m is not None:
        cf = cf[m]
      if cf.shape[0] > 0:
        cf_rgb = torch.tensor([[0.85, 0.1, 0.9]], device=cf.device).expand(
          cf.shape[0], 3)
        self.visualizer.log_pc(cf, cf_rgb, layer="class_frontiers")

    # Vis rays
    if (self.global_rays_orig_angles is not None and
        self.global_rays_orig_angles.shape[0] > 0):

      if self.angle_bin_size >= 360 and not self.infer_direction:
        self.visualizer.log_feature_pc(self.global_rays_orig_angles[:, :3],
                                       self.global_rays_feat,
                                       layer="semantic_frontiers")
      else:
        ray_orig = self.global_rays_orig_angles[:, :3]
        angles = torch.deg2rad(self.global_rays_orig_angles[:, 3:])
        ray_dir = torch.stack(
          g3d.spherical_to_cartesian(1, angles[:, 0], angles[:, 1]), dim=-1)
        self.visualizer.log_feature_arr(ray_orig, ray_dir,
                                        self.global_rays_feat,
                                        layer="semantic_ray_frontiers")

  @override
  def vis_update(self, **kwargs) -> None:
    if "feat_img" in kwargs:
      self.visualizer.log_feature_img(kwargs["feat_img"][-1].permute(1, 2, 0))

  @override
  def vis_query_result(self,
                       query_results: dict,
                       vis_labels: List[str] = None,
                       vis_colors: Dict[str, str] = None,
                       vis_thresh: float = 0) -> None:
    if query_results is None:
      return

    # Vis voxel results
    if "vox_sim" in query_results:
      vox_xyz = query_results["vox_xyz"]
      vox_sim = query_results["vox_sim"]
      for q in range(vox_sim.shape[0]):
        kwargs = dict()
        label = vis_labels[q]
        if vis_colors is not None and label in vis_colors.keys():
          kwargs["high_color"] = vis_colors[label]
          kwargs["low_color"] = (0, 0, 0)
        self.visualizer.log_heat_pc(
          vox_xyz, vox_sim[q, :],
          layer=f"queries/{label.replace(' ', '_').replace('/', '_')}/voxels",
          vis_thresh=vis_thresh,
          **kwargs)

    # Vis rays
    if "ray_sim" in query_results:
      ray_sim = query_results["ray_sim"]
      ray_orig = query_results["ray_orig_angles"][:, :3]
      angles = torch.deg2rad(query_results["ray_orig_angles"][:, 3:])
      ray_dir = torch.stack(
        g3d.spherical_to_cartesian(1, angles[:, 0], angles[:, 1]), dim=-1)

      for q in range(ray_sim.shape[0]):
        kwargs = dict()
        label = vis_labels[q]
        if vis_colors is not None and label in vis_colors and \
            vis_colors[label] is not None:

          kwargs["high_color"] = vis_colors[label]
          kwargs["low_color"] = vis_colors[label]

        if self.angle_bin_size >= 360 and not self.infer_direction:
          self.visualizer.log_heat_pc(
            ray_orig, ray_sim[q, :],
            layer=f"queries/{label.replace(' ', '_').replace('/', '_')}/frontiers",
            vis_thresh=vis_thresh,
            **kwargs)
        else:
          self.visualizer.log_heat_arrows(
            ray_orig, ray_dir, ray_sim[q, :],
            layer=f"queries/{label.replace(' ', '_').replace('/', '_')}/rays",
            vis_thresh=vis_thresh,
            **kwargs)
