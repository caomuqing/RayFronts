"""Frontier-driven exploration planner that runs on top of the mapping server.

Launching this module also launches the mapping process:

  python3 -m rayfronts.exploration_planner \
    --config-dir experiments/preset_configs \
    --config-name starlingmax_decoupled_bag

Behaviour (per planning cycle):
  1. Read the robot pose and the current frontier points (class frontiers by
     default).
  2. Rank frontiers by cost = distance_weight * distance
     + heading_weight * |heading change to face the frontier|.
  3. For the best frontier, search a safe exploration pose: a position
     hover_height above one of the selected-class voxels, whose surrounding
     sphere of safety_radius is fully known-empty, and which has an
     occlusion-free line of sight to the frontier.
  4. If no such pose exists the frontier is blacklisted (removed from future
     consideration) and the next-best frontier is tried.

The selected goal is visualized (rerun) and published as a bare
geometry_msgs/Pose in the pose-topic (NED) world frame on goal_topic. Goal
publishing is feedback-driven: a new goal is selected only after the
outstanding one is reported reached or failed on goal_status_topic
(std_msgs/Int8: 0 in_progress, 1 reached, 2 failed); status messages
received earlier than status_min_delay seconds after the goal was published
are ignored as stale feedback about the previous goal.
"""

import logging
import math
import signal
import threading
import time
from functools import partial

import torch
import numpy as np
import hydra

from rayfronts import geometry3d as g3d
from rayfronts.mapping_server import MappingServer, signal_handler
# rayfronts_cpp is imported (with its build path setup) by the mapper module.
from rayfronts.mapping.semantic_ray_frontiers_map import rayfronts_cpp

try:
  from geometry_msgs.msg import Pose
  from std_msgs.msg import Int8
  from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
  from scipy.spatial.transform import Rotation
except ModuleNotFoundError:
  Pose = None

logger = logging.getLogger(__name__)

# /goal_reach_status convention (std_msgs/Int8):
GOAL_IN_PROGRESS = 0
GOAL_REACHED = 1
GOAL_FAILED = 2


def _wrap_angle(a):
  """Wraps angle(s) to [-pi, pi]."""
  return (a + math.pi) % (2 * math.pi) - math.pi


class ExplorationPlanner:
  """Selects frontiers and safe observation poses on top of a MappingServer.

  All map access is guarded by server.map_lock so it can run concurrently
  with the mapping loop. Core methods (select order, pose search) are pure
  given their inputs and can be used without ROS for testing.
  """

  def __init__(self, cfg, server: MappingServer):
    """
    Args:
      cfg: exploration config node (see configs/default.yaml `exploration`).
      server: A constructed MappingServer whose mapper exposes frontier
        tensors (SemanticRayFrontiersMap).
    """
    self.cfg = cfg
    self.server = server
    self.mapper = server.mapper

    self.distance_weight = float(cfg.distance_weight)
    self.heading_weight = float(cfg.heading_weight)
    self.hover_height = float(cfg.hover_height)
    self.safety_radius = float(cfg.safety_radius)
    self.frontier_source = str(cfg.frontier_source)
    self.candidate_search_radius = float(cfg.candidate_search_radius)
    self.view_min_dist = float(cfg.view_min_dist)
    self.view_max_dist = float(cfg.view_max_dist)
    # Camera pitch below the horizon and allowed deviation from it (vertical
    # FOV half-angle) for the frontier to actually be viewable. When
    # view_max_elevation_deg is null it is derived from the camera intrinsics
    # as half the vertical FOV: atan((H/2) / fy).
    self._view_pitch = math.radians(float(cfg.view_pitch_deg))
    mev = cfg.view_max_elevation_deg
    if mev is None:
      mev = self._derive_half_vfov_deg()
      if mev is None:
        mev = 25.0
        logger.warning(
          "view_max_elevation_deg: could not derive from intrinsics; "
          "falling back to %.1f deg.", mev)
      else:
        logger.info(
          "view_max_elevation_deg derived from intrinsics: %.1f deg "
          "(half vertical FOV).", mev)
    # Optional inflation of the viewable elevation band. >1 pretends the
    # camera FOV is wider than it is, admitting closer viewpoints (min
    # standoff = hover_height / tan(pitch + max_elevation)) at the cost of
    # the frontier possibly falling slightly outside the actual image.
    fov_scale = float(cfg.get("view_fov_scale", 1.0))
    if fov_scale != 1.0:
      mev = float(mev) * fov_scale
      logger.info(
        "view_max_elevation_deg scaled by %.2f -> %.1f deg.", fov_scale, mev)
    self._view_max_elev = math.radians(float(mev))
    self.plan_period = float(cfg.plan_period)
    self.max_attempts_per_cycle = int(cfg.max_attempts_per_cycle)
    self.max_candidates = int(cfg.max_candidates)
    self.safety_allow_unknown = bool(cfg.get("safety_allow_unknown", False))
    self.frontier_proximity_weight = float(
      cfg.get("frontier_proximity_weight", 0.0))

    self.vox_size = float(self.mapper.vox_size)

    # 2D information grid: dict {(ix, iz): [state, confidence, y, gnd_votes,
    # obs_votes]} over the horizontal world plane (cell centers at
    # ix*grid_cell_size, iz*grid_cell_size). Cells accumulate per-voxel votes
    # across keyframes; state is 2 = obstacle iff obs/(obs+gnd) >=
    # obstacle_min_frac (with at least one obstacle vote), else 1 =
    # non-obstacle; unknown cells are absent. Confidence is the winning
    # state's vote fraction. y is display-only.
    self.info_grid = {}
    self.grid_cell_size = float(cfg.grid_cell_size)
    self.obstacle_max_height_voxels = int(cfg.obstacle_max_height_voxels)
    self.obstacle_min_frac = float(cfg.obstacle_min_frac)
    self._grid_last_seq = 0

    # Exploration xy bounds (pose-topic world frame) -> world-RDF AABB.
    # Restricts class-frontier generation (installed on the mapper) and the
    # info grid votes. None = unbounded.
    self.bounds_rdf = self._compute_bounds_rdf(cfg)
    if self.bounds_rdf is not None:
      self.mapper.class_frontier_bounds = self.bounds_rdf
      logger.info(
        "Exploration bounds active: x [%s, %s], y [%s, %s] (pose frame).",
        cfg.xmin, cfg.xmax, cfg.ymin, cfg.ymax)
    # Blacklist keys are rounded to the frontier cluster grid so recomputed
    # frontiers at the same location stay removed. Bans expire with
    # exponential backoff (ban = base * 2^(fails-1), capped) so a frontier
    # that had no safe viewpoint is rechecked once the map matures; the fail
    # count survives expiry, so repeat offenders get progressively longer
    # bans. dict: key -> [fail_count, ban_expiry_walltime].
    self._bl_grid = self.vox_size * float(
      getattr(self.mapper, "class_frontier_subsampling", 3))
    self._blacklist = {}
    self.blacklist_base_ban_s = float(cfg.get("blacklist_base_ban_s", 15.0))
    self.blacklist_max_ban_s = float(cfg.get("blacklist_max_ban_s", 240.0))

    # Precompute safety-sphere offsets at voxel resolution.
    n = max(1, int(math.ceil(self.safety_radius / self.vox_size)))
    o = torch.arange(-n, n + 1, dtype=torch.float) * self.vox_size
    offs = torch.stack(torch.meshgrid(o, o, o, indexing="xy"),
                       dim=-1).reshape(-1, 3)
    self._sphere_offsets = offs[offs.norm(dim=-1) <= self.safety_radius]

    self.current_goal = None  # (pose_4x4 world RDF, frontier_xyz)

    self._stop = threading.Event()
    self._thread = threading.Thread(
      target=self._loop, name="rayfronts_exploration_planner", daemon=True)

    # Goal output / feedback: the goal is published as a bare
    # geometry_msgs/Pose in the pose-topic (NED) frame on goal_topic. A new
    # goal is published only when the previous one is reported reached or
    # failed on goal_status_topic (std_msgs/Int8: 0 in_progress, 1 reached,
    # 2 failed), ignoring statuses received earlier than status_min_delay
    # seconds after the goal was published (stale feedback for the previous
    # goal). Without ROS, the planner falls back to replanning every cycle.
    self.status_min_delay = float(cfg.get("status_min_delay", 0.3))
    self._status_lock = threading.Lock()
    self._latest_status = None      # (value, reception wall-time)
    self._last_goal_pub_time = None  # wall-time of the outstanding goal
    self._last_goal_key = None       # blacklist key of the outstanding goal
    self._goal_pub = None
    # External enable switch for goal publishing (std_msgs/Int8 on
    # goal_publish_allow_topic: 1 = allow, 0 = suppress). Only gates
    # /goal_point output; mapping, visualization and feedback handling are
    # unaffected. Defaults to allowed until a message says otherwise.
    self._goal_publish_allowed = False
    ds = getattr(server, "dataset", None)
    if (Pose is not None and ds is not None
        and getattr(ds, "_rosnode", None) is not None):
      # Latched (transient_local) so late-joining subscribers receive the
      # outstanding goal; must also match subscribers requesting
      # transient_local durability (e.g. the onboard goal follower).
      goal_qos = QoSProfile(
        depth=10,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL)
      self._goal_pub = ds._rosnode.create_publisher(
        Pose, str(cfg.goal_topic), goal_qos)
      ds._rosnode.create_subscription(
        Int8, str(cfg.get("goal_status_topic", "/goal_reach_status")),
        self._on_goal_status, 10)
      ds._rosnode.create_subscription(
        Int8, str(cfg.get("goal_publish_allow_topic", "/goal_publish_allow")),
        self._on_goal_publish_allow, 10)
      logger.info("Exploration goal publisher on %s (Pose, NED frame); "
                  "listening for reach status on %s; goal publishing "
                  "enable switch on %s.", cfg.goal_topic,
                  cfg.get("goal_status_topic", "/goal_reach_status"),
                  cfg.get("goal_publish_allow_topic", "/goal_publish_allow"))

  def _derive_half_vfov_deg(self):
    """Half vertical FOV in degrees from the projection intrinsics, or None.

    Uses the mapper's (rectified, depth-registered) intrinsics and the
    dataset's registered image height: VFOV/2 = atan((H/2) / fy).
    """
    K = getattr(self.mapper, "intrinsics_3x3", None)
    ds = getattr(self.server, "dataset", None)
    h = getattr(ds, "depth_h", 0) if ds is not None else 0
    if K is None or h is None or h <= 0:
      return None
    fy = float(K[1, 1])
    return math.degrees(math.atan2(h / 2.0, fy))

  def _compute_bounds_rdf(self, cfg):
    """Converts pose-frame xy bounds into a world-RDF (min, max) AABB.

    Bounds are axis-aligned in the pose-topic world frame (e.g. PX4 NED);
    the src->RDF conversion is a pure axis permutation, so the box stays
    axis-aligned. The vertical axis is left unbounded. Returns None when no
    bound is set.
    """
    vals = [cfg.get("xmin", None), cfg.get("xmax", None),
            cfg.get("ymin", None), cfg.get("ymax", None)]
    if all(v is None for v in vals):
      return None
    big = 1e6
    xmin = float(vals[0]) if vals[0] is not None else -big
    xmax = float(vals[1]) if vals[1] is not None else big
    ymin = float(vals[2]) if vals[2] is not None else -big
    ymax = float(vals[3]) if vals[3] is not None else big

    ds = getattr(self.server, "dataset", None)
    if ds is not None and hasattr(ds, "src2rdf"):
      R = ds.src2rdf[:3, :3]
    else:
      logger.warning("Exploration bounds: no dataset src2rdf available; "
                     "assuming bounds are already in world RDF.")
      R = torch.eye(3)
    # Transform the 8 corners (z unbounded) and take the axis-aligned hull.
    corners = torch.tensor(
      [[x, y, z] for x in (xmin, xmax) for y in (ymin, ymax)
       for z in (-big, big)], dtype=torch.float)
    corners = corners @ R.T
    return corners.min(dim=0).values, corners.max(dim=0).values

  def _in_bounds_mask(self, pts):
    """Boolean mask of points inside the RDF bounds (all True if unbounded)."""
    if self.bounds_rdf is None:
      return torch.ones(pts.shape[0], dtype=torch.bool, device=pts.device)
    mn, mx = self.bounds_rdf
    mn = mn.to(pts.device).reshape(1, 3)
    mx = mx.to(pts.device).reshape(1, 3)
    return ((pts >= mn) & (pts <= mx)).all(dim=-1)

  # ---------- lifecycle ----------

  def start(self):
    self._thread.start()
    logger.info(
      "Exploration planner started (source=%s, a=%.2f, b=%.2f, h=%.2fm, "
      "r=%.2fm, period=%.1fs).",
      self.frontier_source, self.distance_weight, self.heading_weight,
      self.hover_height, self.safety_radius, self.plan_period)

  def shutdown(self):
    self._stop.set()

  @torch.inference_mode()
  def _loop(self):
    # inference_mode is thread-local: the mapper's tensors are inference
    # tensors (created under the mapping thread's inference mode), and
    # running encoder modules on them from this thread without it would
    # attempt to record autograd state and fail.
    while not self._stop.is_set():
      try:
        self._update_info_grid()
      except Exception:
        logger.exception("Info grid update failed.")
      try:
        self._plan_once()
      except Exception:  # Keep planning alive; mapping owns the process.
        logger.exception("Exploration planning cycle failed.")
      self._stop.wait(self.plan_period)

  # ---------- inputs ----------

  def _get_robot_pose_rdf(self):
    """Latest body pose as 4x4 in world RDF from the dataset pose buffer."""
    ds = getattr(self.server, "dataset", None)
    if ds is None or not hasattr(ds, "_pose_buf"):
      return None
    with ds._pose_lock:
      if len(ds._pose_buf) == 0:
        return None
      _, pose_src = ds._pose_buf[-1]
    pose = torch.tensor(pose_src, dtype=torch.float)
    return g3d.transform_pose_4x4(pose, ds.src2rdf)

  def _bl_key(self, p):
    return tuple(torch.round(p / self._bl_grid).long().tolist())

  def _blacklist_add(self, key, severity=1):
    """Registers a failure for key; returns the resulting ban in seconds.

    severity is added to the fail count (use >1 for stronger offenses, e.g.
    a goal the drone physically failed to reach).
    """
    cnt = self._blacklist.get(key, (0, 0.0))[0] + severity
    ban = min(self.blacklist_base_ban_s * 2.0 ** (cnt - 1),
              self.blacklist_max_ban_s)
    self._blacklist[key] = (cnt, time.time() + ban)
    return ban

  def _blacklisted(self, key):
    """True while key's ban has not yet expired."""
    entry = self._blacklist.get(key)
    return entry is not None and time.time() < entry[1]

  # ---------- core logic (pure given inputs) ----------

  def rank_frontiers(self, robot_pose_4x4, frontiers):
    """Returns frontier indices sorted by ascending selection cost.

    cost = distance_weight * euclidean distance
         + heading_weight * |yaw change to face the frontier| (radians).
    Heading is measured in the horizontal (x, z) plane of world RDF.
    """
    p = robot_pose_4x4[:3, 3]
    fwd = robot_pose_4x4[:3, 2]  # camera/body forward (+z) in world
    yaw_robot = math.atan2(float(fwd[0]), float(fwd[2]))

    delta = frontiers - p.reshape(1, 3)
    dist = delta.norm(dim=-1)
    yaw_f = torch.atan2(delta[:, 0], delta[:, 2])
    dyaw = torch.abs(_wrap_angle(yaw_f - yaw_robot))
    cost = self.distance_weight * dist + self.heading_weight * dyaw
    return torch.argsort(cost)

  def _query_occ(self, pts):
    """Batch occupancy lookup (log-odds; 0 = unobserved). CPU tensors."""
    occ = rayfronts_cpp.query_occ(self.mapper.occ_map_vdb, pts.cpu())
    return occ.reshape(-1)

  def _los_clear(self, p, f):
    """True if the segment p->f has no occupied voxel (stops ~1 vox short)."""
    d = f - p
    length = float(d.norm())
    if length < self.vox_size:
      return True
    n = max(2, int(length / (self.vox_size * 0.5)))
    ts = torch.linspace(0.0, max(0.0, 1.0 - self.vox_size / length), n)
    pts = p.reshape(1, 3) + ts.reshape(-1, 1) * d.reshape(1, 3)
    return bool((self._query_occ(pts) <= 0).all())

  def find_exploration_pose(self, frontier, class_vox, robot_pos=None):
    """Searches a safe observation pose for a frontier point.

    The pose is hover_height above a selected-class voxel near the frontier,
    its safety_radius sphere is fully known-empty (unobserved counts as
    unsafe), the frontier lies within the camera's viewable elevation band
    (view_pitch_deg +- view_max_elevation_deg), and it has an occlusion-free
    line of sight to the frontier.

    Args:
      frontier: Float tensor of size 3 (world RDF).
      class_vox: Nx3 float tensor of selected-class voxel centers.
      robot_pos: Optional size-3 tensor to prefer candidates close to the
        robot; falls back to closeness to the frontier.

    Returns:
      A 4x4 world-RDF pose facing the frontier, or None.
    """
    delta = class_vox - frontier.reshape(1, 3)
    horiz = torch.stack([delta[:, 0], delta[:, 2]], dim=-1).norm(dim=-1)
    anchors = class_vox[horiz <= self.candidate_search_radius]
    if anchors.shape[0] == 0:
      return None

    # Candidate positions hover_height above the anchors (up = -y).
    cand = anchors.clone()
    cand[:, 1] -= self.hover_height

    # Keep a good observation standoff from the frontier, and require the
    # frontier's elevation (relative to the camera pitch) to be within the
    # viewable band — the goal yaw faces the frontier but pitch is fixed.
    cd = cand - frontier.reshape(1, 3)
    choriz = torch.stack([cd[:, 0], cd[:, 2]], dim=-1).norm(dim=-1)
    # Depression angle of the frontier as seen from the candidate
    # (positive = below the candidate; world RDF down = +y).
    depression = torch.atan2(-cd[:, 1], choriz)
    keep = ((choriz >= self.view_min_dist) & (choriz <= self.view_max_dist) &
            ((depression - self._view_pitch).abs() <= self._view_max_elev))
    cand = cand[keep]
    cand_standoff = choriz[keep]
    if cand.shape[0] == 0:
      return None

    # Rank candidates by cost = distance-to-robot + frontier_proximity_weight
    # * horizontal standoff. Weight 0 = cheapest to reach (drone observes
    # from wherever is convenient); larger weights push the goal toward the
    # frontier so reaching it actually advances the frontier. Don't let one
    # bad pocket exhaust the budget: keep the best half, then a uniform
    # random spread of the remainder (kept in cost order) so every region of
    # the standoff ring gets probed.
    ref = robot_pos if robot_pos is not None else frontier
    cost = ((cand - ref.reshape(1, 3)).norm(dim=-1) +
            self.frontier_proximity_weight * cand_standoff)
    order = torch.argsort(cost)
    if order.shape[0] > self.max_candidates:
      n_near = self.max_candidates // 2
      rest = order[n_near:]
      pick = torch.randperm(rest.shape[0])[:self.max_candidates - n_near]
      order = torch.cat([order[:n_near], rest[pick.sort().values]])
    cand = cand[order]

    # Safety: no point of the sphere may be occupied (log-odds > 0). By
    # default unobserved space (log-odds 0) also counts as unsafe; with
    # safety_allow_unknown it is accepted (consistent with the LOS check),
    # trusting the follower's local avoidance in never-scanned air.
    C, S = cand.shape[0], self._sphere_offsets.shape[0]
    pts = (cand.reshape(C, 1, 3) +
           self._sphere_offsets.reshape(1, S, 3)).reshape(-1, 3)
    occ = self._query_occ(pts).reshape(C, S)
    if self.safety_allow_unknown:
      safe = (occ <= 0).all(dim=-1)
    else:
      safe = (occ < 0).all(dim=-1)

    # Aim the sight line one voxel above the frontier: a ground frontier sits
    # inside the occupied surface layer, so a ray to the exact surface point
    # grazes through neighboring ground voxels at shallow approach angles and
    # falsely reports occlusion (blocking distant candidates in particular).
    f_view = frontier.clone()
    f_view[1] -= self.vox_size  # up = -y (world RDF)
    for i in torch.nonzero(safe).reshape(-1):
      pos = cand[i]
      if self._los_clear(pos, f_view):
        return self._goal_pose(pos, frontier)
    return None

  def _goal_pose(self, pos, frontier):
    """Builds a 4x4 world-RDF pose at pos with yaw facing the frontier."""
    d = frontier - pos
    d[1] = 0.0  # horizontal facing
    n = float(d.norm())
    fwd = d / n if n > 1e-6 else torch.tensor([0., 0., 1.])
    down = torch.tensor([0., 1., 0.])
    right = torch.cross(down, fwd, dim=0)
    right = right / right.norm().clamp(min=1e-9)
    T = torch.eye(4)
    T[:3, 0] = right
    T[:3, 1] = down
    T[:3, 2] = fwd
    T[:3, 3] = pos
    return T

  # ---------- 2D information grid ----------

  @staticmethod
  def _cell_hash(xyz, res):
    """Full 3D voxel-cell hash at resolution res."""
    c = torch.round(xyz / res).long()
    m = 2 ** 20
    return (c[:, 0] * m + c[:, 1]) * m + c[:, 2]

  @staticmethod
  def _col_hash(xyz, res):
    """Lateral (x, z) column hash at resolution res."""
    c = torch.round(xyz[:, [0, 2]] / res).long()
    return c[:, 0] * (2 ** 20) + c[:, 1]

  @staticmethod
  def _member(query_h, sorted_h):
    """Per-row membership of query hashes in a sorted hash tensor."""
    if sorted_h.shape[0] == 0:
      return torch.zeros(query_h.shape[0], dtype=torch.bool)
    p = torch.searchsorted(sorted_h, query_h).clamp(max=sorted_h.shape[0] - 1)
    return sorted_h[p] == query_h

  def _update_info_grid(self):
    """Updates the 2D info grid from new keyframe visible-voxel snapshots.

    Per keyframe: chosen-class visible voxels with no other-class voxel
    beneath them (same lateral voxel column, checked against the GLOBAL
    labeled map) mark their cell non-obstacle; other-class visible voxels
    within obstacle_max_height_voxels above a chosen-class voxel in their
    column mark their cell obstacle. Obstacle wins cell conflicts.
    """
    with self.server.map_lock:
      snaps = [(s, v.detach().cpu().clone())
               for s, v in self.mapper.keyframe_visible
               if s > self._grid_last_seq]
      part = self.mapper.get_class_partition() if snaps else None
      if part is not None:
        chosen = part[0].detach().cpu().clone()
        other = part[1].detach().cpu().clone()
    if not snaps or part is None or chosen.shape[0] == 0:
      return

    vox = self.vox_size
    # Global per-column structures (up = -y):
    # - highest chosen voxel per column (min y)
    # - deepest other voxel per column (max y): any other BELOW a voxel v in
    #   the column exists iff this deepest y > y_v.
    ch_col = self._col_hash(chosen, vox)
    ch_u, ch_inv = torch.unique(ch_col, return_inverse=True)
    ch_top = torch.full((ch_u.shape[0],), torch.inf)
    ch_top.scatter_reduce_(0, ch_inv, chosen[:, 1], reduce="amin",
                           include_self=False)
    ot_deep_u = torch.empty(0, dtype=torch.long)
    ot_deep = torch.empty(0)
    if other.shape[0] > 0:
      ot_col = self._col_hash(other, vox)
      ot_deep_u, ot_inv = torch.unique(ot_col, return_inverse=True)
      ot_deep = torch.full((ot_deep_u.shape[0],), -torch.inf)
      ot_deep.scatter_reduce_(0, ot_inv, other[:, 1], reduce="amax",
                              include_self=False)

    ch_set = torch.sort(self._cell_hash(chosen, vox)).values
    ot_set = torch.sort(self._cell_hash(other, vox)).values \
      if other.shape[0] > 0 else torch.empty(0, dtype=torch.long)

    g = self.grid_cell_size
    n_h = self.obstacle_max_height_voxels * vox
    for seq, vis in snaps:
      self._grid_last_seq = max(self._grid_last_seq, seq)
      # Only voxels inside the exploration bounds vote.
      vis = vis[self._in_bounds_mask(vis)]
      if vis.shape[0] == 0:
        continue
      vis_h = self._cell_hash(vis, vox)
      is_chosen = self._member(vis_h, ch_set)
      is_other = self._member(vis_h, ot_set) & ~is_chosen

      # Non-obstacle: chosen visible voxels with NO other-class beneath.
      vc = vis[is_chosen]
      if vc.shape[0] > 0:
        col = self._col_hash(vc, vox)
        has_other_below = torch.zeros(vc.shape[0], dtype=torch.bool)
        if ot_deep_u.shape[0] > 0:
          p = torch.searchsorted(ot_deep_u, col).clamp(
            max=ot_deep_u.shape[0] - 1)
          found = ot_deep_u[p] == col
          has_other_below = found & (ot_deep[p] > vc[:, 1] + 0.5 * vox)
        self._mark_cells(vc[~has_other_below], 1, g)

      # Obstacle: other visible voxels within n voxels above a chosen voxel
      # of the same column (chosen top strictly below the voxel).
      vo = vis[is_other]
      if vo.shape[0] > 0:
        col = self._col_hash(vo, vox)
        p = torch.searchsorted(ch_u, col).clamp(max=ch_u.shape[0] - 1)
        found = ch_u[p] == col
        top = ch_top[p]
        near_ground = (found & (top > vo[:, 1] + 0.5 * vox) &
                       (top - vo[:, 1] <= n_h + 1e-6))
        # Obstacle cells are displayed at their column's ground height so the
        # grid renders as a flat 2D map draped on the floor (the grid itself
        # is 2D; y is visualization-only).
        self._mark_cells(vo[near_ground], 2, g, y_vals=top[near_ground])

    self._vis_info_grid()

  def _mark_cells(self, pts, state, cell_size, y_vals=None):
    """Adds one vote per voxel to its cell and refreshes the cell state.

    Cells accumulate ground/obstacle vote counts across voxels and
    keyframes. State becomes obstacle iff there is at least one obstacle
    vote AND obs/(obs+gnd) >= obstacle_min_frac, else non-obstacle.
    Confidence is the winning state's vote fraction. y_vals optionally
    overrides the stored display height per point (display only).
    """
    if pts.shape[0] == 0:
      return
    ix = torch.round(pts[:, 0] / cell_size).long()
    iz = torch.round(pts[:, 2] / cell_size).long()
    ys = pts[:, 1] if y_vals is None else y_vals
    for k in range(pts.shape[0]):
      key = (int(ix[k]), int(iz[k]))
      cell = self.info_grid.get(key)
      if cell is None:
        cell = [0, 0.0, float(ys[k]), 0, 0]  # state, conf, y, gnd, obs
        self.info_grid[key] = cell
      if state == 2:
        cell[4] += 1
      else:
        cell[3] += 1
      cell[2] = float(ys[k])
      frac_obs = cell[4] / (cell[3] + cell[4])
      if cell[4] > 0 and frac_obs >= self.obstacle_min_frac:
        cell[0] = 2
        cell[1] = frac_obs
      else:
        cell[0] = 1
        cell[1] = 1.0 - frac_obs

  def _vis_info_grid(self):
    vis = getattr(self.server, "vis", None)
    if vis is None or len(self.info_grid) == 0:
      return
    try:
      g = self.grid_cell_size
      keys = list(self.info_grid.keys())
      vals = [self.info_grid[k] for k in keys]
      # Render the whole layer on ONE flat plane: per-cell heights inherit
      # classification noise (e.g. furniture voxels misclassified as ground
      # lift their column's "ground top"), which made cells float. Plane
      # height = configured grid_vis_height, else the median stored ground
      # height across cells.
      y_plane = self.cfg.get("grid_vis_height", None)
      if y_plane is None:
        ys = torch.tensor([v[2] for v in vals], dtype=torch.float)
        y_plane = float(ys.median())
      pts = torch.tensor([[k[0] * g, y_plane, k[1] * g]
                          for k in keys], dtype=torch.float)
      colors = torch.tensor([[0.2, 0.85, 0.3] if v[0] == 1
                             else [0.95, 0.2, 0.15] for v in vals],
                            dtype=torch.float)
      if hasattr(vis, "sync_thread_time"):
        vis.sync_thread_time()
      radii = torch.full((pts.shape[0],), g * 0.45)
      vis.log_pc(pts, colors, radii, layer="exploration/info_grid")
    except Exception:
      logger.exception("Failed to visualize info grid.")

  # ---------- goal feedback ----------

  def _on_goal_status(self, msg):
    with self._status_lock:
      self._latest_status = (int(msg.data), time.time())

  def _on_goal_publish_allow(self, msg):
    allowed = int(msg.data) != 0
    if allowed != self._goal_publish_allowed:
      logger.info("Goal publishing %s via goal_publish_allow.",
                  "enabled" if allowed else "disabled")
    self._goal_publish_allowed = allowed

  def _should_plan_new_goal(self):
    """Feedback gate for goal publishing.

    True when no goal is outstanding, or when the outstanding goal was
    reported reached/failed by a status message received at least
    status_min_delay seconds after the goal was published (earlier messages
    are stale feedback about the previous goal). A FAILED goal blacklists
    its frontier so it is not immediately re-selected.
    """
    if self._goal_pub is None or self._last_goal_pub_time is None:
      return True
    with self._status_lock:
      st = self._latest_status
    if st is None:
      return False
    val, t_recv = st
    if t_recv < self._last_goal_pub_time + self.status_min_delay:
      return False  # stale status (refers to the previous goal)
    if val == GOAL_IN_PROGRESS:
      return False
    if val == GOAL_FAILED and self._last_goal_key is not None:
      # A physically unreachable goal is a stronger signal than a missing
      # viewpoint; start it deeper into the backoff.
      ban = self._blacklist_add(self._last_goal_key, severity=2)
      logger.info("Goal reported FAILED; frontier banned for %.0fs "
                  "(%d blacklisted).", ban, len(self._blacklist))
    elif val == GOAL_REACHED:
      logger.info("Goal reported REACHED; planning next goal.")
    self._last_goal_pub_time = None
    self._last_goal_key = None
    return True

  # ---------- planning cycle ----------

  def _plan_once(self):
    if not self._goal_publish_allowed:
      return  # goal publishing suppressed externally; keep mapping as usual
    if not self._should_plan_new_goal():
      return
    robot_pose = self._get_robot_pose_rdf()
    if robot_pose is None:
      return

    with self.server.map_lock:
      frontiers = getattr(self.mapper, self.frontier_source, None)
      class_vox = self.mapper.class_voxels_xyz
      if frontiers is None or frontiers.shape[0] == 0 or class_vox is None:
        return
      frontiers = frontiers.detach().cpu().clone()
      class_vox = class_vox.detach().cpu().clone()

    # Drop frontiers with an active (unexpired) ban.
    keep = [i for i in range(frontiers.shape[0])
            if not self._blacklisted(self._bl_key(frontiers[i]))]
    if len(keep) == 0:
      logger.info("Exploration: all %d frontiers banned; waiting for new "
                  "frontiers or ban expiry.", frontiers.shape[0])
      return
    frontiers = frontiers[keep]

    order = self.rank_frontiers(robot_pose.cpu(), frontiers)
    robot_pos = robot_pose[:3, 3].cpu()

    for rank, i in enumerate(order.tolist()):
      if rank >= self.max_attempts_per_cycle:
        break
      f = frontiers[i]
      with self.server.map_lock:
        goal = self.find_exploration_pose(f, class_vox, robot_pos)
      if goal is None:
        ban = self._blacklist_add(self._bl_key(f))
        logger.info(
          "Exploration: no safe viewpoint for frontier (%.2f, %.2f, %.2f); "
          "banned for %.0fs (%d blacklisted).", *f.tolist(), ban,
          len(self._blacklist))
        continue

      self.current_goal = (goal, f)
      logger.info(
        "Exploration goal: pos (%.2f, %.2f, %.2f) observing frontier "
        "(%.2f, %.2f, %.2f).", *goal[:3, 3].tolist(), *f.tolist())
      self._publish_goal(goal, f)
      if self._goal_pub is not None:
        self._last_goal_pub_time = time.time()
        self._last_goal_key = self._bl_key(f)
      return

  # ---------- outputs ----------

  def _publish_goal(self, goal_rdf, frontier):
    # Visualize in rerun: big green goal sphere + 1m heading arrow + sight
    # line to the selected frontier (yellow). log_goal_pose also syncs this
    # thread's rerun timeline so the markers appear at the current playhead.
    vis = getattr(self.server, "vis", None)
    if vis is not None and hasattr(vis, "log_goal_pose"):
      try:
        vis.log_goal_pose(goal_rdf, frontier, layer="exploration_goal")
      except Exception:
        logger.exception("Failed to visualize exploration goal.")

    # Publish a bare geometry_msgs/Pose in the pose-topic (NED) world frame.
    if self._goal_pub is None:
      return
    ds = self.server.dataset
    src2rdf_inv = torch.linalg.inv(ds.src2rdf)
    goal_src = g3d.transform_pose_4x4(goal_rdf, src2rdf_inv)
    msg = Pose()
    t = goal_src[:3, 3].tolist()
    q = Rotation.from_matrix(goal_src[:3, :3].numpy()).as_quat()
    msg.position.x, msg.position.y, msg.position.z = t
    (msg.orientation.x, msg.orientation.y,
     msg.orientation.z, msg.orientation.w) = q.tolist()
    self._goal_pub.publish(msg)


@hydra.main(version_base=None, config_path="configs", config_name="default")
@torch.inference_mode()
def main(cfg=None):
  try:
    server = MappingServer(cfg)
  except KeyboardInterrupt:
    logger.info("Shutdown before initializing completed.")
    return

  planner = ExplorationPlanner(cfg.exploration, server)
  signal.signal(signal.SIGINT, partial(signal_handler, server))
  planner.start()
  try:
    server.run()
  except Exception as e:
    server.shutdown()
    raise e
  finally:
    planner.shutdown()


if __name__ == "__main__":
  main()
