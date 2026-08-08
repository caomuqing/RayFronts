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
from collections import deque
from functools import partial

import torch
import numpy as np
import hydra

from rayfronts import geometry3d as g3d
from rayfronts.geo_frame import FrontierFrame
from rayfronts.mapping_server import MappingServer, signal_handler
# rayfronts_cpp is imported (with its build path setup) by the mapper module.
from rayfronts.mapping.semantic_ray_frontiers_map import rayfronts_cpp

try:
  from geometry_msgs.msg import Pose, PolygonStamped, Point32
  from std_msgs.msg import Int8
  from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
  from scipy.spatial.transform import Rotation
except ModuleNotFoundError:
  Pose = None

try:
  from scipy.ndimage import label as scipy_label
except ModuleNotFoundError:
  scipy_label = None

try:
  from vision_msgs.msg import (Detection2DArray, Detection3DArray,
                               Detection3D, ObjectHypothesisWithPose)
except ModuleNotFoundError:
  Detection2DArray = None
  Detection3DArray = None

try:
  from nav_msgs.msg import OccupancyGrid
except ModuleNotFoundError:
  OccupancyGrid = None

from rayfronts.geo_frame import points_in_polygon

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
    v = cfg.get("goal_min_altitude", None)
    self.goal_min_altitude = None if v is None else float(v)
    self.goal_min_move = float(cfg.get("goal_min_move", 0.0))

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

    # Grid frame: the 2D frame in which all planar bookkeeping happens
    # (info grid cell keys, region segmentation, exploration bounds).
    # Without a configured frontier frame this is the legacy world-RDF
    # (x, z) plane with bounds from xmin/xmax/ymin/ymax; with one, it is
    # the mission-level global frame shared with the high-flying drone
    # (map cells align across robots) with bounds from ff_map_*.
    # Sets _grid_R/_grid_t (rdf (x,z) -> grid xy), grid_bounds
    # (min_x, max_x, min_y, max_y in grid xy) and bounds_rdf (RDF AABB
    # superset for the mapper's class-frontier generation).
    self._setup_grid_frame(cfg)
    if self.bounds_rdf is not None:
      self.mapper.class_frontier_bounds = self.bounds_rdf
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
    # Ban a frontier after this many REACHED observation goals that failed
    # to resolve it (prevents staring loops on unresolvable frontiers).
    self.frontier_reobserve_limit = int(
      cfg.get("frontier_reobserve_limit", 3))
    self._reached_counts = {}  # blacklist key -> reached-without-resolve

    # Region-based hierarchical exploration (MAIPP-style): segment the
    # semantically unexplored space (info-grid cells never classified)
    # inside the xy bounds into connected components, commit to the nearest
    # one, approach it via the frontier closest to its centroid, then
    # exhaust its frontiers before moving on. Region tasks compete with
    # person-track revisit tasks under a MAIPP-greedy score. See _task_plan
    # for the state machine.
    self.region_mode = bool(cfg.get("region_mode", False))
    self.region_min_area_m2 = float(cfg.get("region_min_area_m2", 1.0))
    self.region_max_area_m2 = float(cfg.get("region_max_area_m2", 25.0))
    self.region_min_frontiers = int(cfg.get("region_min_frontiers", 1))
    self.region_coverage_done = float(cfg.get("region_coverage_done", 0.85))
    self.region_time_budget_s = float(cfg.get("region_time_budget_s", 180.0))
    self.region_retire_cooldown_s = float(
      cfg.get("region_retire_cooldown_s", 180.0))
    self._task = None  # committed task: dict with type "region" | "track"
    # MAIPP-style greedy task selection (see _score_tasks / default.yaml).
    self.target_birth_density = float(cfg.get("target_birth_density", 0.01))
    self.task_w_mass = float(cfg.get("task_w_mass", 1.0))
    self.task_w_prob = float(cfg.get("task_w_prob", 0.5))
    self.task_w_exist = float(cfg.get("task_w_exist", 1.0))
    self.task_w_cov = float(cfg.get("task_w_cov", 0.2))
    self.task_w_stale = float(cfg.get("task_w_stale", 0.005))
    self.task_w_travel = float(cfg.get("task_w_travel", 0.05))
    self.track_task_min_existence = float(
      cfg.get("track_task_min_existence", 0.3))
    self.track_task_min_sigma = float(cfg.get("track_task_min_sigma", 0.8))
    self.track_task_min_unseen_s = float(
      cfg.get("track_task_min_unseen_s", 240.0))
    self.track_task_time_budget_s = float(
      cfg.get("track_task_time_budget_s", 60.0))
    z = cfg.get("task_priority_zone", None)
    self.task_priority_zone = (None if z is None else
                               [None if v is None else float(v) for v in z])
    self.task_priority_bonus = float(cfg.get("task_priority_bonus", 5.0))

    # Multi-robot MAIPP comms (see _setup_comms): coverage grids, person
    # tracks, and task claims exchanged with peers on standard topics
    # (/robot_<id>/maipp/...), all in frontier-frame coordinates.
    self.robot_id = int(cfg.get("robot_id", 0))
    self.peer_robot_ids = [int(x)
                           for x in (cfg.get("peer_robot_ids", None) or [])]
    self.comms_publish_period_s = float(
      cfg.get("comms_publish_period_s", 3.0))
    self.claim_ttl_s = float(cfg.get("claim_ttl_s", 10.0))
    self._peer_cover = {}   # rid -> set of grid-window cells (i, j)
    self._peer_claim = {}   # rid -> dict(cells=set|None, point=xy|None, t)
    self._imported = {}     # "rid:tid" -> local track_id
    self._comms_queue = deque(maxlen=64)
    self._comms_lock = threading.Lock()
    self._last_comms_pub = 0.0
    self._pub_cover = None
    self._retired = {}    # cell (i, j) -> retirement expiry walltime
    if self.region_mode and self.grid_bounds is None:
      logger.warning("region_mode requires bounds (xmin/xmax/ymin/ymax or a "
                     "frontier frame with ff_map_*); falling back to global "
                     "mode.")
      self.region_mode = False
    if self.region_mode and scipy_label is None:
      logger.warning("region_mode requires scipy.ndimage; falling back to "
                     "global mode.")
      self.region_mode = False

    # People detections (vision_msgs/Detection2DArray on detection_topic):
    # the bottom-center of each person bbox is undistorted, cast through the
    # camera pose at the detection stamp, and intersected with the ground
    # plane; hits within person_merge_radius of a tracked person update it,
    # otherwise a new person is created. Visualized on exploration/people.
    self.detection_topic = cfg.get("detection_topic", None)
    self.person_class_id = str(cfg.get("person_class_id", "person"))
    self.person_min_score = float(cfg.get("person_min_score", 0.5))
    self.person_merge_radius = float(cfg.get("person_merge_radius", 1.0))
    self.detection_max_range = float(cfg.get("detection_max_range", 12.0))
    self.detection_max_pose_dt = float(cfg.get("detection_max_pose_dt", 0.5))
    self.person_negative_min_interval_s = float(
      cfg.get("person_negative_min_interval_s", 0.5))
    # Bernoulli-Kalman track parameters (see configs/default.yaml).
    self.person_p_detect = float(cfg.get("person_p_detect", 0.6))
    self.person_exist_remove = float(cfg.get("person_exist_remove", 0.2))
    self.person_exist_confirm = float(cfg.get("person_exist_confirm", 0.8))
    self.person_process_noise = float(cfg.get("person_process_noise", 0.02))
    self.person_meas_noise_base = float(
      cfg.get("person_meas_noise_base", 0.3))
    self.person_meas_noise_per_m = float(
      cfg.get("person_meas_noise_per_m", 0.05))
    self.person_gate_chi2 = float(cfg.get("person_gate_chi2", 9.21))
    # Tracks: dicts with track_id, pos (3,), cov (2x2 over frame-plane x/z),
    # existence, status, n_obs, last_seen, score, last_neg_stamp.
    self.people = []
    self._next_track_id = 1
    self._people_prev_predict = time.time()
    self._det_queue = deque(maxlen=200)
    self._det_lock = threading.Lock()

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
      if self.detection_topic:
        if Detection2DArray is None:
          logger.warning("vision_msgs not importable; people detection on %s "
                         "disabled (apt install ros-humble-vision-msgs).",
                         self.detection_topic)
          self.detection_topic = None
        else:
          ds._rosnode.create_subscription(
            Detection2DArray, str(self.detection_topic),
            self._on_detections, 10)
          logger.info(
            "People detections on %s (class '%s', min score %.2f, merge "
            "radius %.1fm).", self.detection_topic, self.person_class_id,
            self.person_min_score, self.person_merge_radius)
      if self.robot_id > 0:
        self._setup_comms(ds)
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

  def _setup_grid_frame(self, cfg):
    """Establishes the planar grid frame, its bounds, and the RDF AABB.

    Legacy mode (no ff_origin_lat): grid xy = world-RDF (x, z); bounds come
    from xmin/xmax/ymin/ymax given in the pose-topic (NED) frame, so
    grid_bounds = (ymin, ymax, xmin, xmax) — RDF x is NED y and RDF z is
    NED x.

    Frontier-frame mode (ff_origin_lat + ned_origin_lat set): grid xy is
    the mission-level global frame shared with the high-flying drone (ENU
    at ff_origin_lat/lon rotated ff_heading_deg CCW from East). The local
    NED odometry frame is georeferenced by ned_origin_lat/lon/alt (global
    position of the local origin) and ned_heading_deg (compass azimuth,
    degrees clockwise from TRUE north, of the local NED +x axis — the
    drone's initialization heading). Bounds come from ff_map_min/max_x/y.

    Sets: _grid_R (2x2), _grid_t, _grid_R_inv, frame_active, grid_bounds
    (min_x, max_x, min_y, max_y in grid xy or None), bounds_rdf (RDF AABB
    enclosing the bounds box, for the mapper's class-frontier generation).
    """
    self._grid_R = torch.eye(2, dtype=torch.float)
    self._grid_t = torch.zeros(2, dtype=torch.float)
    self.frame_active = False
    self.grid_bounds = None
    self.bounds_rdf = None

    ff_lat = cfg.get("ff_origin_lat", None)
    # Georeference of the local NED odometry: flat ned_origin_* values,
    # overridden per robot by a matching ned_origins["robot<id>"] entry so
    # one preset carries every robot's takeoff parameters.
    ned = dict(lat=cfg.get("ned_origin_lat", None),
               lon=cfg.get("ned_origin_lon", None),
               alt=cfg.get("ned_origin_alt", 0.0),
               heading_deg=cfg.get("ned_heading_deg", 0.0))
    origins = cfg.get("ned_origins", None)
    rid = int(cfg.get("robot_id", 0))
    if origins is not None and rid > 0:
      entry = origins.get("robot%d" % rid, None)
      if entry is not None:
        for k in ned:
          v = entry.get(k, None)
          if v is not None:
            ned[k] = v
        logger.info("Using georeference ned_origins.robot%d.", rid)
    ned_lat = ned["lat"]
    if ff_lat is not None and ned_lat is None:
      logger.warning(
        "Frontier frame configured (ff_origin_lat) but the local NED "
        "georeference (ned_origin_lat/lon/heading or a ned_origins entry "
        "for robot_%d) is missing; falling back to local bounds.", rid)
      ff_lat = None

    if ff_lat is not None:
      ff = FrontierFrame(
        origin_lat=float(ff_lat),
        origin_lon=float(cfg.ff_origin_lon),
        origin_alt=float(cfg.get("ff_origin_alt", 0.0)),
        heading_deg=float(cfg.get("ff_heading_deg", 0.0)))
      ff.set_home(float(ned_lat), float(ned["lon"]), float(ned["alt"]))
      # Local NED -> ENU at the local origin: +x points at compass azimuth
      # psi (CW from true north), +y is 90deg right of it, +z is down.
      psi = math.radians(float(ned["heading_deg"]))
      r_n2e = np.array([[math.sin(psi), math.cos(psi), 0.0],
                        [math.cos(psi), -math.sin(psi), 0.0],
                        [0.0, 0.0, -1.0]])
      # Full chain NED -> frame; ENU@home -> frame is affine (geo_frame).
      m = ff._affine_R @ r_n2e
      # Planar 2x2 on RDF (x, z): rdf x = ned y, rdf z = ned x. The dropped
      # altitude column contributes < 1e-4 m/m horizontally at mission
      # scale (< 1 km).
      self._grid_R = torch.tensor(
        [[m[0, 1], m[0, 0]], [m[1, 1], m[1, 0]]], dtype=torch.float)
      self._grid_t = torch.tensor(
        [float(ff._affine_t[0]), float(ff._affine_t[1])],
        dtype=torch.float)
      self.frame_active = True
      self.grid_bounds = (float(cfg.ff_map_min_x), float(cfg.ff_map_max_x),
                          float(cfg.ff_map_min_y), float(cfg.ff_map_max_y))
      logger.info(
        "Frontier frame active: origin (%.6f, %.6f) heading %.2fdeg; local "
        "NED origin (%.6f, %.6f) heading %.2fdeg sits at frame (%.1f, %.1f)"
        "; map bounds x [%.1f, %.1f], y [%.1f, %.1f].",
        ff.origin_lat, ff.origin_lon, ff.heading_deg, float(ned_lat),
        float(ned["lon"]), float(ned["heading_deg"]),
        float(self._grid_t[0]), float(self._grid_t[1]), *self.grid_bounds)
    else:
      # Legacy: bounds in the pose-topic (NED) frame; grid xy = RDF (x, z).
      vals = [cfg.get("xmin", None), cfg.get("xmax", None),
              cfg.get("ymin", None), cfg.get("ymax", None)]
      if not all(v is None for v in vals):
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
        # Transform the box corners through the (axis permutation) src->RDF
        # and read the grid (RDF x, z) ranges off the hull.
        corners = torch.tensor(
          [[x, y, z] for x in (xmin, xmax) for y in (ymin, ymax)
           for z in (-big, big)], dtype=torch.float)
        corners = corners @ R.T
        self.grid_bounds = (float(corners[:, 0].min()),
                            float(corners[:, 0].max()),
                            float(corners[:, 2].min()),
                            float(corners[:, 2].max()))
        logger.info(
          "Exploration bounds active: x [%s, %s], y [%s, %s] (pose frame).",
          *vals)

    self._grid_R_inv = torch.linalg.inv(self._grid_R)
    if self.grid_bounds is not None:
      # RDF AABB enclosing the (possibly rotated) bounds box, y unbounded.
      mnx, mxx, mny, mxy = self.grid_bounds
      corners = torch.tensor([[x, y] for x in (mnx, mxx) for y in (mny, mxy)],
                             dtype=torch.float)
      c_rdf = (corners - self._grid_t) @ self._grid_R_inv.T  # (x_rdf, z_rdf)
      big = 1e6
      self.bounds_rdf = (
        torch.tensor([float(c_rdf[:, 0].min()), -big,
                      float(c_rdf[:, 1].min())]),
        torch.tensor([float(c_rdf[:, 0].max()), big,
                      float(c_rdf[:, 1].max())]))

  def _grid_xy(self, pts):
    """World-RDF points (Nx3) -> grid-frame xy (Nx2)."""
    return pts[:, [0, 2]] @ self._grid_R.T + self._grid_t

  def _grid_xy_to_rdf(self, xy, y):
    """Grid-frame xy (Nx2) -> world-RDF points (Nx3) at height y."""
    xz = (xy - self._grid_t) @ self._grid_R_inv.T
    out = torch.empty((xy.shape[0], 3), dtype=torch.float)
    out[:, 0] = xz[:, 0]
    out[:, 1] = y
    out[:, 2] = xz[:, 1]
    return out

  def _in_bounds_mask(self, pts):
    """Boolean mask of RDF points inside the grid-frame bounds box.

    Exact test in grid xy (the box may be rotated in RDF); all True when
    unbounded.
    """
    if self.grid_bounds is None:
      return torch.ones(pts.shape[0], dtype=torch.bool, device=pts.device)
    xy = self._grid_xy(pts.cpu())
    mnx, mxx, mny, mxy = self.grid_bounds
    mask = ((xy[:, 0] >= mnx) & (xy[:, 0] <= mxx) &
            (xy[:, 1] >= mny) & (xy[:, 1] <= mxy))
    return mask.to(pts.device)

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
        self._update_comms()
      except Exception:
        logger.exception("Peer comms update failed.")
      try:
        self._update_people()
      except Exception:
        logger.exception("People update failed.")
      try:
        self._update_info_grid()
      except Exception:
        logger.exception("Info grid update failed.")
      try:
        self._publish_comms()
      except Exception:
        logger.exception("Comms publish failed.")
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

  # ---------- people detection ----------

  def _on_detections(self, msg):
    """ROS callback: queue person bbox bottom-centers for projection.

    Runs on the ROS executor thread; projection happens on the planner
    thread (_update_people) where map/pose access is already organized.
    """
    stamp_ns = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
    dets = []
    for det in msg.detections:
      best_score = 0.0
      for r in det.results:
        if (str(r.hypothesis.class_id) == self.person_class_id and
            float(r.hypothesis.score) > best_score):
          best_score = float(r.hypothesis.score)
      if best_score < self.person_min_score:
        continue
      u = float(det.bbox.center.position.x)
      v = float(det.bbox.center.position.y) + 0.5 * float(det.bbox.size_y)
      dets.append((u, v, best_score))
    # Empty frames are queued too: they are the negative evidence that
    # removes stale people (see _negative_pass).
    with self._det_lock:
      self._det_queue.append((stamp_ns, dets))

  def _lookup_body_pose_rdf(self, stamp_ns):
    """Body pose (4x4 world RDF) nearest to stamp_ns and its |dt| seconds."""
    ds = self.server.dataset
    with ds._pose_lock:
      buf = list(ds._pose_buf)
    if len(buf) == 0:
      return None
    diffs = [abs(t - stamp_ns) for t, _ in buf]
    i = diffs.index(min(diffs))
    pose = torch.tensor(buf[i][1], dtype=torch.float)
    return g3d.transform_pose_4x4(pose, ds.src2rdf), diffs[i] / 1e9

  def _update_people(self):
    """Projects queued detections to ground and merges them into people.

    Bottom-center pixels are undistorted with the dataset's raw (scaled)
    RGB calibration — detections are assumed to be in the pixel space of
    the same rgb_topic images RayFronts receives — cast through the camera
    pose at the (clock-corrected) detection stamp, and intersected with the
    ground plane at the median ground-class voxel height.
    """
    if not self.detection_topic:
      return
    with self._det_lock:
      batches = list(self._det_queue)
      self._det_queue.clear()
    if len(batches) == 0:
      return
    ds = self.server.dataset
    K = getattr(ds, "_rgb_K_orig", None)
    dist = getattr(ds, "_rgb_dist_coeffs", None)
    if K is None:
      return  # no RGB frame processed yet; calibration space unknown
    with self.server.map_lock:
      cls = self.mapper.class_voxels_xyz
      gy = (float(cls[:, 1].median())
            if cls is not None and cls.shape[0] > 0 else None)
    if gy is None:
      return  # no classified ground yet to intersect with
    import cv2
    # Kalman predict: static-target model, so covariance grows with time
    # while a person goes unobserved (quantifies revisit value / widens the
    # association gate for stale tracks).
    now = time.time()
    dt = max(0.0, now - self._people_prev_predict)
    self._people_prev_predict = now
    if dt > 0.0:
      q = self.person_process_noise * dt
      for person in self.people:
        person["cov"] = person["cov"] + q * torch.eye(2)

    stamp_off = getattr(ds, "_rgb_stamp_offset_ns", 0)
    changed = False
    for stamp_ns, dets in batches:
      r = self._lookup_body_pose_rdf(stamp_ns + stamp_off)
      if r is None or r[1] > self.detection_max_pose_dt:
        continue
      T_wc = r[0] @ ds.T_body_rgb
      rot, cam_pos = T_wc[:3, :3], T_wc[:3, 3]
      matched = set()
      for u, v, score in dets:
        pix = np.array([[[u, v]]], dtype=np.float32)
        n = cv2.undistortPoints(pix, np.asarray(K, dtype=np.float64), dist)
        ray = rot @ torch.tensor(
          [float(n[0, 0, 0]), float(n[0, 0, 1]), 1.0])
        if float(ray[1]) < 1e-3:
          continue  # at/above the horizon; no ground intersection ahead
        t = (gy - float(cam_pos[1])) / float(ray[1])
        if t <= 0:
          continue
        w = cam_pos + t * ray
        rng = math.hypot(float(w[0] - cam_pos[0]), float(w[2] - cam_pos[2]))
        if rng > self.detection_max_range:
          continue  # grazing ray; too unreliable
        w[1] = gy
        # Far/grazing projections carry more ground error; trust them less.
        sigma = self.person_meas_noise_base + self.person_meas_noise_per_m * rng
        matched.add(self._associate_detection(w, score, sigma))
        changed = True
      if len(self.people) > 0:
        changed |= self._negative_pass(rot, cam_pos, K, matched, stamp_ns)
    if changed:
      self._vis_people()

  def _person_status(self, person):
    """Track lifecycle label from existence probability."""
    if (person["existence"] >= self.person_exist_confirm and
        person["n_obs"] >= 2):
      return "confirmed"
    if person["existence"] < 0.35:
      return "stale"
    return "candidate"

  def _associate_detection(self, pos, score, meas_sigma):
    """Bernoulli-KF association/update for one projected detection.

    Associates to the track with the smallest Mahalanobis distance within
    person_gate_chi2 (person_merge_radius as a euclidean floor), then
    Kalman-updates its mean/covariance and raises its existence. Unmatched
    detections give birth to a new track (existence seeded from the
    detector score). Returns the track's index.
    """
    now = time.time()
    z = torch.tensor([float(pos[0]), float(pos[2])])
    R = (meas_sigma ** 2) * torch.eye(2)
    best, best_d2 = None, None
    for i, person in enumerate(self.people):
      mu = torch.tensor([float(person["pos"][0]), float(person["pos"][2])])
      innov = z - mu
      S = person["cov"] + R
      d2 = float(innov @ torch.linalg.solve(S, innov))
      if (d2 <= self.person_gate_chi2 or
          float(innov.norm()) <= self.person_merge_radius):
        if best is None or d2 < best_d2:
          best, best_d2 = i, d2
    if best is not None:
      person = self.people[best]
      mu = torch.tensor([float(person["pos"][0]), float(person["pos"][2])])
      innov = z - mu
      S = person["cov"] + R
      gain = person["cov"] @ torch.linalg.inv(S)
      new_mu = mu + gain @ innov
      person["pos"] = torch.tensor(
        [float(new_mu[0]), float(pos[1]), float(new_mu[1])])
      person["cov"] = (torch.eye(2) - gain) @ person["cov"]
      pd = self.person_p_detect
      person["existence"] = min(
        1.0, 1.0 - (1.0 - person["existence"]) * (1.0 - pd))
      person["n_obs"] += 1
      person["last_seen"] = now
      person["score"] = max(person["score"], score)
      person["status"] = self._person_status(person)
      return best
    # Birth: existence seeded from detector confidence; covariance from the
    # measurement noise plus a placement margin.
    self.people.append(dict(
      track_id=self._next_track_id,
      pos=pos.clone(),
      cov=R + 0.25 * torch.eye(2),
      existence=min(0.9, max(0.4, score)),
      status="candidate",
      n_obs=1, last_seen=now, score=score, last_neg_stamp=-10**18))
    logger.info(
      "Person track #%d born at (%.1f, %.1f), score %.2f, existence %.2f.",
      self._next_track_id, float(pos[0]), float(pos[2]), score,
      self.people[-1]["existence"])
    self._next_track_id += 1
    return len(self.people) - 1

  def _negative_pass(self, rot, cam_pos, K, matched, stamp_ns):
    """Counts missed observations for people the frame should have seen.

    A tracked person not matched by any detection in this frame accrues a
    negative observation only when the detector genuinely should have seen
    them: BOTH a low body point (0.3m above ground — a person may be lying
    down) AND the mid-body point (0.9m) project well inside the image (10%
    border margin), lie in front of the camera within detection_max_range,
    and have occlusion-free lines of sight. Low cover that could hide a
    lying person (tall grass, rocks, ridges) therefore pauses the countdown
    rather than counting against them. Negatives are rate-limited to one
    per person_negative_min_interval_s of message time (detectors flicker
    frame to frame). Each counted miss lowers the track's existence via the
    Bernoulli missed-detection update r <- r(1-p_d)/(1-r*p_d); dropping
    below person_exist_remove deletes the track. Positive matches raise
    existence back up (see _associate_detection).
    """
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    # Approximate frame size from the (near-centered) principal point.
    W, H = 2.0 * cx, 2.0 * cy
    min_gap_ns = int(self.person_negative_min_interval_s * 1e9)
    check_heights = (0.3, 0.9)  # lying-body and standing mid-body points
    changed = False
    keep = []
    for i, person in enumerate(self.people):
      if i in matched:
        keep.append(person)
        continue
      should_see = True
      for h in check_heights:
        pt = person["pos"] + torch.tensor([0.0, -h, 0.0])
        pc = rot.T @ (pt - cam_pos)
        z = float(pc[2])
        if z <= 1.0:
          should_see = False
          break
        u = fx * float(pc[0]) / z + cx
        v = fy * float(pc[1]) / z + cy
        if not (0.1 * W < u < 0.9 * W and 0.1 * H < v < 0.9 * H and
                math.hypot(float(pt[0] - cam_pos[0]),
                           float(pt[2] - cam_pos[2]))
                <= self.detection_max_range):
          should_see = False
          break
        if not self._los_clear(cam_pos, pt):
          should_see = False
          break
      gap = stamp_ns - person["last_neg_stamp"]
      rate_ok = gap >= min_gap_ns or gap < -5 * 10**9  # bag-loop wrap
      if should_see and rate_ok:
        pd = self.person_p_detect
        r = person["existence"]
        denom = 1.0 - r * pd
        person["existence"] = (0.0 if denom <= 1e-9
                               else r * (1.0 - pd) / denom)
        person["last_neg_stamp"] = stamp_ns
        person["status"] = self._person_status(person)
        changed = True
        if person["existence"] < self.person_exist_remove:
          logger.info(
            "Person track #%d at (%.1f, %.1f) removed (existence %.2f "
            "after missed observations).", person["track_id"],
            float(person["pos"][0]), float(person["pos"][2]),
            person["existence"])
          continue  # drop from keep list
      keep.append(person)
    self.people = keep
    return changed

  def _vis_people(self):
    """Rerun overlay: tracked people at ground height.

    Disc radius reflects positional uncertainty (sigma from the covariance
    trace); color reflects the track lifecycle (confirmed red, candidate
    orange, stale gray). Logs an empty set too, so removing the last person
    clears the layer.
    """
    vis = getattr(self.server, "vis", None)
    if vis is None or not hasattr(vis, "log_pc"):
      return
    try:
      n = len(self.people)
      status_rgb = {"confirmed": [230, 30, 30], "candidate": [255, 150, 40],
                    "stale": [140, 90, 90]}
      pts = (torch.stack([p["pos"] for p in self.people]) if n > 0
             else torch.zeros((0, 3)))
      colors = torch.tensor(
        [status_rgb.get(p["status"], [255, 150, 40]) for p in self.people],
        dtype=torch.uint8).reshape(n, 3)
      radii = torch.tensor(
        [min(2.5, 0.25 + math.sqrt(0.5 * float(torch.trace(p["cov"]))))
         for p in self.people]).reshape(n)
      vis.log_pc(pts, colors, radii, layer="exploration/people")
    except Exception:
      logger.exception("Failed to visualize people.")

  # ---------- multi-robot MAIPP comms ----------

  def _setup_comms(self, ds):
    """Publishers/subscribers for inter-robot belief sharing.

    Standard topics under /robot_<id>/maipp, all in frontier-frame
    coordinates:
      coverage_grid  nav_msgs/OccupancyGrid   (-1 unknown, 0 ground,
                                               100 obstacle)
      tracks         vision_msgs/Detection3DArray (id "<rid>:<tid>",
                     score=existence, pose+covariance, stamp=last_seen)
      task_claim     geometry_msgs/PolygonStamped (region bbox | single
                     point at a claimed track | empty = idle)
    """
    if OccupancyGrid is None or Detection3DArray is None:
      logger.warning("nav_msgs/vision_msgs unavailable; multi-robot comms "
                     "disabled.")
      return
    if self.grid_bounds is None:
      logger.warning("Multi-robot comms require xy bounds / frontier frame; "
                     "disabled.")
      return
    node = ds._rosnode
    self._comms_node = node
    ns = "/robot_%d/maipp" % self.robot_id
    self._pub_cover = node.create_publisher(
      OccupancyGrid, ns + "/coverage_grid", 1)
    self._pub_tracks = node.create_publisher(
      Detection3DArray, ns + "/tracks", 1)
    self._pub_claim = node.create_publisher(
      PolygonStamped, ns + "/task_claim", 1)
    for rid in self.peer_robot_ids:
      if rid == self.robot_id:
        continue
      pns = "/robot_%d/maipp" % rid
      node.create_subscription(
        OccupancyGrid, pns + "/coverage_grid",
        partial(self._on_peer_msg, "cover", rid), 1)
      node.create_subscription(
        Detection3DArray, pns + "/tracks",
        partial(self._on_peer_msg, "tracks", rid), 1)
      node.create_subscription(
        PolygonStamped, pns + "/task_claim",
        partial(self._on_peer_msg, "claim", rid), 1)
    logger.info("MAIPP comms up as robot_%d (peers %s, period %.1fs).",
                self.robot_id, self.peer_robot_ids,
                self.comms_publish_period_s)

  def _on_peer_msg(self, kind, rid, msg):
    """ROS callback: reduce peer messages to plain data and queue them."""
    if kind == "cover":
      w, h = msg.info.width, msg.info.height
      if w == 0 or h == 0:
        return
      data = np.array(msg.data, dtype=np.int8).reshape(h, w)
      rows, cols = np.nonzero(data >= 0)
      res = float(msg.info.resolution)
      xs = float(msg.info.origin.position.x) + (cols + 0.5) * res
      ys = float(msg.info.origin.position.y) + (rows + 0.5) * res
      payload = (xs, ys)
    elif kind == "tracks":
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
    else:  # claim
      payload = [(float(p.x), float(p.y)) for p in msg.polygon.points]
    with self._comms_lock:
      self._comms_queue.append((kind, rid, payload))

  def _update_comms(self):
    """Fuses queued peer messages (planner thread)."""
    if self.robot_id <= 0:
      return
    with self._comms_lock:
      items = list(self._comms_queue)
      self._comms_queue.clear()
    if len(items) == 0:
      return
    now = time.time()
    ix0, iz0, nx, nz = self._region_grid()
    g = self.grid_cell_size
    for kind, rid, payload in items:
      if kind == "cover":
        xs, ys = payload
        i = np.round(xs / g).astype(np.int64) - ix0
        j = np.round(ys / g).astype(np.int64) - iz0
        ok = (i >= 0) & (i < nx) & (j >= 0) & (j < nz)
        first = rid not in self._peer_cover
        self._peer_cover[rid] = set(
          zip(i[ok].tolist(), j[ok].tolist()))
        if first:
          logger.info(
            "Receiving coverage from robot_%d: %d classified cells "
            "in bounds (further updates logged at debug level).",
            rid, len(self._peer_cover[rid]))
        else:
          logger.debug("Coverage from robot_%d: %d cells in bounds.",
                       rid, len(self._peer_cover[rid]))
      elif kind == "tracks":
        self._fuse_peer_tracks(rid, payload)
      else:  # claim
        if len(payload) == 0:
          self._peer_claim.pop(rid, None)
        elif len(payload) < 3:
          self._peer_claim[rid] = dict(cells=None, point=payload[0], t=now)
        else:
          ii, jj = np.meshgrid(np.arange(nx), np.arange(nz), indexing="ij")
          centers = np.stack([(ii.reshape(-1) + ix0) * g,
                              (jj.reshape(-1) + iz0) * g], axis=-1)
          inside = points_in_polygon(centers, np.array(payload))
          cells = set(zip((ii.reshape(-1)[inside]).tolist(),
                          (jj.reshape(-1)[inside]).tolist()))
          self._peer_claim[rid] = dict(cells=cells, point=None, t=now)

  def _fuse_peer_tracks(self, rid, dets):
    """Merges a peer's track list into the local Bernoulli-KF tracks.

    Idempotent under periodic re-reception: a matched track adopts the
    peer's mean/covariance only when the peer is more certain (smaller
    covariance trace); existence and last_seen take the max. Unmatched
    peer tracks are imported as local tracks (remembering their foreign
    id so later messages update rather than duplicate).
    """
    with self.server.map_lock:
      cls = self.mapper.class_voxels_xyz
      gy = (float(cls[:, 1].median())
            if cls is not None and cls.shape[0] > 0 else None)
    if gy is None:
      if len(self.people) > 0:
        gy = float(self.people[0]["pos"][1])
      else:
        return  # no ground estimate yet; peers republish periodically
    changed = False
    for key, fx, fy, cov4, r_rem, stamp in dets:
      if key.startswith("%d:" % self.robot_id):
        continue  # our own track echoed back through a relay
      pos = self._grid_xy_to_rdf(torch.tensor([[fx, fy]]), gy)[0]
      P_frame = torch.tensor([[cov4[0], cov4[1]], [cov4[2], cov4[3]]],
                             dtype=torch.float)
      P_rem = self._grid_R_inv @ P_frame @ self._grid_R_inv.T
      target = None
      if key in self._imported:
        target = next((p for p in self.people
                       if p["track_id"] == self._imported[key]), None)
      if target is None:
        z = torch.tensor([float(pos[0]), float(pos[2])])
        for person in self.people:
          mu = torch.tensor([float(person["pos"][0]),
                             float(person["pos"][2])])
          innov = z - mu
          S = person["cov"] + P_rem
          d2 = float(innov @ torch.linalg.solve(S, innov))
          if (d2 <= self.person_gate_chi2 or
              float(innov.norm()) <= self.person_merge_radius):
            target = person
            break
      if target is not None:
        self._imported[key] = target["track_id"]
        if float(torch.trace(P_rem)) < float(torch.trace(target["cov"])):
          target["pos"] = pos.clone()
          target["cov"] = P_rem
        target["existence"] = max(target["existence"], r_rem)
        target["last_seen"] = max(target["last_seen"], stamp)
        target["status"] = self._person_status(target)
        changed = True
      else:
        self.people.append(dict(
          track_id=self._next_track_id, pos=pos.clone(), cov=P_rem,
          existence=r_rem, status="candidate", n_obs=1, last_seen=stamp,
          score=r_rem, last_neg_stamp=-10**18))
        self._imported[key] = self._next_track_id
        logger.info("Imported person track %s from robot_%d as #%d at "
                    "(%.1f, %.1f).", key, rid, self._next_track_id,
                    float(pos[0]), float(pos[2]))
        self._next_track_id += 1
        changed = True
    if changed:
      self._vis_people()

  def _track_claimed(self, person):
    """True if a fresh peer claim points at this track."""
    now = time.time()
    pxy = self._grid_xy(person["pos"].reshape(1, 3))[0]
    for c in self._peer_claim.values():
      if now - c["t"] > self.claim_ttl_s or c.get("point") is None:
        continue
      if (math.hypot(float(pxy[0]) - c["point"][0],
                     float(pxy[1]) - c["point"][1])
          <= self.person_merge_radius):
        return True
    return False

  def _claimed_cells(self):
    """Union of cells inside fresh peer region claims (expired ones drop)."""
    now = time.time()
    cells = set()
    for rid in list(self._peer_claim.keys()):
      c = self._peer_claim[rid]
      if now - c["t"] > self.claim_ttl_s:
        self._peer_claim.pop(rid)
      elif c.get("cells"):
        cells |= c["cells"]
    return cells

  def _publish_comms(self):
    """Periodically publishes coverage grid, tracks, and the task claim."""
    if self._pub_cover is None:
      return
    now = time.time()
    if now - self._last_comms_pub < self.comms_publish_period_s:
      return
    self._last_comms_pub = now
    stamp = self._comms_node.get_clock().now().to_msg()
    try:
      self._pub_cover.publish(self._build_cover_msg(stamp))
      self._pub_tracks.publish(self._build_tracks_msg(stamp))
      self._pub_claim.publish(self._build_claim_msg(stamp))
    except Exception:
      logger.exception("Failed to publish MAIPP comms.")

  def _build_cover_msg(self, stamp):
    ix0, iz0, nx, nz = self._region_grid()
    g = self.grid_cell_size
    grid = np.full((nz, nx), -1, dtype=np.int8)
    for (ix, iz), cell in self.info_grid.items():
      i, j = ix - ix0, iz - iz0
      if 0 <= i < nx and 0 <= j < nz:
        grid[j, i] = 100 if cell[0] == 2 else 0
    msg = OccupancyGrid()
    msg.header.stamp = stamp
    msg.header.frame_id = "frontier_frame"
    msg.info.resolution = float(g)
    msg.info.width = nx
    msg.info.height = nz
    msg.info.origin.position.x = (ix0 - 0.5) * g
    msg.info.origin.position.y = (iz0 - 0.5) * g
    msg.info.origin.orientation.w = 1.0
    msg.data = grid.reshape(-1).tolist()
    return msg

  def _build_tracks_msg(self, stamp):
    arr = Detection3DArray()
    arr.header.stamp = stamp
    arr.header.frame_id = "frontier_frame"
    for p in self.people:
      det = Detection3D()
      det.header.frame_id = "frontier_frame"
      det.header.stamp.sec = int(p["last_seen"])
      det.header.stamp.nanosec = int((p["last_seen"] % 1.0) * 1e9)
      det.id = "%d:%d" % (self.robot_id, p["track_id"])
      hyp = ObjectHypothesisWithPose()
      hyp.hypothesis.class_id = self.person_class_id
      hyp.hypothesis.score = float(p["existence"])
      pxy = self._grid_xy(p["pos"].reshape(1, 3))[0]
      hyp.pose.pose.position.x = float(pxy[0])
      hyp.pose.pose.position.y = float(pxy[1])
      P_frame = self._grid_R @ p["cov"] @ self._grid_R.T
      cov = [0.0] * 36
      cov[0] = float(P_frame[0, 0])
      cov[1] = float(P_frame[0, 1])
      cov[6] = float(P_frame[1, 0])
      cov[7] = float(P_frame[1, 1])
      hyp.pose.covariance = cov
      det.results.append(hyp)
      arr.detections.append(det)
    return arr

  def _build_claim_msg(self, stamp):
    msg = PolygonStamped()
    msg.header.stamp = stamp
    msg.header.frame_id = "frontier_frame"
    t = self._task
    if t is not None and t["type"] == "region":
      ix0, iz0, _, _ = self._region_grid()
      g = self.grid_cell_size
      arr = np.array(sorted(t["cells"]), dtype=np.float64)
      x_lo = (arr[:, 0].min() + ix0) * g - 0.5 * g
      x_hi = (arr[:, 0].max() + ix0) * g + 0.5 * g
      y_lo = (arr[:, 1].min() + iz0) * g - 0.5 * g
      y_hi = (arr[:, 1].max() + iz0) * g + 0.5 * g
      for x, y in ((x_lo, y_lo), (x_hi, y_lo), (x_hi, y_hi), (x_lo, y_hi)):
        msg.polygon.points.append(Point32(x=float(x), y=float(y), z=0.0))
    elif t is not None:
      track = next((p for p in self.people
                    if p["track_id"] == t["track_id"]), None)
      if track is not None:
        pxy = self._grid_xy(track["pos"].reshape(1, 3))[0]
        msg.polygon.points.append(
          Point32(x=float(pxy[0]), y=float(pxy[1]), z=0.0))
    return msg

  # ---------- region-based hierarchical exploration ----------

  def _region_grid(self):
    """Info-grid index window covering the bounds.

    Returns (ix0, iz0, nx, nz): region cells are the info-grid cells
    (centers at ix*grid_cell_size, iz*grid_cell_size in GRID-FRAME xy)
    whose centers lie inside the grid-frame bounds; array index (i, j)
    maps to info-grid key (i + ix0, j + iz0).
    """
    mnx, mxx, mny, mxy = self.grid_bounds
    g = self.grid_cell_size
    ix0 = int(math.ceil(mnx / g))
    iz0 = int(math.ceil(mny / g))
    nx = max(1, int(math.floor(mxx / g)) - ix0 + 1)
    nz = max(1, int(math.floor(mxy / g)) - iz0 + 1)
    return ix0, iz0, nx, nz

  def _unknown_mask(self):
    """Boolean [nx, nz] numpy mask of semantically unexplored cells.

    A cell is unknown while it has no entry in the 2D info grid, i.e. the
    camera has never classified ground/obstacle there (mere lidar coverage
    does not count — the mission is semantic coverage). Cells of recently
    completed regions (retired, on cooldown) are masked out so the planner
    does not immediately re-select them.
    """
    ix0, iz0, nx, nz = self._region_grid()
    unknown = np.ones((nx, nz), dtype=bool)
    for (ix, iz) in self.info_grid.keys():
      i, j = ix - ix0, iz - iz0
      if 0 <= i < nx and 0 <= j < nz:
        unknown[i, j] = False
    now = time.time()
    self._retired = {c: t for c, t in self._retired.items() if t > now}
    for (i, j) in self._retired:
      if 0 <= i < nx and 0 <= j < nz:
        unknown[i, j] = False
    # Ground a peer already classified is not unexplored (this also counts
    # toward our committed region's coverage, which is correct).
    for cells in self._peer_cover.values():
      for (i, j) in cells:
        if 0 <= i < nx and 0 <= j < nz:
          unknown[i, j] = False
    return unknown

  def _segment_unknown(self, unknown):
    """Splits the unknown mask into region cell sets.

    8-connected components, recursively median-split along the higher-
    variance axis above region_max_area_m2, dropped below
    region_min_area_m2.
    """
    labeled, n = scipy_label(unknown, structure=np.ones((3, 3), dtype=int))
    cs2 = self.grid_cell_size ** 2
    min_cells = max(1, int(round(self.region_min_area_m2 / cs2)))
    max_cells = max(min_cells * 2, int(round(self.region_max_area_m2 / cs2)))
    regions = []
    for k in range(1, n + 1):
      idx = np.argwhere(labeled == k)  # [K, 2] (i, j)
      if idx.shape[0] < min_cells:
        continue
      regions.extend(self._split_cells(idx, max_cells))
    return [set(map(tuple, r.tolist())) for r in regions]

  def _split_cells(self, idx, max_cells):
    if idx.shape[0] <= max_cells:
      return [idx]
    axis = int(np.argmax(idx.astype(np.float32).var(axis=0)))
    order = np.argsort(idx[:, axis], kind="stable")
    mid = idx.shape[0] // 2
    return (self._split_cells(idx[order[:mid]], max_cells) +
            self._split_cells(idx[order[mid:]], max_cells))

  def _region_centroid(self, cells):
    """World (x, z) centroid of a region cell set (array indices)."""
    ix0, iz0, _, _ = self._region_grid()
    g = self.grid_cell_size
    arr = np.array(sorted(cells), dtype=np.float32)
    return ((float(arr[:, 0].mean()) + ix0) * g,
            (float(arr[:, 1].mean()) + iz0) * g)

  def _frontier_region_mask(self, frontiers, cells):
    """Boolean mask of frontiers whose grid-frame cell belongs to `cells`."""
    ix0, iz0, _, _ = self._region_grid()
    g = self.grid_cell_size
    xy = self._grid_xy(frontiers)
    i = torch.round(xy[:, 0] / g).long() - ix0
    j = torch.round(xy[:, 1] / g).long() - iz0
    return torch.tensor(
      [(int(a), int(b)) in cells for a, b in zip(i.tolist(), j.tolist())],
      dtype=torch.bool)

  def _finish_task(self, reason):
    """Completes the committed task; region cells go on cooldown."""
    t = self._task
    if t["type"] == "region":
      expiry = time.time() + self.region_retire_cooldown_s
      for c in t["cells"]:
        self._retired[c] = expiry
      logger.info(
        "Region task at (%.1f, %.1f) done: %s (%d cells, %.0fs cooldown).",
        *t["centroid"], reason, len(t["cells"]),
        self.region_retire_cooldown_s)
    else:
      logger.info("Track task (person #%d) done: %s.",
                  t["track_id"], reason)
    self._task = None

  def _score_tasks(self, regions, robot_xy):
    """MAIPP-greedy utilities over region + track candidate tasks.

    Region (explore): w_mass * lambda_sum + w_prob * (1 - exp(-lambda_sum))
      - w_travel * dist, with lambda_sum = target_birth_density * unexplored
      area (v1 PMB-lite: uniform undetected-target birth prior; no
      detector-sweep thinning yet).
    Track (revisit): w_exist * existence + w_cov * trace(P) + w_stale *
      time_since_seen - w_travel * dist, for tracks worth revisiting:
      existence >= track_task_min_existence, position sigma >=
      track_task_min_sigma, AND unseen for at least track_task_min_unseen_s
      (the empirical re-observation cadence; exploration wins until then).
    Returns (best task dict or None, n_region_cands, n_track_cands).
    """
    now = time.time()

    def zone_bonus(x, y):
      """task_priority_bonus when (x, y) lies in the priority zone."""
      z = self.task_priority_zone
      if z is None:
        return 0.0
      xmin, xmax, ymin, ymax = z
      if ((xmin is None or x >= xmin) and (xmax is None or x <= xmax) and
          (ymin is None or y >= ymin) and (ymax is None or y <= ymax)):
        return self.task_priority_bonus
      return 0.0

    cands = []
    cell_area = self.grid_cell_size ** 2
    for cells in regions:
      cx, cy = self._region_centroid(cells)
      dist = math.hypot(cx - robot_xy[0], cy - robot_xy[1])
      lam = self.target_birth_density * len(cells) * cell_area
      score = (self.task_w_mass * lam
               + self.task_w_prob * (1.0 - math.exp(-lam))
               - self.task_w_travel * dist
               + zone_bonus(cx, cy))
      cands.append((score, dict(type="region", cells=cells,
                                centroid=(cx, cy), t_start=now)))
    n_region = len(cands)
    for p in self.people:
      sigma = math.sqrt(0.5 * float(torch.trace(p["cov"])))
      if (p["existence"] < self.track_task_min_existence or
          sigma < self.track_task_min_sigma or
          now - p["last_seen"] < self.track_task_min_unseen_s or
          self._track_claimed(p)):
        continue
      pxy = self._grid_xy(p["pos"].reshape(1, 3))[0]
      dist = math.hypot(float(pxy[0]) - robot_xy[0],
                        float(pxy[1]) - robot_xy[1])
      score = (self.task_w_exist * p["existence"]
               + self.task_w_cov * float(torch.trace(p["cov"]))
               + self.task_w_stale * (now - p["last_seen"])
               - self.task_w_travel * dist
               + zone_bonus(float(pxy[0]), float(pxy[1])))
      cands.append((score, dict(type="track", track_id=p["track_id"],
                                t_start=now, start_n_obs=p["n_obs"])))
    if len(cands) == 0:
      return None, 0, 0
    best = max(cands, key=lambda c: c[0])
    return best[1], n_region, len(cands) - n_region

  def _task_plan(self, frontiers, robot_pose, class_vox):
    """MAIPP-style task layer over frontier/viewpoint selection.

    Maintains one committed task at a time and returns
    (targets, attempt_order) for the viewpoint search:
    - region task (explore): approach via centroid-pulled frontiers, then
      cover the region's own frontiers; completes when the classified
      fraction reaches region_coverage_done or region_time_budget_s
      elapses (cells then cool down for region_retire_cooldown_s).
    - track task (revisit person): the track mean is returned as a single
      pseudo-frontier, so the standard viewpoint search finds a safe pose
      observing the person's location; completes on re-acquisition (a new
      detection collapsed the covariance below track_task_min_sigma),
      track removal, or track_task_time_budget_s.
    Candidates are scored greedily (_score_tasks); falls back to global
    frontier ranking when no task exists.
    """
    unknown = self._unknown_mask()
    unknown_set = set(map(tuple, np.argwhere(unknown).tolist()))
    robot_pos = robot_pose[:3, 3]

    # Completion checks on the committed task.
    if self._task is not None and self._task["type"] == "region":
      cells = self._task["cells"]
      cov = 1.0 - len(cells & unknown_set) / max(len(cells), 1)
      age = time.time() - self._task["t_start"]
      if cov >= self.region_coverage_done:
        self._finish_task("coverage %.0f%% reached" % (cov * 100))
      elif age > self.region_time_budget_s:
        self._finish_task("time budget exceeded (%.0fs, coverage %.0f%%)"
                          % (age, cov * 100))
    elif self._task is not None:  # track task
      track = next((p for p in self.people
                    if p["track_id"] == self._task["track_id"]), None)
      age = time.time() - self._task["t_start"]
      if track is None:
        self._finish_task("track disappeared")
      elif (track["n_obs"] > self._task["start_n_obs"] and
            math.sqrt(0.5 * float(torch.trace(track["cov"])))
            < self.track_task_min_sigma):
        self._finish_task(
          "re-acquired (existence %.2f)" % track["existence"])
      elif age > self.track_task_time_budget_s:
        self._finish_task("time budget exceeded (%.0fs)" % age)

    # Cells retired above (this same cycle) and cells inside fresh peer
    # region claims must not be selectable (claims are selection-only:
    # they do not count as covered for our committed task).
    excl = set(self._retired.keys()) | self._claimed_cells()
    if excl:
      nx, nz = unknown.shape
      for (i, j) in excl:
        if 0 <= i < nx and 0 <= j < nz:
          unknown[i, j] = False
      unknown_set -= excl

    # Commit to the best-scoring task if none is active.
    if self._task is None:
      regions = self._segment_unknown(unknown)
      robot_xy = self._grid_xy(robot_pos.reshape(1, 3))[0]
      task, n_r, n_t = self._score_tasks(
        regions, (float(robot_xy[0]), float(robot_xy[1])))
      if task is None:
        self._vis_regions(unknown_set, class_vox)
        return frontiers, self.rank_frontiers(robot_pose, frontiers)
      self._task = task
      if task["type"] == "region":
        logger.info(
          "Task selected: explore region at (%.1f, %.1f), %.1f m^2 "
          "(%d region / %d track candidates).", *task["centroid"],
          len(task["cells"]) * self.grid_cell_size ** 2, n_r, n_t)
      else:
        logger.info(
          "Task selected: revisit person track #%d "
          "(%d region / %d track candidates).", task["track_id"], n_r, n_t)

    self._vis_regions(unknown_set, class_vox)

    if self._task["type"] == "track":
      track = next((p for p in self.people
                    if p["track_id"] == self._task["track_id"]), None)
      if track is None:
        return frontiers, self.rank_frontiers(robot_pose, frontiers)
      # Track mean as a single pseudo-frontier for the viewpoint search.
      return track["pos"].reshape(1, 3), torch.tensor([0])

    cells = self._task["cells"]
    in_mask = self._frontier_region_mask(frontiers, cells)
    if int(in_mask.sum()) >= self.region_min_frontiers:
      sub = frontiers[in_mask]
      return sub, self.rank_frontiers(robot_pose, sub)
    # Approach: pull toward the region centroid (grid-frame coordinates).
    cx, cy = self._task["centroid"]
    fxy = self._grid_xy(frontiers)
    d_cent = ((fxy[:, 0] - cx) ** 2 + (fxy[:, 1] - cy) ** 2).sqrt()
    d_robot = (frontiers - robot_pos.reshape(1, 3)).norm(dim=-1)
    return frontiers, torch.argsort(d_cent + 1e-3 * d_robot)

  def _vis_regions(self, unknown_set, class_vox):
    """Rerun overlay: unknown cells (gray) + current region (orange)."""
    vis = getattr(self.server, "vis", None)
    if vis is None or not hasattr(vis, "log_pc") or len(unknown_set) == 0:
      return
    try:
      ix0, iz0, _, _ = self._region_grid()
      g = self.grid_cell_size
      gy = float(class_vox[:, 1].median())
      cur = (self._task["cells"] if self._task is not None and
             self._task["type"] == "region" else set())
      cells = sorted(unknown_set)
      xy = torch.tensor(
        [[(i + ix0) * g, (j + iz0) * g] for i, j in cells],
        dtype=torch.float)
      pts = self._grid_xy_to_rdf(xy, gy)
      colors = torch.tensor(
        [[255, 140, 0] if c in cur else [90, 90, 90] for c in cells],
        dtype=torch.uint8)
      radii = torch.full((len(cells),), 0.35 * g)
      vis.log_pc(pts, colors, radii, layer="exploration/regions")
    except Exception:
      logger.exception("Failed to visualize exploration regions.")

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
    # Altitude floor (pose-frame origin = takeoff ground level): anchors on
    # low-sitting ground voxels (quantization, lidar under grass) cannot
    # drag goals below goal_min_altitude. Applied before the view/safety
    # checks so they evaluate the final position.
    if self.goal_min_altitude is not None:
      cand[:, 1] = torch.clamp(cand[:, 1], max=-self.goal_min_altitude)
    # Require actual movement: candidates closer than goal_min_move to the
    # robot are dropped, so consecutive goals never park in place and
    # re-observations happen from a different viewpoint.
    if self.goal_min_move > 0 and robot_pos is not None:
      cand = cand[(cand - robot_pos.reshape(1, 3)).norm(dim=-1)
                  >= self.goal_min_move]
      if cand.shape[0] == 0:
        return None

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
    now = time.time()
    xy = self._grid_xy(pts)
    ix = torch.round(xy[:, 0] / cell_size).long()
    iz = torch.round(xy[:, 1] / cell_size).long()
    ys = pts[:, 1] if y_vals is None else y_vals
    for k in range(pts.shape[0]):
      key = (int(ix[k]), int(iz[k]))
      cell = self.info_grid.get(key)
      if cell is None:
        # state, conf, y, gnd votes, obs votes, last-observed walltime
        cell = [0, 0.0, float(ys[k]), 0, 0, now]
        self.info_grid[key] = cell
      if state == 2:
        cell[4] += 1
      else:
        cell[3] += 1
      cell[2] = float(ys[k])
      cell[5] = now
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
      xy = torch.tensor([[k[0] * g, k[1] * g] for k in keys],
                        dtype=torch.float)
      pts = self._grid_xy_to_rdf(xy, y_plane)
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
      # A frontier that survives several successfully-reached observation
      # goals is not resolving (occluded interior, follower ignoring yaw,
      # stalled semantics, ...); ban it with backoff so the planner does
      # not stare at it forever. Resolved frontiers never come back, so
      # their counts are just inert.
      if self._last_goal_key is not None:
        n = self._reached_counts.get(self._last_goal_key, 0) + 1
        self._reached_counts[self._last_goal_key] = n
        if n >= self.frontier_reobserve_limit:
          ban = self._blacklist_add(self._last_goal_key)
          self._reached_counts.pop(self._last_goal_key, None)
          logger.info(
            "Frontier near %s reached %d times without resolving; banned "
            "for %.0fs.", str(self._last_goal_key), n, ban)
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

    # Exact bounds filter: the mapper generates class frontiers in the RDF
    # AABB *superset* of the (possibly rotated) grid-frame bounds box; drop
    # anything outside the exact box so no goal is ever selected beyond the
    # shared map bounds.
    if self.grid_bounds is not None:
      frontiers = frontiers[self._in_bounds_mask(frontiers)]
      if frontiers.shape[0] == 0:
        return

    # Drop frontiers with an active (unexpired) ban.
    keep = [i for i in range(frontiers.shape[0])
            if not self._blacklisted(self._bl_key(frontiers[i]))]
    if len(keep) == 0:
      logger.info("Exploration: all %d frontiers banned; waiting for new "
                  "frontiers or ban expiry.", frontiers.shape[0])
      return
    frontiers = frontiers[keep]

    robot_pose_cpu = robot_pose.cpu()
    robot_pos = robot_pose_cpu[:3, 3]
    if self.region_mode:
      frontiers, order = self._task_plan(
        frontiers, robot_pose_cpu, class_vox)
      if frontiers.shape[0] == 0:
        return
    else:
      order = self.rank_frontiers(robot_pose_cpu, frontiers)

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
