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

The selected goal is visualized (rerun) and published as a PoseStamped on
goal_topic when ROS is available.
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
  from geometry_msgs.msg import PoseStamped
  from scipy.spatial.transform import Rotation
except ModuleNotFoundError:
  PoseStamped = None

logger = logging.getLogger(__name__)


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
    self._view_max_elev = math.radians(float(mev))
    self.plan_period = float(cfg.plan_period)
    self.max_attempts_per_cycle = int(cfg.max_attempts_per_cycle)
    self.max_candidates = int(cfg.max_candidates)

    self.vox_size = float(self.mapper.vox_size)
    # Blacklist keys are rounded to the frontier cluster grid so recomputed
    # frontiers at the same location stay removed.
    self._bl_grid = self.vox_size * float(
      getattr(self.mapper, "class_frontier_subsampling", 3))
    self._blacklist = set()

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

    # Optional ROS goal publisher on the dataset's node.
    self._goal_pub = None
    ds = getattr(server, "dataset", None)
    if (PoseStamped is not None and ds is not None
        and getattr(ds, "_rosnode", None) is not None):
      self._goal_pub = ds._rosnode.create_publisher(
        PoseStamped, str(cfg.goal_topic), 10)
      logger.info("Exploration goal publisher on %s", cfg.goal_topic)

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

  def _loop(self):
    while not self._stop.is_set():
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
    if cand.shape[0] == 0:
      return None

    # Prefer candidates cheap to reach.
    ref = robot_pos if robot_pos is not None else frontier
    order = torch.argsort((cand - ref.reshape(1, 3)).norm(dim=-1))
    cand = cand[order][:self.max_candidates]

    # Safety: the whole sphere must be known-empty (log-odds < 0).
    C, S = cand.shape[0], self._sphere_offsets.shape[0]
    pts = (cand.reshape(C, 1, 3) +
           self._sphere_offsets.reshape(1, S, 3)).reshape(-1, 3)
    occ = self._query_occ(pts).reshape(C, S)
    safe = (occ < 0).all(dim=-1)

    for i in torch.nonzero(safe).reshape(-1):
      pos = cand[i]
      if self._los_clear(pos, frontier):
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

  # ---------- planning cycle ----------

  def _plan_once(self):
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

    # Drop blacklisted frontiers.
    keep = [i for i in range(frontiers.shape[0])
            if self._bl_key(frontiers[i]) not in self._blacklist]
    if len(keep) == 0:
      logger.info("Exploration: all %d frontiers blacklisted; waiting for "
                  "new frontiers.", frontiers.shape[0])
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
        self._blacklist.add(self._bl_key(f))
        logger.info(
          "Exploration: no safe viewpoint for frontier (%.2f, %.2f, %.2f); "
          "removed (%d blacklisted).", *f.tolist(), len(self._blacklist))
        continue

      self.current_goal = (goal, f)
      logger.info(
        "Exploration goal: pos (%.2f, %.2f, %.2f) observing frontier "
        "(%.2f, %.2f, %.2f).", *goal[:3, 3].tolist(), *f.tolist())
      self._publish_goal(goal, f)
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

    # Publish PoseStamped in the source (pose topic) world frame.
    if self._goal_pub is None:
      return
    ds = self.server.dataset
    src2rdf_inv = torch.linalg.inv(ds.src2rdf)
    goal_src = g3d.transform_pose_4x4(goal_rdf, src2rdf_inv)
    msg = PoseStamped()
    msg.header.frame_id = str(self.cfg.goal_frame_id)
    msg.header.stamp = ds._rosnode.get_clock().now().to_msg()
    t = goal_src[:3, 3].tolist()
    q = Rotation.from_matrix(goal_src[:3, :3].numpy()).as_quat()
    msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = t
    (msg.pose.orientation.x, msg.pose.orientation.y,
     msg.pose.orientation.z, msg.pose.orientation.w) = q.tolist()
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
