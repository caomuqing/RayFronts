"""Person detection projection + Bernoulli-KF tracking for the high flyer.

Port of the low-flying drone's people pipeline (see
reference_only/RayFronts_low/rayfronts/exploration_planner.py, "people
detection" section), adapted to the high drone's detection source:

  - Detections arrive on /robot_1/gimbal/boundingbox
    (vision_msgs/Detection2DArray), produced on the front-stereo left camera
    frames but with bbox pixels in the camera's NATIVE resolution
    (det_resolution, e.g. 1920x1080), not the downscaled republished image.
  - The camera pose at the detection stamp is reconstructed from the raw
    odometry topic exactly the way the mapping dataset does it (fixed camera
    pitch offset, then FLU->RDF), instead of the dataset's synced pose
    buffer (detections outnumber synced mapping frames).
  - The ground height for ray intersection comes from the 2D info grid's
    ground cells (median display height) instead of the low mapper's
    class_voxels_xyz.

Track state and semantics are identical to the low drone (Bernoulli
existence with Kalman position on the ground plane), so the exchanged
/maipp/tracks messages fuse symmetrically on either robot.

The class is ROS-free except for message *construction* (imported lazily),
so the projection/association logic is unit-testable without a ROS runtime.
"""

import logging
import math
import threading
import time
from collections import deque

import numpy as np
import torch

from rayfronts import geometry3d as g3d

logger = logging.getLogger(__name__)


class PersonTracker:
  """Projects person bboxes to the ground and tracks them on the map.

  Attributes:
    people: List of track dicts (track_id, pos [world RDF, y=ground],
      cov [2x2 planar (x_rdf, z_rdf)], existence, status, n_obs, last_seen,
      score, last_neg_stamp).
  """

  def __init__(self,
               intrinsics_3x3,
               det_resolution=(1920, 1080),
               src2rdf=None,
               cam_pitch_rad=0.261799,
               person_class_id="PERSON",
               person_min_score=0.5,
               person_merge_radius=2.5,
               detection_max_range=60.0,
               detection_max_pose_dt=0.5,
               person_p_detect=0.7,
               person_exist_remove=0.2,
               person_exist_confirm=0.8,
               person_process_noise=0.02,
               person_meas_noise_base=0.3,
               person_meas_noise_per_m=0.08,
               person_gate_chi2=9.21,
               person_negative_min_interval_s=0.5,
               vox_size=0.5,
               robot_id=1):
    """
    Args:
      intrinsics_3x3: 3x3 float tensor; camera intrinsics IN det_resolution
        pixel space (scale your camera_info K to det_resolution first).
      det_resolution: (width, height) of the detector's pixel space.
      src2rdf: 4x4 float tensor converting the odometry source coordinate
        system to RDF (the mapping dataset's src2rdf_transform).
      cam_pitch_rad: Fixed camera pitch-down offset applied to the body
        pose, matching the mapping dataset's pose handling.
      person_class_id: class_id string that counts as a person.
      person_min_score: Minimum detector score to accept a detection.
      person_merge_radius: Euclidean association floor in meters.
      detection_max_range: Discard ground intersections farther than this
        (grazing rays at high altitude are unreliable).
      detection_max_pose_dt: Max |stamp difference| between a detection and
        the nearest odometry sample to accept the projection.
      person_p_detect: Detector sensitivity used in the Bernoulli existence
        updates (both positive and missed-detection).
      person_exist_remove: Tracks below this existence are deleted.
      person_exist_confirm: Tracks above this (with >= 2 obs) are confirmed.
      person_process_noise: Covariance growth (m^2/s) while unobserved.
      person_meas_noise_base: Measurement sigma at zero range (m).
      person_meas_noise_per_m: Additional sigma per meter of ground range.
      person_gate_chi2: Mahalanobis association gate (chi2, 2 dof).
      person_negative_min_interval_s: Rate limit for missed-detection
        (negative) updates per track, in message time.
      vox_size: Mapper voxel size (line-of-sight stepping).
      robot_id: This robot's id for track message ids ("<rid>:<tid>").
    """
    self.K = torch.as_tensor(intrinsics_3x3, dtype=torch.float)
    self.det_w, self.det_h = int(det_resolution[0]), int(det_resolution[1])
    self.src2rdf = (src2rdf if src2rdf is not None
                    else torch.eye(4, dtype=torch.float))
    # Camera-from-body: fixed pitch about the FLU left axis, exactly as the
    # mapping dataset applies it to the odometry pose.
    p = float(cam_pitch_rad)
    self.T_body_cam = torch.tensor(
      [[math.cos(p), 0.0, math.sin(p), 0.0],
       [0.0, 1.0, 0.0, 0.0],
       [-math.sin(p), 0.0, math.cos(p), 0.0],
       [0.0, 0.0, 0.0, 1.0]], dtype=torch.float)

    self.person_class_id = str(person_class_id)
    self.person_min_score = float(person_min_score)
    self.person_merge_radius = float(person_merge_radius)
    self.detection_max_range = float(detection_max_range)
    self.detection_max_pose_dt = float(detection_max_pose_dt)
    self.person_p_detect = float(person_p_detect)
    self.person_exist_remove = float(person_exist_remove)
    self.person_exist_confirm = float(person_exist_confirm)
    self.person_process_noise = float(person_process_noise)
    self.person_meas_noise_base = float(person_meas_noise_base)
    self.person_meas_noise_per_m = float(person_meas_noise_per_m)
    self.person_gate_chi2 = float(person_gate_chi2)
    self.person_negative_min_interval_s = float(person_negative_min_interval_s)
    self.vox_size = float(vox_size)
    self.robot_id = int(robot_id)

    self.people = []
    self._next_track_id = 1
    self._people_prev_predict = time.time()
    self._ground_y = None  # world-RDF ground height (up = -y)
    self._imported = {}    # peer "rid:tid" -> local track_id

    self._det_lock = threading.Lock()
    self._det_queue = deque(maxlen=200)
    self._pose_lock = threading.Lock()
    self._pose_buf = deque(maxlen=400)  # (stamp_ns, 4x4 src-frame body pose)

  # ---------- ROS-side feeders (called from subscription callbacks) ----------

  def on_detections(self, msg):
    """Queues person bbox bottom-centers; projection happens in update()."""
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

  def on_odom(self, msg):
    """Buffers raw odometry poses for stamp-matched projection."""
    stamp_ns = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
    pose = _pose_to_4x4(msg.pose.pose)
    with self._pose_lock:
      self._pose_buf.append((stamp_ns, pose))

  def set_ground_height(self, y_rdf):
    """Sets the world-RDF ground height detections are intersected with."""
    self._ground_y = float(y_rdf)

  # ---------- projection + tracking (called from the mapping loop) ----------

  def _lookup_cam_pose_rdf(self, stamp_ns):
    """Camera pose (4x4 world RDF) nearest to stamp_ns and |dt| seconds."""
    with self._pose_lock:
      buf = list(self._pose_buf)
    if len(buf) == 0:
      return None
    diffs = [abs(t - stamp_ns) for t, _ in buf]
    i = diffs.index(min(diffs))
    src_pose = buf[i][1] @ self.T_body_cam
    return g3d.transform_pose_4x4(src_pose, self.src2rdf), diffs[i] / 1e9

  def update(self, mapper=None):
    """Projects queued detections to the ground and merges them into tracks.

    Bottom-center pixels (native det_resolution space) are cast through the
    camera pose at the detection stamp and intersected with the ground
    plane at the info-grid ground height. Safe to call every mapping loop
    iteration; cheap when the queue is empty.
    """
    with self._det_lock:
      batches = list(self._det_queue)
      self._det_queue.clear()
    if len(batches) == 0:
      return False
    gy = self._ground_y
    if gy is None:
      return False  # no classified ground yet to intersect with

    # Kalman predict: static-target model, so covariance grows with time
    # while a person goes unobserved (widens the association gate and
    # quantifies revisit value).
    now = time.time()
    dt = max(0.0, now - self._people_prev_predict)
    self._people_prev_predict = now
    if dt > 0.0:
      q = self.person_process_noise * dt
      for person in self.people:
        person["cov"] = person["cov"] + q * torch.eye(2)

    fx, fy = float(self.K[0, 0]), float(self.K[1, 1])
    cx, cy = float(self.K[0, 2]), float(self.K[1, 2])
    changed = False
    for stamp_ns, dets in batches:
      r = self._lookup_cam_pose_rdf(stamp_ns)
      if r is None or r[1] > self.detection_max_pose_dt:
        continue
      rot, cam_pos = r[0][:3, :3], r[0][:3, 3]
      matched = set()
      for u, v, score in dets:
        # ZED images are rectified; plain pinhole normalization suffices.
        ray = rot @ torch.tensor(
          [(u - cx) / fx, (v - cy) / fy, 1.0], dtype=torch.float)
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
        changed |= self._negative_pass(rot, cam_pos, matched, stamp_ns, mapper)
    return changed

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

  def _negative_pass(self, rot, cam_pos, matched, stamp_ns, mapper):
    """Counts missed observations for people the frame should have seen.

    A tracked person not matched by any detection in this frame accrues a
    negative observation only when the detector genuinely should have seen
    them: BOTH a low body point (0.3m above ground) AND the mid-body point
    (0.9m) project well inside the image (10% border margin), lie in front
    of the camera within detection_max_range, and have occlusion-free lines
    of sight. Negatives are rate-limited per track; each counted miss
    applies the Bernoulli missed-detection update
    r <- r(1-p_d)/(1-r*p_d); dropping below person_exist_remove deletes
    the track.
    """
    fx, fy = float(self.K[0, 0]), float(self.K[1, 1])
    cx, cy = float(self.K[0, 2]), float(self.K[1, 2])
    W, H = float(self.det_w), float(self.det_h)
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
        if not self._los_clear(cam_pos, pt, mapper):
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

  def _los_clear(self, p, f, mapper):
    """True if the segment p->f has no occupied voxel (stops ~1 vox short)."""
    if mapper is None:
      return True
    try:
      import rayfronts_cpp
    except ImportError:
      return True
    d = f - p
    length = float(d.norm())
    if length < self.vox_size:
      return True
    n = max(2, int(length / (self.vox_size * 0.5)))
    ts = torch.linspace(0.0, max(0.0, 1.0 - self.vox_size / length), n)
    pts = p.reshape(1, 3) + ts.reshape(-1, 1) * d.reshape(1, 3)
    occ = rayfronts_cpp.query_occ(mapper.occ_map_vdb, pts.cpu())
    return bool((occ <= 0).all())

  def fuse_peer_tracks(self, dets, geo_frame):
    """Merges a peer's track list into the local Bernoulli-KF tracks.

    Port of the low drone's _fuse_peer_tracks. Idempotent under periodic
    re-reception: a matched track adopts the peer's mean/covariance only
    when the peer is more certain (smaller covariance trace); existence
    and last_seen take the max. Unmatched peer tracks are imported as
    local tracks (remembering their foreign id so later messages update
    rather than duplicate).

    Args:
      dets: list of (key, frame_x, frame_y, cov4, existence, stamp_s)
        tuples parsed from a peer Detection3DArray (frontier-frame
        coordinates; cov4 = planar covariance rows 0/1/6/7).
      geo_frame: FrontierFrame for frame->local conversion.
    """
    if geo_frame is None or not geo_frame.is_ready():
      return False
    gy = self._ground_y
    if gy is None:
      if len(self.people) > 0:
        gy = float(self.people[0]["pos"][1])
      else:
        return False  # no ground estimate yet; peers republish periodically
    # Planar frame-xy -> (x_rdf, z_rdf): inverse of the rotation used in
    # build_tracks_msg.
    A2 = torch.tensor(np.asarray(geo_frame._affine_R)[:2, :2],
                      dtype=torch.float)
    P = torch.tensor([[0.0, 1.0], [-1.0, 0.0]])
    R2_inv = torch.linalg.inv(A2 @ P)
    changed = False
    for key, fx, fy, cov4, r_rem, stamp in dets:
      if key.startswith("%d:" % self.robot_id):
        continue  # our own track echoed back through a relay
      local = geo_frame.frame_to_local(np.array([fx, fy, 0.0]))
      # local ENU -> world RDF: (x_r, y_r, z_r) = (-y_l, -z_l, x_l), with
      # the height replaced by the ground estimate.
      pos = torch.tensor([-float(local[1]), float(gy), float(local[0])])
      P_frame = torch.tensor([[cov4[0], cov4[1]], [cov4[2], cov4[3]]],
                             dtype=torch.float)
      P_rem = R2_inv @ P_frame @ R2_inv.T
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
        target["existence"] = max(target["existence"], float(r_rem))
        target["last_seen"] = max(target["last_seen"], float(stamp))
        target["status"] = self._person_status(target)
        changed = True
      else:
        self.people.append(dict(
          track_id=self._next_track_id, pos=pos.clone(), cov=P_rem,
          existence=float(r_rem), status="candidate", n_obs=1,
          last_seen=float(stamp), score=float(r_rem),
          last_neg_stamp=-10**18))
        self._imported[key] = self._next_track_id
        logger.info("Imported person track %s as #%d at (%.1f, %.1f).",
                    key, self._next_track_id, float(pos[0]), float(pos[2]))
        self._next_track_id += 1
        changed = True
    return changed

  # ---------- outgoing messages ----------

  def build_tracks_msg(self, stamp, geo_frame):
    """Detection3DArray on the MAIPP wire format, in frontier-frame coords.

    Mirrors the low drone's _build_tracks_msg: id "<rid>:<tid>", score =
    existence, planar covariance in rows 0/1/6/7, header stamp = last_seen.
    Returns None when the frontier frame is not ready or vision_msgs is
    unavailable.
    """
    try:
      from vision_msgs.msg import (Detection3D, Detection3DArray,
                                   ObjectHypothesisWithPose)
    except ImportError:
      return None
    if geo_frame is None or not geo_frame.is_ready():
      return None
    arr = Detection3DArray()
    arr.header.stamp = stamp
    arr.header.frame_id = "frontier_frame"
    # Planar rotation (x_rdf, z_rdf) -> frontier-frame xy: RDF to local ENU
    # planar axes is (x_l, y_l) = (z_rdf, -x_rdf), then the affine rotation.
    A2 = torch.tensor(np.asarray(geo_frame._affine_R)[:2, :2],
                      dtype=torch.float)
    P = torch.tensor([[0.0, 1.0], [-1.0, 0.0]])
    R2 = A2 @ P
    for p in self.people:
      det = Detection3D()
      det.header.frame_id = "frontier_frame"
      det.header.stamp.sec = int(p["last_seen"])
      det.header.stamp.nanosec = int((p["last_seen"] % 1.0) * 1e9)
      det.id = "%d:%d" % (self.robot_id, p["track_id"])
      hyp = ObjectHypothesisWithPose()
      hyp.hypothesis.class_id = self.person_class_id
      hyp.hypothesis.score = float(p["existence"])
      pos = p["pos"]
      local = np.array([float(pos[2]), -float(pos[0]), -float(pos[1])])
      pf = geo_frame.local_to_frame(local)
      hyp.pose.pose.position.x = float(pf[0])
      hyp.pose.pose.position.y = float(pf[1])
      P_frame = R2 @ p["cov"] @ R2.T
      cov = [0.0] * 36
      cov[0] = float(P_frame[0, 0])
      cov[1] = float(P_frame[0, 1])
      cov[6] = float(P_frame[1, 0])
      cov[7] = float(P_frame[1, 1])
      hyp.pose.covariance = cov
      det.results.append(hyp)
      arr.detections.append(det)
    return arr

  def build_markers_msg(self, stamp, marker_cls, marker_array_cls):
    """RViz MarkerArray of tracked people in the 'map' (local ENU) frame.

    Sphere radius reflects positional uncertainty; color the lifecycle
    (confirmed red, candidate orange, stale gray). A DELETEALL leads so
    removed tracks disappear.
    """
    arr = marker_array_cls()
    clear = marker_cls()
    clear.header.frame_id = "map"
    clear.header.stamp = stamp
    clear.ns = "people_tracks"
    clear.action = marker_cls.DELETEALL
    arr.markers.append(clear)
    status_rgb = {"confirmed": (0.9, 0.12, 0.12),
                  "candidate": (1.0, 0.59, 0.16),
                  "stale": (0.55, 0.35, 0.35)}
    for p in self.people:
      m = marker_cls()
      m.header.frame_id = "map"
      m.header.stamp = stamp
      m.ns = "people_tracks"
      m.id = int(p["track_id"])
      m.type = marker_cls.SPHERE
      m.action = marker_cls.ADD
      pos = p["pos"]
      # world RDF -> local ENU ('map') via [z, -x, -y].
      m.pose.position.x = float(pos[2])
      m.pose.position.y = float(-pos[0])
      m.pose.position.z = float(-pos[1])
      m.pose.orientation.w = 1.0
      r = min(2.5, 0.25 + math.sqrt(0.5 * float(torch.trace(p["cov"]))))
      m.scale.x = m.scale.y = m.scale.z = 2.0 * r
      c = status_rgb.get(p["status"], status_rgb["candidate"])
      m.color.r, m.color.g, m.color.b = c
      m.color.a = 0.75
      arr.markers.append(m)
    return arr


def _pose_to_4x4(pose):
  """geometry_msgs/Pose -> 4x4 float tensor."""
  q = pose.orientation
  t = pose.position
  x, y, z, w = float(q.x), float(q.y), float(q.z), float(q.w)
  R = torch.tensor([
    [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
  ], dtype=torch.float)
  T = torch.eye(4, dtype=torch.float)
  T[:3, :3] = R
  T[0, 3], T[1, 3], T[2, 3] = float(t.x), float(t.y), float(t.z)
  return T
