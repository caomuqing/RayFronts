"""Calibrate point_cloud_to_world_transform from a recorded bag.

Estimates the tilt (roll/pitch) and height (z) components of the static
transform placing the registered-scan world (e.g. superodom_map) into the
pose world frame (e.g. PX4 map) — the components whose error makes carve rays
cut through the ground at range. Yaw and xy translation are kept from the
current transform (they do not affect ground carving).

Method: the PX4 map frame is gravity-aligned, and at bag start the drone sits
on the ground. So the dominant ground plane of the accumulated registered
cloud, after the *correct* transform, must (a) have an exactly vertical
normal and (b) sit at ground height = first body position + ground clearance.
The script fits that plane (normal-prior RANSAC + SVD refine), reports the
current transform's residual tilt/height error, and prints a corrected
[x y z qx qy qz qw] ready to paste into the preset.

Run from the repository root (ROS libs needed for message deserialization,
no ROS runtime / GPU required):

  python3 scripts/calibrate_scan_transform.py \
    --bag ../Datasets/rosbag2_ric_indoor_open \
    --ground-clearance 0.10
"""
import argparse
import glob
import math
import os
import sqlite3
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rclpy.serialization import deserialize_message  # noqa: E402
from sensor_msgs.msg import PointCloud2  # noqa: E402
from nav_msgs.msg import Odometry  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from rayfronts.ros_utils import pointcloud2_to_array  # noqa: E402


def read_bag(bag_dir, pc_topic, pose_topic, scan_stride, max_scans,
             max_poses):
  db = glob.glob(os.path.join(bag_dir, "*.db3"))
  if not db:
    raise FileNotFoundError(f"No .db3 in {bag_dir}")
  con = sqlite3.connect(db[0])
  cur = con.cursor()
  tmap = {n: i for i, n, _ in
          cur.execute("SELECT id, name, type FROM topics").fetchall()}
  for t in (pc_topic, pose_topic):
    if t not in tmap:
      raise ValueError(f"topic {t} not in bag (has: {list(tmap)})")

  clouds = []
  rows = cur.execute(
    "SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp",
    (tmap[pc_topic],)).fetchall()
  for i in range(0, len(rows), scan_stride):
    if len(clouds) >= max_scans:
      break
    msg = deserialize_message(bytes(rows[i][0]), PointCloud2)
    arr = pointcloud2_to_array(msg, squeeze=True).reshape(-1)
    xyz = np.stack([np.asarray(arr["x"], np.float64),
                    np.asarray(arr["y"], np.float64),
                    np.asarray(arr["z"], np.float64)], axis=-1)
    xyz = xyz[np.isfinite(xyz).all(axis=-1)]
    clouds.append(xyz)
  cloud = np.concatenate(clouds, axis=0)

  poses = []
  rows = cur.execute(
    "SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp LIMIT ?",
    (tmap[pose_topic], max_poses)).fetchall()
  for (blob,) in rows:
    msg = deserialize_message(bytes(blob), Odometry)
    p = msg.pose.pose.position
    poses.append([p.x, p.y, p.z])
  return cloud, np.array(poses), len(clouds)


def voxel_downsample(xyz, vox):
  key = np.round(xyz / vox).astype(np.int64)
  _, idx = np.unique(key, axis=0, return_index=True)
  return xyz[idx]


def transform_from_xyz_qxyzw(v):
  T = np.eye(4)
  T[:3, :3] = Rotation.from_quat(v[3:]).as_matrix()
  T[:3, 3] = v[:3]
  return T


def fit_ground_plane(pts_map, up, max_tilt_deg=25.0, iters=800, thresh=0.05,
                     seed=0):
  """Normal-prior RANSAC + SVD refine. Returns (normal(up-signed), centroid,
  inlier_count)."""
  rng = np.random.default_rng(seed)
  n_pts = pts_map.shape[0]
  best = (None, None, -1)
  cos_max = math.cos(math.radians(max_tilt_deg))
  for _ in range(iters):
    i = rng.choice(n_pts, 3, replace=False)
    p0, p1, p2 = pts_map[i]
    n = np.cross(p1 - p0, p2 - p0)
    nn = np.linalg.norm(n)
    if nn < 1e-9:
      continue
    n = n / nn
    if n @ up < 0:
      n = -n
    if n @ up < cos_max:
      continue  # not ground-like
    d = np.abs((pts_map - p0) @ n)
    cnt = int((d < thresh).sum())
    if cnt > best[2]:
      best = (n, p0, cnt)
  if best[2] < 100:
    raise RuntimeError("Ground plane not found (too few inliers). "
                       "Is the environment mostly non-flat?")
  n, p0, _ = best
  # Refine on inliers via SVD (two rounds).
  for _ in range(2):
    inl = pts_map[np.abs((pts_map - p0) @ n) < thresh]
    c = inl.mean(axis=0)
    _, _, vt = np.linalg.svd(inl - c, full_matrices=False)
    n = vt[-1]
    if n @ up < 0:
      n = -n
    p0 = c
  inl = pts_map[np.abs((pts_map - p0) @ n) < thresh]
  return n, p0, inl.shape[0]


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--bag", required=True)
  ap.add_argument("--pc-topic", default="/registered_scan")
  ap.add_argument("--pose-topic", default="/odom_px4_fused")
  ap.add_argument("--current-transform", type=float, nargs=7,
                  default=[0.0, 0.0, 0.0,
                           0.1800308, -0.1800308, 0.6838047, -0.6838047],
                  help="x y z qx qy qz qw (src/FRD frame), as in the preset")
  ap.add_argument("--ground-clearance", type=float, default=0.10,
                  help="Body-origin height above ground when landed (m)")
  ap.add_argument("--scan-stride", type=int, default=5)
  ap.add_argument("--max-scans", type=int, default=60)
  ap.add_argument("--downsample", type=float, default=0.1)
  args = ap.parse_args()

  # NOTE: assumes FRD pose world frame (PX4): gravity-aligned, down = +z.
  up = np.array([0.0, 0.0, -1.0])

  print("reading bag...", flush=True)
  cloud_super, poses, n_scans = read_bag(
    args.bag, args.pc_topic, args.pose_topic, args.scan_stride,
    args.max_scans, max_poses=400)
  print(f"  scans used: {n_scans} | raw points: {cloud_super.shape[0]}")
  cloud_super = voxel_downsample(cloud_super, args.downsample)
  print(f"  downsampled points: {cloud_super.shape[0]}")

  # Ground height in the pose world frame: the drone starts on the ground.
  z0 = poses[:40, 2]
  if z0.std() > 0.15:
    print(f"  WARNING: first poses vary {z0.std():.2f}m in z — drone may "
          "already be flying at bag start; --ground-clearance/ground height "
          "may be unreliable.")
  z_ground = float(np.median(z0)) + args.ground_clearance  # FRD: down = +z
  print(f"  body z at start: {np.median(z0):.3f} -> ground z: {z_ground:.3f}")

  # Current transform: superodom -> map (both in src/FRD coords).
  T_cur = transform_from_xyz_qxyzw(args.current_transform)
  pts_map = cloud_super @ T_cur[:3, :3].T + T_cur[:3, 3]

  n, c, n_inl = fit_ground_plane(pts_map, up)
  tilt_deg = math.degrees(math.acos(np.clip(n @ up, -1, 1)))
  dz = c[2] - z_ground  # plane below expected ground (FRD: +z down)
  print(f"\ncurrent transform residuals:")
  print(f"  ground-plane inliers: {n_inl}")
  print(f"  tilt vs gravity:      {tilt_deg:6.3f} deg")
  print(f"  height offset:        {dz*100:+6.1f} cm "
        "(plane_z - expected_ground_z)")

  # Correction: minimal rotation carrying n -> up, applied about the plane
  # centroid (keeps yaw and lateral placement), plus a z snap to z_ground.
  axis = np.cross(n, up)
  s = np.linalg.norm(axis)
  if s < 1e-9:
    dR = np.eye(3)
  else:
    dR = Rotation.from_rotvec(
      axis / s * math.asin(np.clip(s, -1, 1))).as_matrix()
  dt = c - dR @ c
  dt[2] += z_ground - c[2]

  R_new = dR @ T_cur[:3, :3]
  t_new = dR @ T_cur[:3, 3] + dt
  q_new = Rotation.from_matrix(R_new).as_quat()  # xyzw

  # Verify.
  pts2 = cloud_super @ R_new.T + t_new
  n2, c2, _ = fit_ground_plane(pts2, up)
  tilt2 = math.degrees(math.acos(np.clip(n2 @ up, -1, 1)))
  print(f"\nafter correction (verified on the cloud):")
  print(f"  tilt vs gravity:      {tilt2:6.3f} deg")
  print(f"  height offset:        {(c2[2]-z_ground)*100:+6.1f} cm")

  vals = [*t_new.tolist(), *q_new.tolist()]
  print("\ncorrected point_cloud_to_world_transform "
        "(paste into the preset):")
  print("  point_cloud_to_world_transform:")
  print("    [" + ", ".join(f"{v:.7f}" for v in vals) + "]")

  if tilt_deg < 0.3 and abs(dz) < 0.05:
    print("\nNOTE: current transform is already good (tilt <0.3deg, "
          "height <5cm) — far-point flicker is registration jitter, "
          "not the static transform.")


if __name__ == "__main__":
  main()
