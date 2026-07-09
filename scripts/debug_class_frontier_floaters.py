"""Headless bag run + class-frontier floater analysis on the live map.

Run INSIDE the mapping docker container from the repository root:

  python3 scripts/debug_class_frontier_floaters.py \
    --bag ../Datasets/rosbag2_ric_indoor_open --seconds 180

Plays the bag, runs the decoupled pipeline headless (no rerun), queries every
keyframe, and finally reports every class-frontier point that floats above
its local class surface, with the occupancy context (occ above/below,
below-is-class) that explains WHY it survived the filters.
"""
import argparse
import logging
import os
import subprocess
import sys
import threading
import time

import torch
from hydra import compose, initialize_config_dir

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")

from rayfronts.mapping_server import MappingServer  # noqa: E402
from rayfronts.mapping.semantic_ray_frontiers_map import rayfronts_cpp  # noqa: E402


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--bag", default="../Datasets/rosbag2_ric_indoor_open")
  ap.add_argument("--seconds", type=int, default=180)
  ap.add_argument("--config-name", default="starlingmax_decoupled_bag")
  args = ap.parse_args()

  print("cuda available:", torch.cuda.is_available(), flush=True)

  root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  os.chdir(root)
  pd = os.path.join(root, "experiments/preset_configs")
  sp = "hydra.searchpath=[file://" + os.path.join(root, "rayfronts/configs") + "]"
  with initialize_config_dir(config_dir=pd, version_base=None):
    cfg = compose(config_name=args.config_name,
                  overrides=[sp, "~vis", "~messaging_service",
                             "querying.period=1"])

  bag = subprocess.Popen(
    ["ros2", "bag", "play", args.bag, "--loop"],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
  print("bag playing, building server...", flush=True)

  server = MappingServer(cfg)
  t = threading.Thread(target=server.run, daemon=True)
  t.start()

  for i in range(max(1, args.seconds // 30)):
    time.sleep(30)
    with server.map_lock:
      mp = server.mapper
      occ = "empty" if mp.occ_map_vdb.empty() else "nonempty"
      sv = 0 if mp.global_vox_xyz is None else mp.global_vox_xyz.shape[0]
      cv = 0 if mp.class_voxels_xyz is None else mp.class_voxels_xyz.shape[0]
      cfn = 0 if mp.class_frontiers is None else mp.class_frontiers.shape[0]
    print(f"[t={30*(i+1):3d}s] frames={server.dataset._frame_out_idx} "
          f"occ={occ} sem_vox={sv} class_vox={cv} class_frontiers={cfn}",
          flush=True)

  vox = float(server.mapper.vox_size)
  with server.map_lock:
    cf = server.mapper.class_frontiers
    cls = server.mapper.class_voxels_xyz
    if cf is None or cls is None:
      print("NO class frontiers / class voxels produced!", flush=True)
    else:
      cf = cf.detach().cpu().clone()
      cls = cls.detach().cpu().clone()
      print(f"\nclass_frontiers: {cf.shape[0]} | class voxels: {cls.shape[0]}")
      floaters = 0
      for i in range(cf.shape[0]):
        p = cf[i]
        lat = (cls[:, [0, 2]] - p[[0, 2]].reshape(1, 2)).norm(dim=-1)
        near = cls[lat <= 0.3]
        if near.shape[0] == 0:
          floaters += 1
          print(f"  FLOATER ({p[0]:6.2f},{p[1]:6.2f},{p[2]:6.2f}) "
                "no class voxel in column")
          continue
        top_y = near[:, 1].min()  # up = -y
        dy = float(top_y - p[1])  # >0 => above local class top
        if dy > 0.15:
          floaters += 1
          below = torch.round((p + torch.tensor([0., vox, 0.])) / vox) * vox
          above = torch.round((p - torch.tensor([0., vox, 0.])) / vox) * vox
          ob = int(rayfronts_cpp.query_occ(
            server.mapper.occ_map_vdb, below.reshape(1, 3)).reshape(-1)[0])
          oa = int(rayfronts_cpp.query_occ(
            server.mapper.occ_map_vdb, above.reshape(1, 3)).reshape(-1)[0])
          bic = bool(((cls - below.reshape(1, 3)).norm(dim=-1)
                      < vox * 0.5).any())
          print(f"  FLOATER ({p[0]:6.2f},{p[1]:6.2f},{p[2]:6.2f}) "
                f"dy={dy:5.2f} occ_below={ob} occ_above={oa} "
                f"below_is_class={bic}")
      print(f"floaters above local class surface: {floaters}/{cf.shape[0]}")

  with server._status_lock:
    server.status = MappingServer.Status.CLOSING
  try:
    server.dataset.shutdown()
  except Exception:
    pass
  bag.terminate()
  print("done.", flush=True)


if __name__ == "__main__":
  main()
