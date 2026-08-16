#!/usr/bin/env python3
"""Multi-robot MAIPP visualization of the merged 3dronetest bag.

Renders, in the shared frontier frame, an animation with per-robot colors:
  - drone poses + trails (robot1 high flyer, robot2/3 starlings)
  - each robot's coverage-grid contribution (translucent fill, obstacles darker)
  - people tracks each robot publishes on .../maipp/tracks (X markers)
  - the claimed task on .../maipp/task_claim: region -> rectangle outline,
    person track -> star at the claimed point + dashed line from the robot
    (the raw 1-point polygon that RViz can't display).

Pose transforms into the frontier frame follow the reference code:
  - robot1: local ENU odometry georeferenced online from mavros GPS fixes
    (FrontierFrame.update_home, same as mapping_server_rosnode.py).
  - robot2/3: NED odometry through the surveyed ned_origins entries in
    RayFronts_small/experiments/preset_configs/starlingmax_decoupled_bag.yaml.

Run inside the robot container:
  python3 visualize_merged_bag.py --bag /bags/merged_3dronetest \
      --out /bags/viz_3dronetest.mp4
"""

import argparse
import math
import os
import sys

import numpy as np

import importlib.util as _ilu

_spec = _ilu.spec_from_file_location(
    "geo_frame",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "RayFronts_high", "rayfronts", "geo_frame.py"))
_geo = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_geo)
FrontierFrame = _geo.FrontierFrame

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import animation  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import rosbag2_py  # noqa: E402
from rclpy.serialization import deserialize_message  # noqa: E402
from rosidl_runtime_py.utilities import get_message  # noqa: E402

# Frontier frame definition (mapping_server_rosnode.py:105-109 /
# starlingmax_decoupled_bag.yaml ff_*).
FF_ORIGIN = dict(origin_lat=40.413131, origin_lon=-79.946393,
                 origin_alt=220.80565047594274, heading_deg=-76.17)
# Surveyed NED odometry origins for the starlings
# (starlingmax_decoupled_bag.yaml ned_origins).
NED_ORIGINS = {
    2: dict(lat=40.41329, lon=-79.94657, alt=0.0, heading_deg=166.17),
    3: dict(lat=40.41332, lon=-79.94648, alt=0.0, heading_deg=166.17),
}
# Bag-replay fallback if GPS never locks (mapping_server_rosnode.py:111-114).
FIX_HOME_R1 = (40.41350, -79.94658, 220.8)

ROBOTS = {
    1: dict(color="#1f77b4", name="robot_1 (high)"),
    2: dict(color="#ff7f0e", name="robot_2 (starling)"),
    3: dict(color="#2ca02c", name="robot_3 (starling)"),
}

TOPICS = {
    "/robot_1/odometry_conversion/odometry": ("odom", 1),
    "/robot_1/interface/mavros/global_position/global": ("gps", 1),
    "/robot_2/odom_laser_ned_relay": ("odom", 2),
    "/robot_3/odom_laser_ned_relay": ("odom", 3),
}
for _rid in (1, 2, 3):
    TOPICS[f"/robot_{_rid}/maipp/coverage_grid"] = ("coverage", _rid)
    TOPICS[f"/robot_{_rid}/maipp/tracks"] = ("tracks", _rid)
    TOPICS[f"/robot_{_rid}/maipp/task_claim"] = ("claim", _rid)


def ned_to_frame_transform(rid):
    """Returns (R 3x3, t 3) mapping starling NED odometry -> frontier frame."""
    o = NED_ORIGINS[rid]
    ff = FrontierFrame(**FF_ORIGIN)
    ff.set_home(o["lat"], o["lon"], o["alt"])
    psi = math.radians(o["heading_deg"])
    # NED -> ENU at home (exploration_planner.py:459-462).
    r_n2e = np.array([[math.sin(psi), math.cos(psi), 0.0],
                      [math.cos(psi), -math.sin(psi), 0.0],
                      [0.0, 0.0, -1.0]])
    return ff._affine_R @ r_n2e, ff._affine_t.copy()


def read_bag(bag_path):
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3"),
                rosbag2_py.ConverterOptions("", ""))
    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
    reader.set_filter(rosbag2_py.StorageFilter(topics=list(TOPICS)))
    msg_classes = {n: get_message(type_map[n]) for n in TOPICS if n in type_map}

    data = {rid: dict(odom=[], coverage=[], tracks=[], claim=[])
            for rid in ROBOTS}
    gps = []
    n = 0
    while reader.has_next():
        topic, raw, t_ns = reader.read_next()
        kind, rid = TOPICS[topic]
        msg = deserialize_message(raw, msg_classes[topic])
        t = t_ns * 1e-9
        if kind == "odom":
            p = msg.pose.pose.position
            data[rid]["odom"].append((t, p.x, p.y, p.z))
        elif kind == "gps":
            if not math.isnan(msg.latitude):
                gps.append((t, msg.latitude, msg.longitude, msg.altitude))
        elif kind == "coverage":
            grid = np.array(msg.data, dtype=np.int8).reshape(
                msg.info.height, msg.info.width)
            data[rid]["coverage"].append(
                (t, msg.info.origin.position.x, msg.info.origin.position.y,
                 float(msg.info.resolution), grid))
        elif kind == "tracks":
            people = []
            for det in msg.detections:
                if not det.results:
                    continue
                hyp = det.results[0]
                people.append((str(det.id),
                               hyp.pose.pose.position.x,
                               hyp.pose.pose.position.y,
                               float(hyp.hypothesis.score)))
            data[rid]["tracks"].append((t, people))
        elif kind == "claim":
            pts = [(p.x, p.y) for p in msg.polygon.points]
            data[rid]["claim"].append((t, pts))
        n += 1
        if n % 10000 == 0:
            print(f"  read {n} messages...", flush=True)
    print(f"  done: {n} messages.", flush=True)
    return data, gps


def estimate_r1_home(gps, odom):
    """Replays FrontierFrame.update_home over the bag's GPS + ENU odometry."""
    ff = FrontierFrame(**FF_ORIGIN)
    if not gps or not odom:
        print("WARN: no GPS/odom for robot_1 home estimation; "
              "using surveyed fallback.", flush=True)
        ff.set_home(*FIX_HOME_R1)
        return ff
    ot = np.array([o[0] for o in odom])
    op = np.array([o[1:] for o in odom])
    for t, lat, lon, alt in gps:
        i = np.searchsorted(ot, t)
        if i >= len(ot):
            i = len(ot) - 1
        if ff.update_home(lat, lon, alt, op[i]):
            break
    if not ff.is_ready():
        ff.set_home(*FIX_HOME_R1)
    print(f"robot_1 home: lat={ff._home_lat:.6f} lon={ff._home_lon:.6f} "
          f"fixed={ff.home_fixed}", flush=True)
    return ff


def latest_before(items, t):
    """items sorted by time; returns last element with time <= t, else None."""
    lo, hi = 0, len(items)
    while lo < hi:
        mid = (lo + hi) // 2
        if items[mid][0] <= t:
            lo = mid + 1
        else:
            hi = mid
    return items[lo - 1] if lo else None


def coverage_rgba(grid, color):
    rgba = np.zeros(grid.shape + (4,), dtype=np.float32)
    c = matplotlib.colors.to_rgb(color)
    observed = grid == 0
    obstacle = grid == 100
    rgba[observed] = (*c, 0.22)
    rgba[obstacle] = (c[0] * 0.45, c[1] * 0.45, c[2] * 0.45, 0.75)
    return rgba


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", default="/bags/merged_3dronetest")
    ap.add_argument("--out", default="/bags/viz_3dronetest.mp4")
    ap.add_argument("--dt", type=float, default=1.0,
                    help="simulated seconds per animation frame")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--start", type=float, default=0.0,
                    help="offset into the bag, seconds")
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--trail", type=float, default=60.0,
                    help="trail length in seconds (0 = full history)")
    ap.add_argument("--stale", type=float, default=6.0,
                    help="hide claims/tracks not refreshed within this many s")
    args = ap.parse_args()

    print(f"Reading {args.bag} ...", flush=True)
    data, gps = read_bag(args.bag)

    # --- transform poses into the frontier frame -------------------------
    ff1 = estimate_r1_home(gps, data[1]["odom"])
    R1, t1 = ff1._affine_R, ff1._affine_t
    frames = {1: (R1, t1)}
    for rid in (2, 3):
        frames[rid] = ned_to_frame_transform(rid)

    traj = {}
    for rid in ROBOTS:
        od = data[rid]["odom"]
        if not od:
            traj[rid] = (np.zeros(0), np.zeros((0, 2)))
            continue
        ts = np.array([o[0] for o in od])
        pos = np.array([o[1:] for o in od])
        R, tvec = frames[rid]
        pf = (pos @ R.T + tvec)[:, :2]
        # Truncate diverged odometry (robot_2's laser odom blows up to
        # km-scale near the end of its bag): find a large jump after which
        # the trajectory *stays* far away, i.e. divergence, not a glitch.
        step = np.linalg.norm(np.diff(pf, axis=0), axis=1)
        for j in np.where(step > 3.0)[0]:
            anchor = np.median(pf[max(0, j - 200):j + 1], axis=0)
            tail = pf[j + 1:j + 201]
            if len(tail) and np.median(
                    np.linalg.norm(tail - anchor, axis=1)) > 50.0:
                print(f"robot_{rid}: odom diverges at t={ts[j] - ts[0]:.1f}s "
                      f"into bag; dropping {len(ts) - j - 1} samples",
                      flush=True)
                ts, pf = ts[:j + 1], pf[:j + 1]
                break
        traj[rid] = (ts, pf)
        print(f"robot_{rid}: {len(ts)} poses, frame x "
              f"[{pf[:, 0].min():.1f}, {pf[:, 0].max():.1f}] y "
              f"[{pf[:, 1].min():.1f}, {pf[:, 1].max():.1f}]", flush=True)

    t0 = min(ts[0] for ts, _ in traj.values() if len(ts))
    t1_ = max(ts[-1] for ts, _ in traj.values() if len(ts))
    t_begin = t0 + args.start
    t_end = min(t1_, t_begin + args.duration) if args.duration else t1_
    times = np.arange(t_begin, t_end, args.dt)
    print(f"{len(times)} frames over {t_end - t_begin:.0f}s", flush=True)

    # --- figure -----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 9), dpi=110)
    ax.set_facecolor("#f4f4f4")
    ax.set_xlabel("frontier frame x [m]")
    ax.set_ylabel("frontier frame y [m]")
    ax.set_aspect("equal")

    all_xy = np.vstack([xy for _, xy in traj.values() if len(xy)])
    pad = 12.0
    ax.set_xlim(all_xy[:, 0].min() - pad, all_xy[:, 0].max() + pad)
    ax.set_ylim(all_xy[:, 1].min() - pad, all_xy[:, 1].max() + pad)

    art = {}
    for rid, meta in ROBOTS.items():
        c = meta["color"]
        art[rid] = dict(
            cov=ax.imshow(np.zeros((1, 1, 4)), origin="lower",
                          extent=(0, 1, 0, 1), interpolation="nearest",
                          zorder=1),
            trail=ax.plot([], [], "-", color=c, lw=1.2, alpha=0.65,
                          zorder=3)[0],
            pose=ax.plot([], [], "o", color=c, ms=11, mec="black", mew=1.2,
                         zorder=6)[0],
            tracks=ax.plot([], [], "X", color=c, ms=12, mec="darkred",
                           mew=1.4, ls="", zorder=5)[0],
            region=ax.plot([], [], "-", color=c, lw=2.2, zorder=4)[0],
            star=ax.plot([], [], "*", color="gold", ms=26, mec=c, mew=2.0,
                         ls="", zorder=7)[0],
            link=ax.plot([], [], "--", color=c, lw=1.6, alpha=0.9,
                         zorder=4)[0],
            label=ax.annotate("", (0, 0), fontsize=8, color="black",
                              xytext=(8, 8), textcoords="offset points",
                              zorder=8),
        )
    track_texts = []
    title = ax.set_title("")
    status = ax.text(0.01, 0.99, "", transform=ax.transAxes, va="top",
                     fontsize=9, family="monospace",
                     bbox=dict(fc="white", alpha=0.8, ec="none"))

    handles = [Line2D([], [], marker="o", color=m["color"], ls="",
                      mec="black", label=m["name"]) for m in ROBOTS.values()]
    handles += [
        Line2D([], [], marker="X", color="gray", mec="darkred", ls="",
               label="person track (maipp/tracks)"),
        Line2D([], [], marker="*", color="gold", mec="gray", ms=14, ls="",
               label="claimed person-track task"),
        Line2D([], [], color="gray", lw=2.2, label="claimed region task"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)

    def update(fi):
        t = times[fi]
        lines = []
        for txt in track_texts:
            txt.remove()
        track_texts.clear()
        for rid, meta in ROBOTS.items():
            a = art[rid]
            ts, xy = traj[rid]
            # pose + trail
            if len(ts) and ts[0] <= t:
                i = np.searchsorted(ts, t)
                i = max(1, min(i, len(ts)))
                j0 = 0
                if args.trail > 0:
                    j0 = np.searchsorted(ts, t - args.trail)
                a["trail"].set_data(xy[j0:i, 0], xy[j0:i, 1])
                if t - ts[i - 1] <= args.stale:
                    px, py = xy[i - 1]
                    a["pose"].set_data([px], [py])
                    a["label"].set_text(f"R{rid}")
                    a["label"].xy = (px, py)
                    robot_here = (px, py)
                else:
                    # odometry ended/diverged before t
                    a["pose"].set_data([], [])
                    a["label"].set_text("")
                    robot_here = None
            else:
                a["trail"].set_data([], [])
                a["pose"].set_data([], [])
                a["label"].set_text("")
                robot_here = None
            # coverage
            cov = latest_before(data[rid]["coverage"], t)
            if cov is not None:
                _, ox, oy, res, grid = cov
                a["cov"].set_data(coverage_rgba(grid, meta["color"]))
                a["cov"].set_extent((ox, ox + grid.shape[1] * res,
                                     oy, oy + grid.shape[0] * res))
            # people tracks
            trk = latest_before(data[rid]["tracks"], t)
            if trk is not None and t - trk[0] <= args.stale and trk[1]:
                xs = [p[1] for p in trk[1]]
                ys = [p[2] for p in trk[1]]
                a["tracks"].set_data(xs, ys)
                for pid, x, y, score in trk[1]:
                    track_texts.append(ax.annotate(
                        f"{pid} p={score:.2f}", (x, y), fontsize=7,
                        color="darkred", xytext=(6, -10),
                        textcoords="offset points", zorder=8))
            else:
                a["tracks"].set_data([], [])
            # task claim
            claim = latest_before(data[rid]["claim"], t)
            task = "idle"
            a["region"].set_data([], [])
            a["star"].set_data([], [])
            a["link"].set_data([], [])
            if claim is not None and t - claim[0] <= args.stale:
                pts = claim[1]
                if len(pts) >= 3:
                    ring = pts + [pts[0]]
                    a["region"].set_data([p[0] for p in ring],
                                         [p[1] for p in ring])
                    task = "REGION"
                elif len(pts) >= 1:
                    sx, sy = pts[0]
                    a["star"].set_data([sx], [sy])
                    if robot_here is not None:
                        a["link"].set_data([robot_here[0], sx],
                                           [robot_here[1], sy])
                    task = f"TRACK PERSON @({sx:.1f},{sy:.1f})"
            if robot_here is None and len(ts) and ts[0] <= t:
                task += "  [odom lost/ended]"
            lines.append(f"R{rid}: {task}")
        title.set_text(
            f"3-drone MAIPP mission - t = {t - t0:7.1f}s / "
            f"{t_end - t0:.0f}s (frontier frame)")
        status.set_text("\n".join(lines))

    print("Rendering animation ...", flush=True)
    anim = animation.FuncAnimation(fig, update, frames=len(times),
                                   interval=1000 / args.fps)
    if args.out.endswith(".gif"):
        anim.save(args.out, writer=animation.PillowWriter(fps=args.fps))
    else:
        anim.save(args.out, writer=animation.FFMpegWriter(
            fps=args.fps, bitrate=4000))
    plt.close(fig)
    print(f"Saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
