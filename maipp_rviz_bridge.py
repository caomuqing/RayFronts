#!/usr/bin/env python3
"""Live RViz2 bridge for the merged 3dronetest bag (or a live mission).

Subscribes to the raw per-robot MAIPP topics and republishes RViz-friendly
visualization in the shared frontier frame:
  /viz/poses     MarkerArray  colored sphere + "R<i> [TASK]" label per robot
  /viz/robot_<i>/trail  Path  trajectory in the frontier frame
  /viz/coverage  MarkerArray  per-robot colored CUBE_LIST of coverage cells
                              (observed translucent, obstacles darker)
  /viz/tracks    MarkerArray  people tracks as spheres + "id p=..." labels
  /viz/tasks     MarkerArray  region claim -> rectangle outline,
                              person-track claim -> gold star sphere + line
                              from the robot (the 1-point polygon RViz can't
                              show natively)

Pose transforms replicate the reference code: robot_1's ENU odometry is
georeferenced online from mavros GPS (FrontierFrame.update_home); robot_2/3's
NED odometry uses the surveyed ned_origins from
RayFronts_small/experiments/preset_configs/starlingmax_decoupled_bag.yaml.
Diverged laser odometry (robot_2 near the end of its bag) is gated out.

Usage (inside the robot container, two terminals):
  terminal 1:  python3 maipp_rviz_bridge.py
  terminal 2:  ros2 bag play /bags/merged_3dronetest
  then:        rviz2 -d viz_3dronetest.rviz   (fixed frame: frontier_frame)
"""

import math
import os

import numpy as np

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "geo_frame",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "RayFronts_high", "rayfronts", "geo_frame.py"))
_geo = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_geo)
FrontierFrame = _geo.FrontierFrame

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.duration import Duration

from geometry_msgs.msg import Point, PolygonStamped, PoseStamped
from nav_msgs.msg import Odometry, OccupancyGrid, Path
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import ColorRGBA
from vision_msgs.msg import Detection3DArray
from visualization_msgs.msg import Marker, MarkerArray

FRAME = "frontier_frame"

# Frontier frame definition (mapping_server_rosnode.py:105-109).
FF_ORIGIN = dict(origin_lat=40.413131, origin_lon=-79.946393,
                 origin_alt=220.80565047594274, heading_deg=-76.17)
# Surveyed NED odometry origins for the starlings
# (starlingmax_decoupled_bag.yaml ned_origins).
NED_ORIGINS = {
    2: dict(lat=40.41329, lon=-79.94657, alt=0.0, heading_deg=166.17),
    3: dict(lat=40.41332, lon=-79.94648, alt=0.0, heading_deg=166.17),
}

COLORS = {1: (0.12, 0.47, 0.71), 2: (1.0, 0.5, 0.05), 3: (0.17, 0.63, 0.17)}
NAMES = {1: "R1 (high)", 2: "R2 (starling)", 3: "R3 (starling)"}


def rgba(c, a):
    return ColorRGBA(r=float(c[0]), g=float(c[1]), b=float(c[2]), a=float(a))


def ned_to_frame_transform(rid):
    o = NED_ORIGINS[rid]
    ff = FrontierFrame(**FF_ORIGIN)
    ff.set_home(o["lat"], o["lon"], o["alt"])
    psi = math.radians(o["heading_deg"])
    # NED -> ENU at home (exploration_planner.py:459-462).
    r_n2e = np.array([[math.sin(psi), math.cos(psi), 0.0],
                      [math.cos(psi), -math.sin(psi), 0.0],
                      [0.0, 0.0, -1.0]])
    return ff._affine_R @ r_n2e, ff._affine_t.copy()


class MaippRvizBridge(Node):

    def __init__(self):
        super().__init__("maipp_rviz_bridge")
        qos = QoSProfile(depth=10,
                         reliability=ReliabilityPolicy.BEST_EFFORT)

        # robot_1 georeference, estimated online from GPS + odometry.
        self.ff1 = FrontierFrame(**FF_ORIGIN)
        self._last_gps = None
        self._home_announced = False
        self.tf = {rid: ned_to_frame_transform(rid) for rid in (2, 3)}

        self.pos = {}          # rid -> latest accepted frame position (3,)
        self.task = {rid: "idle" for rid in (1, 2, 3)}
        self.paths = {}
        self.pub_poses = self.create_publisher(MarkerArray, "/viz/poses", 5)
        self.pub_cov = self.create_publisher(MarkerArray, "/viz/coverage", 5)
        self.pub_trk = self.create_publisher(MarkerArray, "/viz/tracks", 5)
        self.pub_task = self.create_publisher(MarkerArray, "/viz/tasks", 5)
        self.pub_path = {}
        for rid in (1, 2, 3):
            p = Path()
            p.header.frame_id = FRAME
            self.paths[rid] = p
            self.pub_path[rid] = self.create_publisher(
                Path, f"/viz/robot_{rid}/trail", 5)

        self.create_subscription(
            Odometry, "/robot_1/odometry_conversion/odometry",
            lambda m: self.on_odom(1, m), qos)
        self.create_subscription(
            NavSatFix, "/robot_1/interface/mavros/global_position/global",
            self.on_gps, qos)
        for rid in (2, 3):
            self.create_subscription(
                Odometry, f"/robot_{rid}/odom_laser_ned_relay",
                lambda m, r=rid: self.on_odom(r, m), qos)
        for rid in (1, 2, 3):
            self.create_subscription(
                OccupancyGrid, f"/robot_{rid}/maipp/coverage_grid",
                lambda m, r=rid: self.on_coverage(r, m), qos)
            self.create_subscription(
                Detection3DArray, f"/robot_{rid}/maipp/tracks",
                lambda m, r=rid: self.on_tracks(r, m), qos)
            self.create_subscription(
                PolygonStamped, f"/robot_{rid}/maipp/task_claim",
                lambda m, r=rid: self.on_claim(r, m), qos)
        self.get_logger().info(
            "maipp_rviz_bridge up; play the bag and open RViz "
            f"(fixed frame: {FRAME}).")

    # ---------------- transforms ----------------

    def on_gps(self, msg):
        if not math.isnan(msg.latitude):
            self._last_gps = msg

    def frame_pos(self, rid, p):
        """Odometry position -> frontier frame, or None if not ready."""
        v = np.array([p.x, p.y, p.z])
        if rid == 1:
            if not self.ff1.home_fixed:
                if self._last_gps is None:
                    return None
                g = self._last_gps
                self.ff1.update_home(g.latitude, g.longitude, g.altitude, v)
                if self.ff1.home_fixed and not self._home_announced:
                    self._home_announced = True
                    self.get_logger().info(
                        "robot_1 frontier-frame home locked (lat=%.6f "
                        "lon=%.6f)." % (self.ff1._home_lat,
                                        self.ff1._home_lon))
                if not self.ff1.is_ready():
                    return None
            return self.ff1._affine_R @ v + self.ff1._affine_t
        R, t = self.tf[rid]
        return R @ v + t

    # ---------------- odometry -> pose marker + trail ----------------

    def on_odom(self, rid, msg):
        pf = self.frame_pos(rid, msg.pose.pose.position)
        if pf is None:
            return
        # Gate diverged laser odometry: an in-bounds sanity box plus a jump
        # limit vs the last accepted sample (10-20 Hz odom can't move 5 m).
        if abs(pf[0]) > 100.0 or abs(pf[1]) > 100.0:
            return
        if rid in self.pos and np.linalg.norm(pf[:2] - self.pos[rid][:2]) > 5.0:
            return
        self.pos[rid] = pf
        now = self.get_clock().now().to_msg()
        c = COLORS[rid]

        arr = MarkerArray()
        m = Marker()
        m.header.frame_id = FRAME
        m.header.stamp = now
        m.ns = f"robot_{rid}"
        m.id = 0
        m.type = Marker.SPHERE
        m.pose.position.x, m.pose.position.y, m.pose.position.z = map(
            float, pf)
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 1.0
        m.color = rgba(c, 1.0)
        m.lifetime = Duration(seconds=3.0).to_msg()
        arr.markers.append(m)

        t = Marker()
        t.header.frame_id = FRAME
        t.header.stamp = now
        t.ns = f"robot_{rid}"
        t.id = 1
        t.type = Marker.TEXT_VIEW_FACING
        t.pose.position.x = float(pf[0])
        t.pose.position.y = float(pf[1])
        t.pose.position.z = float(pf[2]) + 1.6
        t.scale.z = 1.0
        t.color = rgba(c, 1.0)
        t.text = f"{NAMES[rid]} [{self.task[rid]}]"
        t.lifetime = Duration(seconds=3.0).to_msg()
        arr.markers.append(t)
        self.pub_poses.publish(arr)

        path = self.paths[rid]
        if path.poses:
            q = path.poses[-1].pose.position
            if np.linalg.norm(pf - np.array([q.x, q.y, q.z])) < 0.3:
                return
        ps = PoseStamped()
        ps.header.frame_id = FRAME
        ps.header.stamp = now
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = map(
            float, pf)
        ps.pose.orientation.w = 1.0
        path.poses.append(ps)
        if len(path.poses) > 5000:
            path.poses = path.poses[-5000:]
        path.header.stamp = now
        self.pub_path[rid].publish(path)

    # ---------------- coverage grid -> colored cube list ----------------

    def on_coverage(self, rid, msg):
        c = COLORS[rid]
        grid = np.array(msg.data, dtype=np.int8).reshape(
            msg.info.height, msg.info.width)
        res = float(msg.info.resolution)
        ox = msg.info.origin.position.x
        oy = msg.info.origin.position.y

        m = Marker()
        m.header.frame_id = FRAME
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = f"coverage_{rid}"
        m.id = 0
        m.type = Marker.CUBE_LIST
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = res * 0.95
        m.scale.z = 0.05
        # Slight per-robot z offset so overlapping cells don't z-fight.
        z = -0.10 + 0.03 * rid
        js, is_ = np.nonzero(grid >= 0)
        for j, i in zip(js, is_):
            m.points.append(Point(x=ox + (i + 0.5) * res,
                                  y=oy + (j + 0.5) * res, z=z))
            m.colors.append(rgba(c, 0.85) if grid[j, i] == 100
                            else rgba(c, 0.25))
        arr = MarkerArray()
        arr.markers.append(m)
        self.pub_cov.publish(arr)

    # ---------------- people tracks ----------------

    def on_tracks(self, rid, msg):
        c = COLORS[rid]
        now = self.get_clock().now().to_msg()
        arr = MarkerArray()
        for k, det in enumerate(msg.detections):
            if not det.results:
                continue
            hyp = det.results[0]
            x = hyp.pose.pose.position.x
            y = hyp.pose.pose.position.y
            m = Marker()
            m.header.frame_id = FRAME
            m.header.stamp = now
            m.ns = f"tracks_{rid}"
            m.id = 2 * k
            m.type = Marker.CYLINDER
            m.pose.position.x, m.pose.position.y = float(x), float(y)
            m.pose.position.z = 0.9
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = 0.6
            m.scale.z = 1.8
            m.color = rgba(c, 0.9)
            m.lifetime = Duration(seconds=6.0).to_msg()
            arr.markers.append(m)

            t = Marker()
            t.header.frame_id = FRAME
            t.header.stamp = now
            t.ns = f"tracks_{rid}"
            t.id = 2 * k + 1
            t.type = Marker.TEXT_VIEW_FACING
            t.pose.position.x, t.pose.position.y = float(x), float(y)
            t.pose.position.z = 2.4
            t.scale.z = 0.6
            t.color = rgba(c, 1.0)
            t.text = "%s p=%.2f" % (det.id, hyp.hypothesis.score)
            t.lifetime = Duration(seconds=6.0).to_msg()
            arr.markers.append(t)
        if arr.markers:
            self.pub_trk.publish(arr)

    # ---------------- task claims ----------------

    def on_claim(self, rid, msg):
        c = COLORS[rid]
        now = self.get_clock().now().to_msg()
        pts = [(p.x, p.y) for p in msg.polygon.points]
        life = Duration(seconds=6.0).to_msg()

        def mk(mid, mtype):
            m = Marker()
            m.header.frame_id = FRAME
            m.header.stamp = now
            m.ns = f"task_{rid}"
            m.id = mid
            m.type = mtype
            m.pose.orientation.w = 1.0
            m.lifetime = life
            return m

        rect = mk(0, Marker.LINE_STRIP)
        star = mk(1, Marker.SPHERE)
        link = mk(2, Marker.LINE_STRIP)
        label = mk(3, Marker.TEXT_VIEW_FACING)

        if len(pts) >= 3:
            self.task[rid] = "REGION"
            rect.scale.x = 0.25
            rect.color = rgba(c, 1.0)
            for x, y in pts + [pts[0]]:
                rect.points.append(Point(x=float(x), y=float(y), z=0.3))
        elif len(pts) >= 1:
            x, y = pts[0]
            self.task[rid] = "TRACK PERSON"
            star.pose.position.x, star.pose.position.y = float(x), float(y)
            star.pose.position.z = 1.0
            star.scale.x = star.scale.y = star.scale.z = 1.6
            star.color = rgba((1.0, 0.84, 0.0), 1.0)  # gold
            label.pose.position.x, label.pose.position.y = float(x), float(y)
            label.pose.position.z = 3.2
            label.scale.z = 0.8
            label.color = rgba((1.0, 0.84, 0.0), 1.0)
            label.text = f"TRACKED by R{rid}"
            if rid in self.pos:
                p = self.pos[rid]
                link.scale.x = 0.15
                link.color = rgba(c, 0.9)
                link.points.append(Point(x=float(p[0]), y=float(p[1]),
                                         z=float(p[2])))
                link.points.append(Point(x=float(x), y=float(y), z=1.0))
        else:
            self.task[rid] = "idle"

        arr = MarkerArray()
        for m in (rect, star, link, label):
            if not m.points and m.type == Marker.LINE_STRIP:
                m.action = Marker.DELETE
            if m.type == Marker.SPHERE and m.scale.x == 0.0:
                m.action = Marker.DELETE
            if m.type == Marker.TEXT_VIEW_FACING and not m.text:
                m.action = Marker.DELETE
            arr.markers.append(m)
        self.pub_task.publish(arr)


def main():
    rclpy.init()
    node = MaippRvizBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()


if __name__ == "__main__":
    main()
