import logging
import time

import torch
from sklearn.cluster import DBSCAN
from std_msgs.msg import Header
from sensor_msgs.msg import PointField
from sensor_msgs_py import point_cloud2
from nav_msgs.msg import Path
import numpy as np
from geometry_msgs.msg import PoseStamped

from rayfronts.geo_frame import points_in_polygon, segments_cross_polygon

logger = logging.getLogger(__name__)

class FrontierBehavior:
    def __init__(self, get_clock, geo_frame=None,
                 map_min_x=None, map_max_x=None, map_min_y=None, map_max_y=None,
                 keepout_polygons=None):
        self.get_clock = get_clock
        self.name = 'Frontier-based'
        # Optional MAIPP task layer (rayfronts.task_planner). When set, the
        # committed region/track task filters or replaces the candidate
        # viewpoints before the distance/momentum selection below; when
        # None the original global selection is unchanged.
        self.task_planner = None
        # Keepout zones: list of (M, 2) arrays of polygon corners in the
        # frontier frame. Frontier points inside any polygon are discarded,
        # and no viewpoint (cluster centroid) inside one may become a waypoint.
        # Zones may extend beyond the map_min/max bounds; the tests are
        # independent.
        self.keepout_polygons = keepout_polygons if keepout_polygons else []
        # Frontier-mapping frame + planar bounds (in that frame, meters).
        # Frontier candidates are transformed from the local/odom frame into
        # geo_frame and only those inside [map_min_x, map_max_x] x
        # [map_min_y, map_max_y] are eligible for the global plan. None on any
        # side leaves that side unbounded. If geo_frame is None or its home has
        # not yet been estimated, no plan is published (we cannot tell which
        # frontiers fall inside the mission boundary yet).
        self.geo_frame = geo_frame
        self.map_min_x = map_min_x
        self.map_max_x = map_max_x
        self.map_min_y = map_min_y
        self.map_max_y = map_max_y
        # Waypoint-lock safety: release the lock when the committed task
        # changes (a new task must redirect the drone immediately) or when
        # the lock is held longer than lock_timeout_s (goal unreachable).
        self._last_task_seq = None
        self._lock_start_t = None
        # 45s: longer than a legitimate transit leg (~70m box at 1-2 m/s)
        # so it does not fire mid-flight, and aligned with the region stall
        # timeout. Task changes release the lock independently of this.
        self.lock_timeout_s = 45.0
        # Throttled diagnostics for cycles that publish no plan.
        self._hold_log_t = 0.0

    def _log_hold(self, reason):
        """Logs (throttled) why this cycle published no plan."""
        now = time.time()
        if now - self._hold_log_t >= 5.0:
            self._hold_log_t = now
            logger.info("No global plan published: %s", reason)

    def condition_check(self, queries_labels, target_object, queries_feats, mapper, publisher_dict, subscriber_dict):
        return True

    def execute(self, mapper, point3d_dict, waypoint_locked, publisher_dict, subscriber_dict):
        cur_pose_np = point3d_dict['cur_pose']
        target_waypoint = point3d_dict['target1']
        target_waypoint2 = point3d_dict['target2']
        viewpoint_publisher = publisher_dict['viewpoint']
        path_publisher = publisher_dict['path']

        if mapper.frontiers is not None:
            transformed_frontiers = torch.stack([mapper.frontiers[:,2],-mapper.frontiers[:,0], -mapper.frontiers[:,1]], dim=1)

            #keep frontier points within the height band (4.0m, 11.0m]. The
            #lower bound drops ground-level frontiers; the upper bound drops
            #the persistent frontier "ceiling" sheet at the top of the sensed
            #volume, which otherwise skews cluster centroids inward.
            transformed_frontiers = transformed_frontiers[
                (transformed_frontiers[:,2] > 4.0) &
                (transformed_frontiers[:,2] <= 11.0)]

            frontiers_cpu = transformed_frontiers.detach().cpu().numpy()
            if frontiers_cpu.shape[0] == 0:
                self._log_hold("no frontiers in the 4-11m height band")
                return waypoint_locked, target_waypoint, target_waypoint2

            #limit frontiers to the frontier-mapping frame boundary.
            #transformed_frontiers are in the local/odom frame (== global_plan
            #frame); map them into the mission frame and keep only those inside
            #the [map_min_x, map_max_x] x [map_min_y, map_max_y] box.
            if self.geo_frame is None or not self.geo_frame.is_ready():
                #home/frame transform not established yet -> cannot tell which
                #frontiers fall inside the mission boundary, so publish nothing.
                self._log_hold(
                    "frontier frame home not estimated yet (waiting for GPS)")
                return waypoint_locked, target_waypoint, target_waypoint2
            frame_xy = self.geo_frame.local_to_frame(frontiers_cpu)[:, :2]
            mask = np.ones(frame_xy.shape[0], dtype=bool)
            if self.map_min_x is not None:
                mask &= frame_xy[:,0] >= self.map_min_x
            if self.map_max_x is not None:
                mask &= frame_xy[:,0] <= self.map_max_x
            if self.map_min_y is not None:
                mask &= frame_xy[:,1] >= self.map_min_y
            if self.map_max_y is not None:
                mask &= frame_xy[:,1] <= self.map_max_y

            #discard frontier points inside any keepout zone
            for poly in self.keepout_polygons:
                mask &= ~points_in_polygon(frame_xy, poly)

            frontiers_cpu = frontiers_cpu[mask]
            if frontiers_cpu.shape[0] == 0:
                self._log_hold(
                    "no frontiers inside mission bounds / outside keepouts")
                return waypoint_locked, target_waypoint, target_waypoint2

            #DBSCAN clustering for frontier-points. min_samples=4 (not 5):
            #the height band keeps only 2-3 layers of the ~2.5m subsampled
            #frontier grid, and with eps=3.0 a point in a 2-layer wall strip
            #reaches at most 4 neighbors (self + left/right + vertical;
            #diagonals at 3.54m are out of range), so min_samples=5 would mark
            #thin lateral walls as noise.
            clustering = DBSCAN(eps=3.0, min_samples=4).fit(frontiers_cpu)
            labels = clustering.labels_
            unique_labels = [l for l in set(labels) if l != -1]
            viewpoints = []

            for l in unique_labels:
                cluster_pts = frontiers_cpu[labels==l]
                centroid = cluster_pts.mean(axis=0)

                #a cluster of allowed points can still average to a centroid
                #inside a keepout zone -> never send the robot there
                if self.keepout_polygons:
                    centroid_frame_xy = self.geo_frame.local_to_frame(centroid)[None, :2]
                    if any(points_in_polygon(centroid_frame_xy, poly)[0]
                           for poly in self.keepout_polygons):
                        continue

                centroid_torch = torch.from_numpy(centroid)
                centroid_torch = centroid_torch.to(transformed_frontiers.device, dtype = transformed_frontiers.dtype)

                if centroid_torch[2] > 4.0:
                    centroid_torch[2] = 8.0 #manually set height of frontier 6m
                    viewpoints.append(centroid_torch)
            if len(viewpoints) > 0:
                viewpoints = torch.stack(viewpoints)
            else:
                # Keep going with an empty candidate set: a committed track
                # task can still produce a pseudo-viewpoint even when the
                # bounded area has no frontier clusters left.
                viewpoints = torch.zeros(
                    (0, 3), dtype=transformed_frontiers.dtype)

            # MAIPP task layer: restrict candidates to the committed
            # region's viewpoints, an approach viewpoint toward its
            # centroid, or a track-revisit pseudo-viewpoint.
            if self.task_planner is not None:
                viewpoints = self.task_planner.select(
                    viewpoints.cpu(), cur_pose_np, mapper=mapper).to(
                        dtype=transformed_frontiers.dtype)
                # A new committed task must redirect the drone immediately:
                # release the waypoint lock so this cycle re-selects.
                seq = self.task_planner.task_seq
                if seq != self._last_task_seq:
                    if self._last_task_seq is not None and waypoint_locked:
                        waypoint_locked = False
                        logger.info(
                            "Waypoint lock released: committed task changed.")
                    self._last_task_seq = seq

            # Stale-lock timeout: a goal that was never reached (blocked
            # path, controller refusal) must not pin the drone forever.
            if (waypoint_locked and self._lock_start_t is not None and
                    time.time() - self._lock_start_t > self.lock_timeout_s):
                waypoint_locked = False
                logger.info(
                    "Waypoint lock released: goal not reached within %.0fs.",
                    self.lock_timeout_s)

            if viewpoints.shape[0] == 0:
                self._log_hold(
                    "no goal candidates left after task-layer selection")
                return waypoint_locked, target_waypoint, target_waypoint2

            #non-crossing test: the straight line from the robot to a goal
            #must not pass through a keepout zone. Strict: blocked candidates
            #are always dropped (the goal itself may be legal while the
            #direct path is not), and an all-blocked set publishes nothing.
            #The hold is bounded: the task layer's stall/time budget retires
            #the committed task within <=60s and rotates to a reachable one.
            if self.keepout_polygons and viewpoints.shape[0] > 0:
                vp_np = viewpoints.detach().cpu().numpy()
                vp_frame_xy = self.geo_frame.local_to_frame(vp_np)[:, :2]
                robot_frame_xy = self.geo_frame.local_to_frame(
                    np.asarray(cur_pose_np, dtype=np.float64))[:2]
                blocked = np.zeros(vp_frame_xy.shape[0], dtype=bool)
                for poly in self.keepout_polygons:
                    blocked |= segments_cross_polygon(
                        robot_frame_xy, vp_frame_xy, poly)
                if blocked.all():
                    self._log_hold(
                        "all %d candidate paths cross a keepout zone"
                        % viewpoints.shape[0])
                    return waypoint_locked, target_waypoint, target_waypoint2
                viewpoints = viewpoints[torch.from_numpy(~blocked)]

            cent_msg = self.create_pointcloud2_msg(viewpoints)
            viewpoint_publisher.publish(cent_msg)

            robot_pos_torch = torch.tensor(cur_pose_np, dtype=viewpoints.dtype, device=viewpoints.device)
            distances = torch.norm(viewpoints - robot_pos_torch, dim=1)

            if target_waypoint is not None:
                target_waypoint_tensor = torch.tensor(target_waypoint, device=viewpoints.device, dtype=viewpoints.dtype)
                cur_motion_vec = target_waypoint_tensor - robot_pos_torch
                cur_motion_vec = cur_motion_vec / (torch.norm(cur_motion_vec) + 1e-6)
                candidate_vecs = viewpoints - robot_pos_torch
                candidate_vecs = candidate_vecs / (torch.norm(candidate_vecs, dim=1, keepdim=True) + 1e-6)
                cos_sim = torch.matmul(candidate_vecs, cur_motion_vec)
                momentum_weight = 2.0
                scores = distances + momentum_weight*(1.0-cos_sim)
            else:
                scores = distances

            top_n = 5
            num_candidates = min(top_n, viewpoints.shape[0])
            top_indices = torch.argsort(scores)[:num_candidates]
            best_idx = top_indices[torch.randint(0, num_candidates, (1,))]
            best_cent = viewpoints[best_idx].view(-1)

            path = Path()
            path.header.stamp = self.get_clock().now().to_msg()
            path.header.frame_id = 'map'

            if not waypoint_locked:
                best_cent_np = best_cent.cpu().numpy()
                target_waypoint = best_cent_np
                self._lock_start_t = time.time()
                direction = target_waypoint - cur_pose_np
                direction = direction / np.linalg.norm(target_waypoint - cur_pose_np)
                target_waypoint2 = target_waypoint + 1.0*direction
                waypoint_locked = True

            target_pose = PoseStamped()
            target_pose.header.stamp = self.get_clock().now().to_msg()
            target_pose.header.frame_id = 'map'
            target_pose.pose.position.x = float(target_waypoint[0])
            target_pose.pose.position.y = float(target_waypoint[1])
            target_pose.pose.position.z = float(target_waypoint[2])
            target_pose.pose.orientation.w = 1.0
            path.poses.append(target_pose)
            
            target_pose2 = PoseStamped()
            target_pose2.header.stamp = self.get_clock().now().to_msg()
            target_pose2.header.frame_id = 'map'
            target_pose2.pose.position.x = float(target_waypoint2[0])
            target_pose2.pose.position.y = float(target_waypoint2[1])
            target_pose2.pose.position.z = float(target_waypoint2[2])
            target_pose2.pose.orientation.w = 1.0
            path.poses.append(target_pose2)
            
            path_publisher.publish(path)
            if np.linalg.norm(cur_pose_np - target_waypoint) < 5.0:
            	waypoint_locked = False
            	
        return waypoint_locked, target_waypoint, target_waypoint2
    
    def create_pointcloud2_msg(self, xyz):
        if isinstance(xyz, torch.Tensor):
            xyz = xyz.detach().cpu().numpy()
        elif isinstance(xyz, np.ndarray):
            xyz = xyz
        else:
            raise TypeError(f"Expected torch.Tensor or numpy.ndarray, got {type(xyz)}")
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'map'
        fields = [PointField(name='x',offset=0,datatype=PointField.FLOAT32, count=1), PointField(name='y',offset=4,datatype=PointField.FLOAT32, count=1), PointField(name='z',offset=8,datatype=PointField.FLOAT32, count=1)]
        points = []
        for i in range(xyz.shape[0]):
            x,y,z = xyz[i]
            points.append([x,y,z])
        return point_cloud2.create_cloud(header, fields, points)
