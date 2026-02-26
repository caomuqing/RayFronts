from rayfronts.utils import compute_cos_sim
from nav_msgs.msg import Path
import torch
from rayfronts import geometry3d as g3d
import numpy as np
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

class RayBehavior:
    def __init__(self, get_clock):
        self.get_clock = get_clock
        self.name = 'Ray-based'
        self.prev_filtered_marker_ids = 0

    def condition_check(self, queries_labels, target_objects, queries_feats, mapper, publisher_dict, subscriber_dict):
        if queries_labels is None:
            return False

        if queries_labels['text'] is None:
            return False

        if len(target_objects) == 0:
            return False

        if queries_labels is not None and queries_labels['text'] is not None and len(target_objects) > 0:
            label_indices = [queries_labels['text'].index(target_object) for target_object in target_objects]

            ray_feat = mapper.global_rays_feat
            ray_orig_angles = mapper.global_rays_orig_angles

            if ray_feat is not None and ray_orig_angles is not None and ray_feat.shape[0] > 0:
                ray_lang_aligned = mapper.encoder.align_spatial_features_with_language(ray_feat.unsqueeze(-1).unsqueeze(-1))

                if ray_lang_aligned.ndim == 4:
                    ray_lang_aligned = ray_lang_aligned.squeeze(-1).squeeze(-1)
                if ray_lang_aligned.ndim == 2:
                    ray_lang_aligned = ray_lang_aligned
                if ray_lang_aligned.ndim == 1:
                    ray_lang_aligned = ray_lang_aligned.unsqueeze(0)

                if queries_feats is not None:
                    ray_scores = compute_cos_sim(queries_feats['text'], ray_lang_aligned, softmax=True)
                    threshold=0.6

                    relevant_scores = ray_scores[:, label_indices]
                    mask = (relevant_scores > threshold).any(dim=1)

                    indices = mask.nonzero(as_tuple=True)[0]
                    if indices.numel() > 0:
                        self.indices = indices
                        self.ray_orig_angles = ray_orig_angles
                        return True

        return False

    def execute(self, mapper, point3d_dict, waypoint_locked, publisher_dict, subscriber_dict):
        path_publisher = publisher_dict['path']
        cur_pose_np = point3d_dict['cur_pose']
        target_waypoint1 = point3d_dict['target1']
        target_waypoint2 = point3d_dict['target2']
        ray_orig = self.ray_orig_angles[:,:3]
        ray_angles = torch.deg2rad(self.ray_orig_angles[:,3:])
        ray_dir = torch.stack(g3d.spherical_to_cartesian(1,ray_angles[:,0],ray_angles[:,1]),dim=-1)

        fo = ray_orig[self.indices]
        fd = ray_dir[self.indices]
        orig_world = torch.stack([fo[:,2], -fo[:,0], -fo[:,1]],dim=1)
        dir_world = torch.stack([fd[:,2], -fd[:,0], -fd[:,1]],dim=1)
        xy_dirs = dir_world[:,:2]

        xy_dirs_np = xy_dirs.cpu().numpy()
        xy_dirs_np_normed = xy_dirs_np / np.linalg.norm(xy_dirs_np, axis=1, keepdims=True)

        #filter rays that are behind the robot XY
        cur_xy = cur_pose_np[:2]
        orig_xy = orig_world[:,:2]
        dir_xy = xy_dirs_np_normed

        ray_target_xy = orig_xy.cpu().numpy() + dir_xy
        to_ray_target = ray_target_xy - cur_xy

        dot = np.einsum('ij,ij->i',dir_xy,to_ray_target)
        valid_mask = dot > 0

        xy_dirs_np_normed = xy_dirs_np_normed[valid_mask]

        angle_groups = []

        angle_threshold_cos = np.cos(np.deg2rad(45))

        for i, xy_dir in enumerate(xy_dirs_np_normed):
            assigned=False
            for group in angle_groups:
                dot = np.dot(xy_dir, group['centroid'])
                if dot >= angle_threshold_cos:
                    group['indices'].append(i)
                    group['rays'].append(xy_dir)
                    group['centroid'] = np.mean(group['rays'], axis=0)
                    group['centroid'] /= np.linalg.norm(group['centroid'])
                    assigned=True
                    break
            if not assigned:
                angle_groups.append({'centroid':xy_dir, 'rays':[xy_dir], 'indices':[i]})

        MIN_RAYS_PER_GROUP=1
        angle_groups = [g for g in angle_groups if len(g['rays']) >= MIN_RAYS_PER_GROUP]

        group_averages = []
        for group in angle_groups:
            group_idx = group['indices']
            group_origins = orig_world[group_idx]
            group_directions = dir_world[group_idx]

            avg_origin = group_origins.mean(dim=0)
            avg_direction = group_directions.mean(dim=0)
            avg_direction = avg_direction / avg_direction.norm()

            density = len(group['rays'])

            group_averages.append((avg_origin, avg_direction, density))

        k=5.0
        scored_groups = sorted(group_averages, key=lambda g: np.linalg.norm(g[0].cpu().numpy() - cur_pose_np) - k*g[2])
        print("scored_groups", scored_groups)

        if not scored_groups:
            print("No valid ray groups found (all below MIN_RAYS_PER_GROUP)")
            best_group = None
            return waypoint_locked, target_waypoint1, target_waypoint2
        else:
            best_group = scored_groups[0]

        magnitude = 2.0

        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = 'map'

        best_origin, best_direction = best_group[0], best_group[1]
        best_origin_np = best_origin.cpu().numpy()
        best_direction_np = best_direction.cpu().numpy()

        origin = best_origin_np
        direction = best_direction_np / np.linalg.norm(best_direction_np)
        alpha=0.8

        mid_pose_np = cur_pose_np * (1-alpha) + origin*alpha
        mid_pose = PoseStamped()
        mid_pose.header.stamp = self.get_clock().now().to_msg()
        mid_pose.header.frame_id = 'map'
        mid_pose.pose.position.x = float(mid_pose_np[0])
        mid_pose.pose.position.y = float(mid_pose_np[1])
        mid_pose.pose.position.z = float(mid_pose_np[2])
        mid_pose.pose.orientation.w = 1.0
        #path.poses.append(mid_pose)


        target_waypoint1 = origin + direction*magnitude
        target_waypoint2 = origin + direction*magnitude*2

        t1_pose = PoseStamped()
        t1_pose.header.stamp = self.get_clock().now().to_msg()
        t1_pose.header.frame_id = 'map'
        t1_pose.pose.position.x = float(target_waypoint1[0])
        t1_pose.pose.position.y = float(target_waypoint1[1])
        t1_pose.pose.position.z = float(target_waypoint1[2])
        t1_pose.pose.orientation.w = 1.0
        path.poses.append(t1_pose)

        t2_pose = PoseStamped()
        t2_pose.header.stamp = self.get_clock().now().to_msg()
        t2_pose.header.frame_id = 'map'
        t2_pose.pose.position.x = float(target_waypoint2[0])
        t2_pose.pose.position.y = float(target_waypoint2[1])
        t2_pose.pose.position.z = float(target_waypoint2[2])
        t2_pose.pose.orientation.w = 1.0
        path.poses.append(t2_pose)

        path_publisher.publish(path)

        self.visualize_filtered_rays(angle_groups, dir_world, orig_world, publisher_dict)

        if np.linalg.norm(cur_pose_np - target_waypoint2) < 4.0:
            waypoint_locked = False

        return waypoint_locked, target_waypoint1, target_waypoint2

    def visualize_filtered_rays(self, angle_groups, dir_world, orig_world, publisher_dict):
        filtered_rays_publisher = publisher_dict['filtered_rays']
        self.clear_filtered_rays(filtered_rays_publisher)
        arrow_length = 2.0
        filtered_marker_array = MarkerArray()
        colors = [(1.0,0.0,0.0),(0.0,1.0,0.0),(0.0,0.0,1.0),(1.0,1.0,0.0),(0.0,1.0,1.0),(1.0,0.0,1.0),(0.5,0.5,0.5),(1.0,0.5,0.0),(0.5,0.0,1.0),(0.0,0.5,1.0)]

        j=0
        for i, group in enumerate(angle_groups):
            idxes = group['indices']
            rr,gg,bb = colors[i%len(colors)]
            for idx in idxes:
                dir0 = dir_world[idx].cpu().numpy()
                p0 = orig_world[idx].cpu().numpy()
                p1 = p0 + arrow_length*dir0
                arrow = Marker()
                arrow.header.frame_id = 'map'
                arrow.header.stamp = self.get_clock().now().to_msg()
                arrow.ns = 'arrows'
                arrow.id = j
                arrow.type = Marker.ARROW
                arrow.action = Marker.ADD
                arrow.points = [Point(x=float(p0[0]), y=float(p0[1]), z=float(p0[2])), Point(x=float(p1[0]), y=float(p1[1]), z=float(p1[2]))]
                arrow.scale.x = 0.6
                arrow.scale.y = 1.2
                arrow.scale.z = 0.75
                arrow.color.r = rr
                arrow.color.g = gg
                arrow.color.b = bb
                arrow.color.a = 0.5
                filtered_marker_array.markers.append(arrow)
                j += 1
        self.prev_filtered_marker_ids = j
        filtered_rays_publisher.publish(filtered_marker_array)

    def clear_filtered_rays(self, filtered_rays_publisher):
        if self.prev_filtered_marker_ids > 0:
            clear_marker_array = MarkerArray()
            for i in range(self.prev_filtered_marker_ids):
                clear_marker = Marker()
                clear_marker.header.frame_id = 'map'
                clear_marker.header.stamp = self.get_clock().now().to_msg()
                clear_marker.ns = 'arrows'
                clear_marker.id = i
                clear_marker.action = Marker.DELETE
                clear_marker_array.markers.append(clear_marker)
            filtered_rays_publisher.publish(clear_marker_array)
