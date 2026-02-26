import torch
from sklearn.cluster import DBSCAN
from std_msgs.msg import Header
from sensor_msgs.msg import PointField
from sensor_msgs_py import point_cloud2
from nav_msgs.msg import Path
import numpy as np
from geometry_msgs.msg import PoseStamped

class FrontierBehavior:
    def __init__(self, get_clock):
        self.get_clock = get_clock
        self.name = 'Frontier-based'

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

            #filter out frontier points that are under the height of 4.0m
            transformed_frontiers = transformed_frontiers[transformed_frontiers[:,2] > 4.0]

            #DBSCAN clustering for frontier-points
            frontiers_cpu = transformed_frontiers.detach().cpu().numpy()
            clustering = DBSCAN(eps=4.0, min_samples=5).fit(frontiers_cpu)
            labels = clustering.labels_
            unique_labels = [l for l in set(labels) if l != -1]
            viewpoints = []

            for l in unique_labels:
                cluster_pts = frontiers_cpu[labels==l]
                centroid = cluster_pts.mean(axis=0)
                centroid_torch = torch.from_numpy(centroid)
                centroid_torch = centroid_torch.to(transformed_frontiers.device, dtype = transformed_frontiers.dtype)

                if centroid_torch[2] > 4.0:
                    centroid_torch[2] = 6.0 #manually set height of frontier 6m
                    viewpoints.append(centroid_torch)
            if len(viewpoints) == 0:
                return waypoint_locked, target_waypoint, target_waypoint2
            viewpoints = torch.stack(viewpoints)

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
