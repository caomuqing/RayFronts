import torch
import time
import numpy as np

from rayfronts.behaviors.frontier_behavior import FrontierBehavior
#from rayfronts.behaviors.voxel_behavior import VoxelBehavior
from rayfronts.behaviors.ray_behavior import RayBehavior
from rayfronts.utils import compute_cos_sim

from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
import scipy.ndimage

class BehaviorManager:
    def __init__(self, get_clock, geo_frame=None,
                 map_min_x=None, map_max_x=None,
                 map_min_y=None, map_max_y=None,
                 keepout_polygons=None):
        self.behavior_mode = 'Frontier-based'
        self.get_clock = get_clock
        #self.voxel_behavior = VoxelBehavior(self.get_clock)
        self.ray_behavior = RayBehavior(self.get_clock)
        self.frontier_behavior = FrontierBehavior(
            self.get_clock, geo_frame=geo_frame,
            map_min_x=map_min_x, map_max_x=map_max_x,
            map_min_y=map_min_y, map_max_y=map_max_y,
            keepout_polygons=keepout_polygons)
        self.behaviors = [self.ray_behavior, self.frontier_behavior]

    def set_task_planner(self, task_planner):
        """Attaches the MAIPP task layer to frontier viewpoint selection."""
        self.frontier_behavior.task_planner = task_planner

    def mode_select(self, queries_labels, target_objects, queries_feats, mapper, publisher_dict, subscriber_dict):
        for behavior in self.behaviors:
            if behavior.condition_check(queries_labels, target_objects, queries_feats, mapper, publisher_dict, subscriber_dict):
                self.behavior_mode = behavior.name
                return

    def behavior_execute(self, behavior_mode, mapper, point3d_dict, waypoint_locked, publisher_dict, subscriber_dict):
        if behavior_mode == 'Frontier-based':
            wp_locked, tw1, tw2 = self.frontier_behavior.execute(mapper, point3d_dict, waypoint_locked, publisher_dict, subscriber_dict)
            return wp_locked, tw1, tw2

        #elif behavior_mode == 'Voxel-based':
        #    wp_locked, tw1, tw2 = self.voxel_behavior.execute(mapper, point3d_dict, waypoint_locked, publisher_dict)
        #    return wp_locked, tw1, tw2

        
        elif behavior_mode == 'Ray-based':
            wp_locked, tw1, tw2 = self.ray_behavior.execute(mapper, point3d_dict, waypoint_locked, publisher_dict, subscriber_dict)
            return wp_locked, tw1, tw2
