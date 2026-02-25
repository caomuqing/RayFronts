#!/bin/bash

cd /workspace/RayFronts

HYDRA_FULL_ERROR=1 python3 -m rayfronts.mapping_server \
	dataset=ros2macslam \
	mapping=semantic_ray_frontiers_map \
	mapping.vox_size=0.5 \
	dataset.rgb_resolution=[320,320] \
	dataset.depth_resolution=[320,320] \
	dataset.frame_skip=10 \
	mapping.max_rays_per_frame=10000 


