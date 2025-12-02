#!/bin/bash

cd /workspace/RayFronts

HYDRA_FULL_ERROR=1 python3 -m rayfronts.mapping_server \
	dataset=ros2isaacsim \
	mapping=semantic_voxel_map \
	mapping.vox_size=0.1 \
	dataset.rgb_resolution=[224,224] \
	dataset.depth_resolution=[224,224] \
	dataset.frame_skip=10 \
	depth_limit=20
