#!/bin/bash

docker run -it \
	--gpus all \
	--network host \
	--ipc host \
	--runtime=nvidia \
	-e NVIDIA_DRIVER_CAPABILITIES=all \
	-e ROS_DOMAIN_ID=2 \
	-v ~/RayFronts:/workspace/RayFronts \
	-w /workspace/RayFronts \
	seungch2/rayfronts:jetson-savemodel
