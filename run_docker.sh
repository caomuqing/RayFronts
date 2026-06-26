#!/bin/bash

docker run -it \
	--gpus all \
	--network host \
	--ipc host \
	--runtime=nvidia \
	-e NVIDIA_DRIVER_CAPABILITIES=all \
	-e ROS_DOMAIN_ID=1 \
	-v ~/RAVEN-MACVO/RayFronts_muqing:/workspace/RayFronts \
	-w /workspace/RayFronts \
	seungch2/rayfronts:jetson-radio-b-v1
