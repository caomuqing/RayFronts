#!/bin/bash

docker run -it --rm \
	--name rayfronts_container \
	--gpus all \
	--network host \
	--ipc host \
	--privileged \
	--runtime=nvidia \
	-e NVIDIA_DRIVER_CAPABILITIES=all \
	-e ROS_DOMAIN_ID=2 \
	-e DISPLAY="${DISPLAY:-:1}" \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	-v /home/airstationminipro/muqing_ws/rayfronts:/workspace/RayFronts \
	-v /home/airstationminipro/muqing_ws/Datasets:/workspace/Datasets \
	-v /home/airstationminipro/muqing_ws/.docker_cache:/root/.cache \
	-w /workspace/RayFronts \
	rayfronts:desktop