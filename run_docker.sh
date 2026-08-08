#!/bin/bash
# Launch a RayFronts container for one low-flying robot.
#   bash run_docker.sh [robot_id] [command...]
# robot_id defaults to 2. robot_2 -> ROS_DOMAIN_ID 2, GPU 0;
# robot_3 -> ROS_DOMAIN_ID 3, GPU 1. Without a command you get a shell;
# with one (e.g. bash run_mapping_exploration.sh) it runs directly and the
# container exits when it ends. Run in two terminals for both robots.

ROBOT_ID=${1:-2}
[ $# -gt 0 ] && shift
GPU_INDEX=$((ROBOT_ID - 2))

docker run -it --rm \
	--name rayfronts_robot${ROBOT_ID} \
	--gpus all \
	--network host \
	--ipc host \
	--privileged \
	--runtime=nvidia \
	-e NVIDIA_DRIVER_CAPABILITIES=all \
	-e CUDA_VISIBLE_DEVICES=${GPU_INDEX} \
	-e ROS_DOMAIN_ID=${ROBOT_ID} \
	-e ROBOT_ID=${ROBOT_ID} \
	-e DISPLAY="${DISPLAY:-:1}" \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	-v /home/airstationminipro/muqing_ws/rayfronts:/workspace/RayFronts \
	-v /home/airstationminipro/muqing_ws/Datasets:/workspace/Datasets \
	-v /home/airstationminipro/muqing_ws/.docker_cache:/root/.cache \
	-w /workspace/RayFronts \
	rayfronts:desktop "$@"
