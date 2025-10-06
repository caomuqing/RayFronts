#!/bin/bash

sudo docker run -it --gpus all --network host --ipc host --runtime=nvidia \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e ROS_DOMAIN_ID=1 \
  -v ~/RayFronts:/workspace/RayFronts \
  -w /workspace/RayFronts \
  rayfronts:desktop
