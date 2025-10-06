#!/bin/bash

cd /workspace/RayFronts

HYDRA_FULL_ERROR=1 python3 scripts/semseg_eval.py \
  --config-dir experiments/semseg_configs/ \
  --config-name ros2isaacsim_naradio