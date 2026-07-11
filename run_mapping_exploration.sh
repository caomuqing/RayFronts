#!/bin/bash

cd /workspace/RayFronts

HYDRA_FULL_ERROR=1 python3 -m rayfronts.exploration_planner --config-dir experiments/preset_configs --config-name starlingmax_decoupled_bag