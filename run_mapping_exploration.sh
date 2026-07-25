#!/bin/bash

cd /workspace/RayFronts

# All models (RADIO, SigLIP2) are cached in /root/.cache; skip HF Hub HTTP
# checks so startup works without internet (and is faster with it). Unset
# when downloading a new/changed model version for the first time.
export HF_HUB_OFFLINE=1

HYDRA_FULL_ERROR=1 python3 -m rayfronts.exploration_planner --config-dir experiments/preset_configs --config-name starlingmax_decoupled_bag