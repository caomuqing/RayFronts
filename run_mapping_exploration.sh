#!/bin/bash

cd /workspace/RayFronts

# All models (RADIO, SigLIP2) are cached in /root/.cache; skip HF Hub HTTP
# checks so startup works without internet (and is faster with it). Unset
# when downloading a new/changed model version for the first time.
export HF_HUB_OFFLINE=1

source /opt/ros/humble/setup.bash

# Bridge the MAIPP coordination topics between the high-flyer (domain 1)
# and this starling (domain 2). Runs alongside the planner; killed on exit.
# BRIDGE_PID=""
# if ros2 pkg prefix domain_bridge > /dev/null 2>&1; then
#   ros2 run domain_bridge domain_bridge experiments/maipp_domain_bridge.yaml &
#   BRIDGE_PID=$!
#   echo "MAIPP domain bridge running (pid $BRIDGE_PID)."
# else
#   echo "WARNING: domain_bridge not installed; robot_1 <-> robot_2 topics" \
#        "will NOT cross domains (apt install ros-humble-domain-bridge)."
# fi
# trap '[ -n "$BRIDGE_PID" ] && kill $BRIDGE_PID 2> /dev/null' EXIT

HYDRA_FULL_ERROR=1 python3 -m rayfronts.exploration_planner --config-dir experiments/preset_configs --config-name starlingmax_decoupled_bag