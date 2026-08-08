#!/bin/bash
# Launch the exploration planner (+ its MAIPP domain bridges) for one robot.
#   bash run_mapping_exploration.sh [robot_id]
# robot_id defaults to $ROBOT_ID (set by run_docker.sh), then to
# $ROS_DOMAIN_ID, then to 2. Low-flyers are robot_2 (domain 2) and
# robot_3 (domain 3); the high-flyer is robot_1 (domain 1).

cd /workspace/RayFronts

ROBOT_ID=${1:-${ROBOT_ID:-${ROS_DOMAIN_ID:-2}}}
# Peers: everyone in {1, 2, 3} except ourselves.
PEERS=$(python3 -c "print([i for i in (1, 2, 3) if i != ${ROBOT_ID}])")

# All models (RADIO, SigLIP2) are cached in /root/.cache; skip HF Hub HTTP
# checks so startup works without internet (and is faster with it). Unset
# when downloading a new/changed model version for the first time.
export HF_HUB_OFFLINE=1

source /opt/ros/humble/setup.bash

# MAIPP domain bridges for this robot (own topics out to the other two
# domains + robot_1's topics in; one process per config file, since a
# topic key can only carry one from/to pair). Killed on exit.
BRIDGE_PIDS=()
if ros2 pkg prefix domain_bridge > /dev/null 2>&1; then
  for cfg in experiments/bridges/maipp_robot${ROBOT_ID}_d*.yaml; do
    [ -e "$cfg" ] || continue
    ros2 run domain_bridge domain_bridge "$cfg" &
    BRIDGE_PIDS+=($!)
    echo "MAIPP bridge $cfg running (pid ${BRIDGE_PIDS[-1]})."
  done
else
  echo "WARNING: domain_bridge not installed; inter-robot MAIPP topics" \
       "will NOT cross domains (apt install ros-humble-domain-bridge)."
fi
trap '[ ${#BRIDGE_PIDS[@]} -gt 0 ] && kill "${BRIDGE_PIDS[@]}" 2> /dev/null' EXIT

echo "Starting exploration planner as robot_${ROBOT_ID} (peers ${PEERS})."
HYDRA_FULL_ERROR=1 python3 -m rayfronts.exploration_planner \
  --config-dir experiments/preset_configs \
  --config-name starlingmax_decoupled_bag \
  exploration.robot_id=${ROBOT_ID} \
  "exploration.peer_robot_ids=${PEERS}"
