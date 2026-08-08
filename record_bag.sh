#!/bin/bash
# Records everything needed to replay the mapping/exploration pipeline
# offline, plus the planner's outputs (goals + the MAIPP belief topics
# shared with other robots, including what the peers sent us) for
# post-flight analysis and offline fusion replay.
#
# Run inside the robot's container:  bash record_bag.sh [output_name]
# The robot id comes from $ROBOT_ID (set by run_docker.sh), defaulting to
# $ROS_DOMAIN_ID, then 2. Peers are the other members of {1, 2, 3}.

RID=${ROBOT_ID:-${ROS_DOMAIN_ID:-2}}
OUT_DIR=/workspace/RayFronts/rosbags
NAME=${1:-robot${RID}_full_$(date +%Y_%m_%d-%H_%M_%S)}
mkdir -p "$OUT_DIR"

# ---- Pipeline inputs (required for replay) ----
TOPICS=(
  /drone/image_raw          # RGB for semantics (+ what the detector saw)
  /odom_laser_ned_relay     # pose (RGB sync + scan carve origin)
  /registered_scan_relay    # geometry (occupancy + frontiers)
  /drone/detections         # person detections (empty frames matter too)
)

# ---- Planner outputs: goal interface ----
TOPICS+=(
  /goal_point               # published goals (NED pose)
  /goal_reach_status        # follower feedback that drove goal cadence
  /goal_publish_allow       # gating switch state changes
)

# ---- MAIPP belief shared with other robots (ours + what peers sent) ----
for rid in 1 2 3; do
  TOPICS+=(
    /robot_${rid}/maipp/coverage_grid
    /robot_${rid}/maipp/tracks
    /robot_${rid}/maipp/task_claim
  )
done

echo "Recording ${#TOPICS[@]} topics (robot_${RID}) to $OUT_DIR/$NAME"
ros2 bag record -o "$OUT_DIR/$NAME" "${TOPICS[@]}"
