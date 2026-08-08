#!/bin/bash
# Records everything needed to replay the mapping/exploration pipeline
# offline, plus the planner's outputs (goals + the MAIPP belief topics
# shared with other robots) for post-flight analysis.
#
# Run inside the container (same ROS_DOMAIN_ID as the pipeline):
#   bash record_bag.sh [output_name]
#
# Replay later with:
#   ros2 bag play <bag> --loop
# (the pipeline inputs are enough to re-run run_mapping_exploration.sh
# against the bag; the recorded outputs let you compare/analyze without
# re-running).

ROBOT_ID=2        # this starling (matches exploration.robot_id)
PEER_IDS=(1)      # peers whose shared belief we also record

OUT_DIR=/workspace/Datasets/recordings
NAME=${1:-starlingmax_$(date +%Y_%m_%d-%H_%M_%S)}
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
for rid in $ROBOT_ID "${PEER_IDS[@]}"; do
  TOPICS+=(
    /robot_${rid}/maipp/coverage_grid
    /robot_${rid}/maipp/tracks
    /robot_${rid}/maipp/task_claim
  )
done

echo "Recording ${#TOPICS[@]} topics to $OUT_DIR/$NAME"
ros2 bag record -o "$OUT_DIR/$NAME" "${TOPICS[@]}"
