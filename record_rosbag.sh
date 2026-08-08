#!/bin/bash
# Lean field recording: the pipeline inputs needed for offline replay, this
# starling's (robot_2) shared MAIPP task topics, and the published goals.
# (record_bag.sh is the superset: it also captures goal feedback/gating and
# the peer robot_1 topics.)
#
# Run inside the container:  bash record_rosbag.sh [output_name]

OUT_DIR=/workspace/RayFronts/rosbags
NAME=${1:-starlingmax_$(date +%Y_%m_%d-%H_%M_%S)}
mkdir -p "$OUT_DIR"

TOPICS=(
  /drone/detections
  /drone/image_raw
  /odom_laser_ned_relay
  /registered_scan_relay
  /robot_2/maipp/coverage_grid
  /robot_2/maipp/tracks
  /robot_2/maipp/task_claim
  /goal_point
)

echo "Recording ${#TOPICS[@]} topics to $OUT_DIR/$NAME"
ros2 bag record -o "$OUT_DIR/$NAME" "${TOPICS[@]}"
