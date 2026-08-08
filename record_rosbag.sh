#!/bin/bash
# Lean field recording: the pipeline inputs needed for offline replay, this
# robot's shared MAIPP task topics, and the published goals.
# (record_bag.sh is the superset: it also captures goal feedback/gating and
# the peer robots' MAIPP topics.)
#
# Run inside the robot's container:  bash record_rosbag.sh [output_name]
# The robot id comes from $ROBOT_ID (set by run_docker.sh), defaulting to
# $ROS_DOMAIN_ID, then 2.

RID=${ROBOT_ID:-${ROS_DOMAIN_ID:-2}}
OUT_DIR=/workspace/RayFronts/rosbags
NAME=${1:-robot${RID}_$(date +%Y_%m_%d-%H_%M_%S)}
mkdir -p "$OUT_DIR"

TOPICS=(
  /drone/detections
  /drone/image_raw
  /odom_laser_ned_relay
  /registered_scan_relay
  /robot_${RID}/maipp/coverage_grid
  /robot_${RID}/maipp/tracks
  /robot_${RID}/maipp/task_claim
  /goal_point
)

echo "Recording ${#TOPICS[@]} topics (robot_${RID}) to $OUT_DIR/$NAME"
ros2 bag record -o "$OUT_DIR/$NAME" "${TOPICS[@]}"
