"""Defines a ReRun.io implementation of a 3D mapping visualizer.

Typical usage:
  vis = RerunVis(intrinsics_3x3)
  vis.log_pose(pose_4x4, layer="cam0")
  vis.log_img(img, layer="rgb_img", pose_layer="cam0")
  vis.step()
"""
import os
from typing_extensions import override
from typing import Tuple

import rerun as rr
import torch

from rayfronts.visualizers.base import Mapping3DVisualizer
from rayfronts import feat_compressors


def _set_rerun_time(timeline: str, seconds: float) -> None:
  if hasattr(rr, "set_time_seconds"):
    rr.set_time_seconds(timeline, seconds)
  else:
    rr.set_time(timeline, duration=seconds)


class RerunVis(Mapping3DVisualizer):
  """Semantic RGBD visualizer using ReRun.io
  
  Attributes:
    intrinsics_3x3: See base.
    img_size: See base.
    base_point_size: See base.
    global_heat_scale: See base.
    device: See base.
    feat_compressor: See base.
    time_step: See base.
    split_label_vis: See __init__.
  """

  def __init__(self,
               intrinsics_3x3: torch.FloatTensor,
               img_size: Tuple[int] = None,
               base_point_size: float = None,
               global_heat_scale: bool = False,
               feat_compressor: feat_compressors.FeatCompressor = None,
               split_label_vis: bool = False,
               **kwargs):
    """

    Args:
      intrinsics_3x3: See base.
      img_size: See base.
      base_point_size: See base.
      global_heat_scale: See base.
      feat_compressor: See base.
      split_label_vis: Whether to log labeled points and arrows to different
        layers in rerun or not.
    """
    super().__init__(intrinsics_3x3, img_size, base_point_size,
                     global_heat_scale, feat_compressor)

    # Per-robot viewer: multiple robots on one machine each get their own
    # rerun instance (port 9874 + ROBOT_ID), otherwise the second planner
    # silently connects to the first robot's viewer and the streams mix.
    rid = os.environ.get("ROBOT_ID")
    port = int(os.environ.get(
      "RAYFRONTS_RERUN_PORT",
      str(9876 if rid is None else 9874 + int(rid))))
    # Viewer RAM cap (oldest timeline data is dropped when hit). The rerun
    # default of 75% of system RAM is fine solo but overcommits when two
    # robots run their own viewers on one machine.
    mem = os.environ.get("RAYFRONTS_RERUN_MEMLIMIT",
                         "75%" if rid is None else "30%")
    rr.init("semantic_mapping_vis" + ("" if rid is None else f"_robot{rid}"))
    rr.spawn(port=port, connect=True, memory_limit=mem)
    _set_rerun_time("stable_time", 0)
    self._base_name = "world"
    rr.log(self._base_name, rr.ViewCoordinates.RDF, static=True)
    rr.log(self._base_name,
      rr.Arrows3D(
        vectors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
      ))
    self._height = None
    self._width = None
    self._prev_poses_4x4 = dict()
    self.split_label_vis = split_label_vis

  @override
  def log_img(self,
              img: torch.FloatTensor,
              layer: str = "img",
              pose_layer: str = "pose") -> None:
    self._height, self._width, _ = img.shape
    rr.log(f"{self._base_name}/{pose_layer}/{layer}", rr.Image(img.cpu()))

  @override
  def log_depth_img(self,
                    depth_img: torch.FloatTensor,
                    layer: str = "img_depth",
                    pose_layer: str = "pose") -> None:
    self._height, self._width = depth_img.shape
    rr.log(f"{self._base_name}/{pose_layer}/{layer}",
           rr.DepthImage(depth_img.cpu()))

  @override
  def log_pose(self,
               pose_4x4: torch.FloatTensor,
               layer: str = "pose") -> None:
    rr.log(f"{self._base_name}/{layer}",
           rr.Pinhole(image_from_camera=self.intrinsics_3x3,
                      height=self._height, width=self._width,
                      camera_xyz=rr.ViewCoordinates.RDF))
    rr_transform = rr.Transform3D(translation=pose_4x4[:3, 3].cpu(),
                                  mat3x3=pose_4x4[:3, :3].cpu(),
                                  from_parent=False)
    rr.log(f"{self._base_name}/{layer}", rr_transform)

    if layer in self._prev_poses_4x4.keys():
      origin = self._prev_poses_4x4[layer][:3, -1]
      direction = pose_4x4[:3, -1] - origin
      rr_traj_arrows = rr.Arrows3D(origins=[origin.cpu()],
                                   vectors=[direction.cpu()],
                                   colors=[[255,0,0]])

      rr.log(f"{self._base_name}/{layer}_trajectory", rr_traj_arrows)
    self._prev_poses_4x4[layer] = pose_4x4

  @override
  def log_pc(self,
             pc_xyz: torch.FloatTensor,
             pc_rgb: torch.FloatTensor = None,
             pc_radii: torch.FloatTensor = None,
             layer: str = "pc"):
    super().log_pc(pc_xyz, pc_rgb, layer)
    radii = self.base_point_size if pc_radii is None else pc_radii
    try:
      radii = radii.cpu()
    except AttributeError:
      pass
    pc_rgb = pc_rgb.cpu() if pc_rgb is not None else pc_rgb
    rr.log(f"{self._base_name}/{layer}",
           rr.Points3D(positions=pc_xyz.cpu(), colors=pc_rgb, radii=radii))

  @override
  def log_arrows(self, arr_origins, arr_dirs, arr_rgb = None, layer="arrows"):
    rr.log(f"{self._base_name}/{layer}",
           rr.Arrows3D(vectors=(arr_dirs*self.base_point_size*5).cpu(),
                       origins=arr_origins.cpu(),
                       colors=arr_rgb.cpu(),
                       radii=self.base_point_size/3))

  @override
  def log_label_img(self, img_label, layer="img_label", pose_layer="pose"):
    rr.log(f"{self._base_name}/{pose_layer}/{layer}",
           rr.SegmentationImage(img_label.cpu()))

  @override
  def log_label_pc(self,
                   pc_xyz: torch.FloatTensor,
                   pc_labels: torch.FloatTensor = None,
                   layer: str = "pc_label"):
    if (self.split_label_vis and
        len(pc_labels.shape)==1 and
        pc_labels.shape[0] > 0):
      unique = torch.unique(pc_labels)
      pc_labels_onehot = torch.nn.functional.one_hot(pc_labels)
      for i in unique:
        rr.log(f"{self._base_name}/{layer}/{i}",
               rr.Points3D(positions=pc_xyz[pc_labels_onehot[..., i]==1].cpu(),
                           class_ids=i.item(),
                           radii=self.base_point_size))
    else:
      rr.log(f"{self._base_name}/{layer}",
            rr.Points3D(positions=pc_xyz.cpu(), class_ids=pc_labels.cpu(),
                        radii=self.base_point_size))

  @override
  def log_label_arrows(self, arr_origins, arr_dirs, arr_labels,
                       layer="arr_labels"):
    if (self.split_label_vis and len(arr_labels.shape)==1 and
        arr_labels.shape[0] > 0):
      unique = torch.unique(arr_labels)
      labels_onehot = torch.nn.functional.one_hot(arr_labels)
      for i in unique:
        m = labels_onehot[..., i]==1
        rr.log(f"{self._base_name}/{layer}/{i}",
              rr.Arrows3D(vectors=(arr_dirs*self.base_point_size*5)[m].cpu(),
                          origins=arr_origins[m].cpu(),
                          class_ids=i.item(), radii=self.base_point_size/3))
    else:
      rr.log(f"{self._base_name}/{layer}",
            rr.Arrows3D(vectors=(arr_dirs*self.base_point_size*5).cpu(),
                        origins=arr_origins.cpu(), class_ids=arr_labels.cpu(),
                        radii=self.base_point_size/3))
  @override
  def log_box(self, box_mins, box_maxs, layer = ""):
    box_centers = (box_maxs + box_mins) / 2
    box_half_sizes = (box_maxs - box_mins) / 2
    rr.log(f"{self._base_name}/{layer}_boxes",
           rr.Boxes3D(centers=box_centers.cpu(),
                      half_sizes=box_half_sizes.cpu()))

  def _log_axes(self, pose_4x4, layer="axes"):
    for i in range(pose_4x4.shape[0]):
      rr_transform = rr.Transform3D(
        translation=pose_4x4[i, :3, 3].cpu(), mat3x3=pose_4x4[i, :3, :3].cpu(),
        from_parent=False, axis_length=self.base_point_size*4,
        scale=2)
      rr.log(f"{self._base_name}/{layer}/{i}", rr_transform)

  def sync_thread_time(self) -> None:
    """Syncs the calling thread's rerun timeline to the current step.

    Rerun timelines are per-thread: logs from non-visualization threads (e.g.
    the exploration planner) would otherwise not appear at the viewer's
    playhead. Call before logging from such threads.
    """
    _set_rerun_time("stable_time", self.time_step * 0.1)

  def log_goal_pose(self,
                    pose_4x4: torch.FloatTensor,
                    target_xyz: torch.FloatTensor = None,
                    layer: str = "exploration_goal") -> None:
    """Logs a highly visible goal pose marker with its heading.

    Draws a large sphere at the goal position, a thick 1m heading arrow along
    the pose forward (+z) axis, and optionally a sight line to the target the
    goal intends to observe (target drawn in yellow).

    Safe to call from non-visualization threads: rerun timelines are
    per-thread, so this method first syncs the calling thread's timeline to
    the current step (logs from other threads would otherwise not appear at
    the viewer's playhead).
    """
    self.sync_thread_time()
    pos = pose_4x4[:3, 3].detach().cpu().reshape(1, 3)
    fwd = pose_4x4[:3, 2].detach().cpu().reshape(1, 3)
    green = [0, 255, 128]
    rr.log(f"{self._base_name}/{layer}/position",
           rr.Points3D(positions=pos, colors=[green],
                       radii=self.base_point_size * 4))
    rr.log(f"{self._base_name}/{layer}/heading",
           rr.Arrows3D(origins=pos, vectors=fwd * 1.0, colors=[green],
                       radii=self.base_point_size))
    if target_xyz is not None:
      t = target_xyz.detach().cpu().reshape(1, 3)
      rr.log(f"{self._base_name}/{layer}/target",
             rr.Points3D(positions=t, colors=[[255, 220, 0]],
                         radii=self.base_point_size * 3))
      strip = torch.cat([pos, t], dim=0).numpy()
      rr.log(f"{self._base_name}/{layer}/sight_line",
             rr.LineStrips3D(strips=[strip], colors=[green],
                             radii=self.base_point_size * 0.3))

  @override
  def step(self):
    super().step()
    _set_rerun_time("stable_time", self.time_step * 0.1)
