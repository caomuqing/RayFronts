"""Defines ROS related datasets

Typical usage example:
  dataset = Ros2Subscriber(
    rgb_topic="/robot/front_stereo/left/image_rect_color",
    pose_topic="/robot/front_stereo/pose",
    disparity_topic="/robot/front_stereo/disparity/disparity_image",
    intrinsics_topic="/robot/front_stereo/left/camera_info",
    src_coord_system="flu")

  dataloader = torch.utils.data.DataLoader(
    self.dataset, batch_size=4)

  for i, batch in enumerate(dataloader):
    rgb_img = batch["rgb_img"].cuda()
    depth_img = batch["depth_img"].cuda()
    pose_4x4 = batch["pose_4x4"].cuda()
"""

import os
import threading
import queue
from typing_extensions import override, deprecated
from typing import Tuple, Union
from collections import OrderedDict
import logging
import json
from scipy.spatial.transform import Rotation as R

logger = logging.getLogger(__name__)

import numpy as np
from scipy.spatial.transform import Rotation
import torch

try:
  import rclpy
  from rclpy.node import Node
  from rclpy.executors import SingleThreadedExecutor
  from rclpy.qos import QoSProfile, ReliabilityPolicy
  import message_filters
  from sensor_msgs.msg import Image, CameraInfo, PointCloud
  from nav_msgs.msg import Odometry
  from geometry_msgs.msg import PoseStamped
  from stereo_msgs.msg import DisparityImage
  from std_msgs.msg import String
  from nav_msgs.msg import Odometry
  from rayfronts.ros_utils import image_to_numpy, pose_to_numpy
except ModuleNotFoundError:
  logger.warning("ROS2 modules not found !")

from rayfronts.datasets.base import PosedRgbdDataset
from rayfronts import geometry3d as g3d

class Ros2Subscriber(PosedRgbdDataset):
  """ROS2 subscriber node to subscribe to posed RGBD topics.
  
  Attributes:
    intrinsics_3x3:  See base.
    rgb_h: See base.
    rgb_w: See base.
    depth_h: See base.
    depth_w: See base.
    frame_skip: See base.
    interp_mode: See base.
  """
  def __init__(self,
               rgb_topic,
               pose_topic,
               rgb_resolution=None,
               depth_resolution=None,
               disparity_topic = None,
               depth_topic = None,
               confidence_topic = None,
               point_cloud_topic = None,
               intrinsics_topic = None,
               intrinsics_file = None,
               src_coord_system = "flu",
               frame_skip = 0,
               interp_mode="bilinear",
               scene_name: str = "isaac_sim_robot"):
    """

    There can be three sources of depth:
    1- Disparity topic
    2- Depth topic
    3- Point cloud topic (will be projected using pose and intrinsics)
       Using the point cloud through this rgbd loader is inefficient as points
       will be projected then likely unprojected again in the mapping system.

    Args:
      rgb_resolution: See base.
      depth_resolution: See base.
      rgb_topic: Topic containing RGB images of type sensor_msgs/msg/Image
      pose_topic: Topic containing poses of type geometry_msgs/msg/PoseStamped
      disparity_topic: Topic containing disparity images of type
        stereo_msgs/DisparityImage.
      depth_topic: Topic containing depth images of type sensor_msgs/msg/Image
        with 32FC1 encoding in metric scale.
      confidence_topic: (Optional) Topic containing confidence in depth values.
        Message type: sensor_msgs/msg/Image.
      point_cloud_topic: Topic containing point cloud of type
        sensor_msgs/msg/PointCloud.
      intrinsics_topic: Topic containing intrinsics information from messages
        of type sensor_msgs/msg/CameraInfo. Will be used at initialization only.
      intrinsics_file: Path to json file containing intrinsics with the
        following keys, fx, fy, cx, cy, w, h. This will be prioritized
        over the intrinsics topic.
      src_coord_system: A string of 3 letters describing the camera coordinate
        system in r/l u/d f/b in any order. (e.g, rdf, flu, rfu)
      frame_skip: See base.
      interp_mode: See base.
    """
    super().__init__(rgb_resolution=rgb_resolution,
                     depth_resolution=depth_resolution,
                     frame_skip=frame_skip,
                     interp_mode=interp_mode)

    if point_cloud_topic is not None and disparity_topic is not None:
      raise ValueError("You cannot set both the point cloud topic and "
                       "disparity topic as that will lead to an ambiguous "
                       "source of depth information.")

    if intrinsics_file is None and intrinsics_topic is None:
      raise ValueError("Must provide a source for the intrinsics")

    self._shutdown_event = threading.Event()

    self.f = 0
    self.intrinsics_3x3 = None
    if intrinsics_file is not None:
      intrinsics_topic = None
      with open(intrinsics_file, "r") as f:
        int_json = json.load(f)
        self.intrinsics_3x3 = torch.tensor([
          [int_json["fx"], 0, int_json["cx"]],
          [0, int_json["fy"], int_json["cy"]],
          [0, 0, 1]
        ])
    self._intrinsics_loaded_cond = threading.Condition()
    self.src2rdf_transform = g3d.mat_3x3_to_4x4(
      g3d.get_coord_system_transform(src_coord_system, "rdf"))

    # Setup ros node
    msg_str_to_type = OrderedDict(
      rgb = Image,
      pose = Odometry,
      disp = DisparityImage,
      depth = Image,
      pc = PointCloud,
      conf = Image,
    )
    self._topics = [rgb_topic, pose_topic, disparity_topic, depth_topic, point_cloud_topic,
                   confidence_topic]
    if not rclpy.ok():
      rclpy.init()
    self._rosnode = Node("rayfronts_input_streamer")

    if intrinsics_topic is not None:
      self.intrinsics_sub = self._rosnode.create_subscription(
        CameraInfo, intrinsics_topic, self._set_intrinsics_from_msg,
        QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=1))

    self._subs = OrderedDict()
    for i, t in enumerate(self._topics):
      msg_str = list(msg_str_to_type.keys())[i]
      if t is not None:
        self._subs[msg_str] = message_filters.Subscriber(
          self._rosnode, msg_str_to_type[msg_str], t, qos_profile = 10)
    self._frame_msgs_queue = queue.Queue(maxsize=10)

    self._time_sync = message_filters.ApproximateTimeSynchronizer(
      list(self._subs.values()), queue_size = 10, slop = 0.01,
      allow_headerless = False)
    self._time_sync.registerCallback(self._buffer_frame_msgs)

    self._ros_executor = SingleThreadedExecutor()
    self._ros_executor.add_node(self._rosnode)
    self._spin_thread = threading.Thread(
      target=self._spin_ros, name="rayfronts_input_stream_spinner")
    self._spin_thread.daemon = True

    if intrinsics_topic is not None:
      self._intrinsics_loaded_cond.acquire()
      self._spin_thread.start()
      logger.info("Waiting for intrinsics to be published..")
      while True:
        try:
          r = self._intrinsics_loaded_cond.wait(2)
        except KeyboardInterrupt as e:
          self.shutdown()
          raise e
        if r:
          self._rosnode.destroy_subscription(self.intrinsics_sub)
          break
    else:
      self._spin_thread.start()

    logger.info("Ros2Subscriber initialized successfully.")

  def _spin_ros(self):
    try:
      self._ros_executor.spin()
    except (KeyboardInterrupt,
            rclpy.executors.ExternalShutdownException,
            rclpy.executors.ShutdownException):
      pass

  def _set_intrinsics_from_msg(self, msg):
    self._intrinsics_loaded_cond.acquire()
    self.intrinsics_3x3 = torch.tensor(msg.k, dtype = torch.float).reshape(3,3)
    self.original_h = msg.height
    self.original_w = msg.width
    self.rgb_h = self.original_h if self.rgb_h <= 0 else self.rgb_h
    self.rgb_w = self.original_w if self.rgb_w <= 0 else self.rgb_w
    self.depth_h = self.original_h if self.depth_h <= 0 else self.depth_h
    self.depth_w = self.original_w if self.depth_w <= 0 else self.depth_w

    if self.depth_h != self.original_h or self.depth_w != self.original_w:
      h_ratio = self.depth_h / self.original_h
      w_ratio = self.depth_w / self.original_w
      self.intrinsics_3x3[0, :] = self.intrinsics_3x3[0, :] * w_ratio
      self.intrinsics_3x3[1, :] = self.intrinsics_3x3[1, :] * h_ratio

    logger.info("Loaded intrinsics: \n%s", str(self.intrinsics_3x3))
    self._intrinsics_loaded_cond.notify()
    self._intrinsics_loaded_cond.release()

  def _buffer_frame_msgs(self, *msgs):
    if self.frame_skip <= 0 or self.f % (self.frame_skip+1) == 0:
      if self._frame_msgs_queue.full():
        self._frame_msgs_queue.get() # Discard and priortize newer.
      self._frame_msgs_queue.put(msgs)
    self.f += 1

  def __iter__(self):
    while True:
      msgs = None
      try:
        msgs = self._frame_msgs_queue.get(block=True, timeout=2)
      except queue.Empty:
        if not self._shutdown_event.is_set():
          continue

      if msgs is None:
        return

      msgs = dict(zip(self._subs.keys(), msgs))

      # Parse RGB
      # bgra_img = image_to_numpy(msgs["rgb"]).astype("float") / 255
      # bgr_img = bgra_img[..., :3]
      # rgb_img = torch.tensor(bgr_img[..., (2,1,0)],
      #                        dtype=torch.float).permute(2, 0, 1)
      rgba_img = image_to_numpy(msgs["rgb"]).astype("float") / 255
      rgb_img = rgba_img[..., :3]
      rgb_img = torch.tensor(rgb_img, dtype=torch.float).permute(2,0,1)

      # Parse Pose
      src_pose_4x4 = torch.tensor(
        pose_to_numpy(msgs["pose"].pose), dtype=torch.float)
      if True:
        # Static transform from base_link to camera (CameraLeft)
        translation = np.array([0.161, 0.060, 0.712], dtype=np.float32)
        # rpy = [-1.745, 0.000, -1.571]  # in radians
        rpy = [0.0, 0.1745, 0.0]  # in radians
        rotation_matrix = R.from_euler('xyz', rpy).as_matrix()

        # Construct 4x4 transform matrix in numpy
        T_base_to_cam_np = np.eye(4, dtype=np.float32)
        T_base_to_cam_np[:3, :3] = rotation_matrix
        T_base_to_cam_np[:3, 3] = translation

        # Convert to PyTorch tensor on same device and dtype as src_pose_4x4
        T_base_to_cam = torch.from_numpy(T_base_to_cam_np).to(src_pose_4x4.device, dtype=src_pose_4x4.dtype)

        # Transform base_link pose to camera pose in the map frame
        src_pose_4x4 = src_pose_4x4 @ T_base_to_cam
      rdf_pose_4x4 = g3d.transform_pose_4x4(
        src_pose_4x4, self.src2rdf_transform)

      if 'depth' in msgs.keys():
        depth_img = image_to_numpy(msgs["depth"])
        depth_img = torch.tensor(depth_img, dtype=torch.float).unsqueeze(0)
      elif "disp" in msgs.keys():
        # TODO: Why is disparity negative in ros2 zedx and why is max and min
        # flipped? Not sure if this is correct ros2 zedx behaviour but will
        # correct those here for now.
        disparity_img = -image_to_numpy(msgs["disp"].image)
        min_disp = msgs["disp"].max_disparity
        max_disp = msgs["disp"].min_disparity

        focal_length = msgs["disp"].f
        stereo_baseline = msgs["disp"].t
        depth_img = focal_length*stereo_baseline/disparity_img
        depth_img[disparity_img < min_disp] = np.inf
        depth_img[disparity_img > max_disp] = -np.inf
        depth_img = torch.tensor(depth_img, dtype=torch.float).unsqueeze(0)

      elif "pc" in msgs.keys():
        # TODO: This should be more efficient than a for loop
        pc_xyz = torch.tensor([[p.x, p.y, p.z] for p in msgs["pc"].points])
        if len(pc_xyz) == 0:
          continue
        pc_xyz_homo = g3d.pts_to_homogen(pc_xyz)
        pc_xyz_homo = g3d.transform_points_homo(pc_xyz_homo,
                                                self.src2rdf_transform)
        pc_xyz_homo_cam = pc_xyz_homo @ torch.linalg.inv(rdf_pose_4x4).T
        pc_xyz_homo_cam /= pc_xyz_homo_cam[:, -1].unsqueeze(-1)
        pc_depth = pc_xyz_homo_cam[:, 2]
        ch_n2i = {ch.name: i for i,ch in enumerate(msgs["pc"].channels)}
        if "kp_u" in ch_n2i and "kp_v" in ch_n2i:
          u = torch.tensor(msgs["pc"].channels[ch_n2i["kp_u"]].values,
                       dtype=torch.int32)
          v = torch.tensor(msgs["pc"].channels[ch_n2i["kp_v"]].values,
                          dtype=torch.int32)
        else:
          uv = pc_xyz_homo_cam[:, :3] @ self.intrinsics_3x3.T
          uv /= uv[:, -1]
          u = uv[:, 0]
          v = uv[:, 1]

        depth_img = torch.ones_like(rgb_img[0:1])*torch.nan
        mask = torch.logical_and(v < rgb_img.shape[1], u < rgb_img.shape[2])
        depth_img[:, v[mask], u[mask]] = pc_depth[mask]
        depth_img[depth_img < 0] = -torch.infs
      else:
        raise ValueError("Expected at least a depth or a disparity or point cloud topic")

      # Parse confidence map if it exists
      conf_img = None
      if "conf" in msgs:
        conf_img = image_to_numpy(msgs["conf"]).astype("float")
        conf_img = 1 - (torch.tensor(conf_img, dtype=torch.float) / 100)
        conf_img = conf_img.unsqueeze(0)

      if (self.rgb_h != rgb_img.shape[-2] or
          self.rgb_w != rgb_img.shape[-1]):
        rgb_img = torch.nn.functional.interpolate(rgb_img.unsqueeze(0),
          size=(self.rgb_h, self.rgb_w), mode=self.interp_mode,
          antialias=self.interp_mode in ["bilinear", "bicubic"]).squeeze(0)

      if (self.depth_h != depth_img.shape[-2] or
          self.depth_w != depth_img.shape[-1]):
        depth_img = torch.nn.functional.interpolate(depth_img.unsqueeze(0),
          size=(self.depth_h, self.depth_w),
          mode="nearest-exact").squeeze(0)
        if conf_img is not None:
          conf_img = torch.nn.functional.interpolate(conf_img.unsqueeze(0),
            size=(self.depth_h, self.depth_w),
            mode="nearest-exact").squeeze(0)

      if torch.sum(~depth_img.isnan()) == 0:
        logger.warning("Ignoring received depth frame with no valid values")
        continue
      frame_data = dict(rgb_img = rgb_img, depth_img = depth_img,
                        pose_4x4 = rdf_pose_4x4)

      if conf_img is not None:
        frame_data["confidence_map"] = conf_img

      yield frame_data

  def shutdown(self):
    self._shutdown_event.set()
    self._rosnode.context.try_shutdown()
    logger.info("Ros2Subscriber shutdown.")

@deprecated("Use Ros2Subscriber instead")
class RosnpyDataset(PosedRgbdDataset):
  """Processes datasets produced by the ros2npy utility from scripts dir.
  
  The ros2npy utility is located in the scripts directory and it converts ROS 
  bags to npz files to drop the ros dependency. The format can be quite slow 
  since it requires loading huge chunks of memory at a time.

  This will be removed in the future and replaced by a ROS1 bag reader or
  subscriber.

  Attributes:
    intrinsics_3x3:  See base.
    rgb_h: See base.
    rgb_w: See base.
    depth_h: See base.
    depth_w: See base.
    frame_skip: See base.
    interp_mode: See base.
  """

  def __init__(self,
               path: str,
               rgb_resolution: Union[Tuple[int], int] = None,
               depth_resolution: Union[Tuple[int], int] = None,
               frame_skip: int = 0,
               interp_mode: str = "bilinear"):
    """
    Args:
      path: Path to directory. if path ends with .npz only a single file is 
        loaded. If the path is a directory then all .npz files within that
        directory will be loaded in lexsorted order assuming that order
        corresponds to the chronological order as well.
      rgb_resolution: See base.
      depth_resolution: See base.
      frame_skip: See base.
      interp_mode: See base.
    """
    super().__init__(rgb_resolution=rgb_resolution,
                     depth_resolution=depth_resolution,
                     frame_skip=frame_skip,
                     interp_mode=interp_mode)

    if os.path.isdir(path):
      self._data_files = [os.path.join(path, x)
                         for x in os.listdir(path)
                         if x.endswith(".npz")]

      self._data_files = sorted(self._data_files)
    else:
      self._data_files = [path]

    self._data_files_index = 0
    with np.load(self._data_files[self._data_files_index]) as npz_file:
      self._loaded_file = dict(npz_file.items())
    self._next_loaded_file = None
    self._loaded_file_frame = 0

    self._data_files_index += 1
    self._prefetch_thread = None
    if self._data_files_index < len(self._data_files):
      self._prefetch_thread = threading.Thread(target=self._prefetch_next_file)
      self._prefetch_thread.start()

    self.intrinsics_3x3 = self._loaded_file["intrinsics_3x3"][0].reshape(3,3)
    self.intrinsics_3x3 = torch.tensor(self.intrinsics_3x3, dtype=torch.float)

    self.original_h, self.original_w = self._loaded_file["rgb_img"][0].shape[:2]
    self.rgb_h = self.original_h if self.rgb_h <= 0 else self.rgb_h
    self.rgb_w = self.original_w if self.rgb_w <= 0 else self.rgb_w
    self.depth_h = self.original_h if self.depth_h <= 0 else self.depth_h
    self.depth_w = self.original_w if self.depth_w <= 0 else self.depth_w

    if self.depth_h != self.original_h or self.depth_w != self.original_w:
      h_ratio = self.depth_h / self.original_h
      w_ratio = self.depth_w / self.original_w
      self.intrinsics_3x3[0, :] = self.intrinsics_3x3[0, :] * w_ratio
      self.intrinsics_3x3[1, :] = self.intrinsics_3x3[1, :] * h_ratio

    # ROS (X-Forward, Y-Left, Z-Up) to OpenCV (X-Right, Y-Down, Z-Forward):
    self._flu2rdf_transform = g3d.mat_3x3_to_4x4(
      g3d.get_coord_system_transform("flu", "rdf"))

  def _prefetch_next_file(self):
    with np.load(self._data_files[self._data_files_index]) as npz_file:
      self._next_loaded_file = dict(npz_file.items())

  @override
  def __iter__(self):
    f = 0
    while True:
      seq_lens = [len(x) for x in self._loaded_file.values()]

      if self._loaded_file_frame >= min(seq_lens):
        if self._prefetch_thread is None:
          break

        # Make sure prefetch thread has terminated
        self._prefetch_thread.join()

        # Load next file and reset frame index to 0
        self._loaded_file = self._next_loaded_file
        self._loaded_file_frame = 0
        self._next_loaded_file = None

        # Start loading of next file in seperate thread
        self._data_files_index += 1
        if self._data_files_index < len(self._data_files):
          self._prefetch_thread = threading.Thread(
            target=self._prefetch_next_file)
          self._prefetch_thread.start()
        else:
          self._prefetch_thread = None

      i = self._loaded_file_frame
      frames_data = self._loaded_file
      self._loaded_file_frame += 1

      if self.frame_skip > 0 and f % (self.frame_skip+1) != 0:
          f += 1
          continue
      f += 1

      flu_pose_t = frames_data["pose_t"][i]
      flu_pose_q = frames_data["pose_q_wxyz"][i]
      flu_pose_R = Rotation.from_quat(flu_pose_q, scalar_first=True).as_matrix()
      # TODO: Verify
      flu_pose_Rt_3x4 = np.concatenate((flu_pose_R, flu_pose_t.reshape(3, 1)),
                                        axis=1)
      flu_pose_Rt_3x4 = torch.tensor(flu_pose_Rt_3x4, dtype=torch.float)
      flu_pose_4x4 = g3d.mat_3x4_to_4x4(flu_pose_Rt_3x4)

      rdf_pose_4x4 = g3d.transform_pose_4x4(flu_pose_4x4,
                                            self._flu2rdf_transform)

      disparity_img = frames_data["disparity_img"][i]
      min_disp = frames_data["min_disparity"][i]
      max_disp = frames_data["max_disparity"][i]

      focal_length = frames_data["focal_length"][i]
      stereo_baseline = frames_data["stereo_baseline"][i]

      depth_img = focal_length*stereo_baseline/disparity_img
      #TODO: Why does this have to be flipped ?
      depth_img[disparity_img < min_disp] = -np.inf
      depth_img[disparity_img > max_disp] = np.inf
      depth_img = torch.tensor(depth_img, dtype=torch.float).unsqueeze(0)

      rgb_img = frames_data["rgb_img"][i]
      rgb_img = torch.tensor(rgb_img, dtype=torch.float32).permute(2, 0, 1)/255

      if (self.rgb_h != rgb_img.shape[-2] or
          self.rgb_w != rgb_img.shape[-1]):
        rgb_img = torch.nn.functional.interpolate(rgb_img.unsqueeze(0),
          size=(self.rgb_h, self.rgb_w), mode=self.interp_mode,
          antialias=self.interp_mode in ["bilinear", "bicubic"]).squeeze(0)

      if (self.depth_h != depth_img.shape[-2] or
          self.depth_w != depth_img.shape[-1]):
        depth_img = torch.nn.functional.interpolate(depth_img.unsqueeze(0),
          size=(self.depth_h, self.depth_w),
          mode="nearest-exact").squeeze(0)

      frame_data = dict(rgb_img = rgb_img, depth_img = depth_img,
                        pose_4x4 = rdf_pose_4x4)
      yield frame_data

  def shutdown(self):
    if self._prefetch_thread is not None:
      self._prefetch_thread.join()


class Ros2SemSegSubscriber(PosedRgbdDataset):
  """ROS2 subscriber node to subscribe to posed RGBD topics with semantic segmentation.
  
  This extends the basic Ros2Subscriber to include semantic segmentation data from
  ROS2 topics. It subscribes to segmentation images and semantic label mappings.
  
  Attributes:
    intrinsics_3x3:  See base.
    rgb_h: See base.
    rgb_w: See base.
    depth_h: See base.
    depth_w: See base.
    frame_skip: See base.
    interp_mode: See base.
    _cat_id_to_name: Mapping from class IDs to class names
    _cat_index_to_cat_id: Mapping from contiguous indices to class IDs
    _cat_id_to_cat_index: Mapping from class IDs to contiguous indices
    _cat_index_to_cat_name: Mapping from contiguous indices to class names
    num_classes: Number of semantic classes
    cat_id_to_name: Property returning class ID to name mapping
  """
  
  def __init__(self,
               rgb_topic,
               pose_topic,
               semseg_topic,
               semantic_labels_topic,
               rgb_resolution=None,
               depth_resolution=None,
               disparity_topic=None,
               depth_topic=None,
               confidence_topic=None,
               point_cloud_topic=None,
               intrinsics_topic=None,
               intrinsics_file=None,
               src_coord_system="flu",
               frame_skip=0,
               interp_mode="bilinear",
               load_semseg=True,
               scene_name: str = "isaac_sim_robot"):
    """
    Args:
      rgb_resolution: See base.
      depth_resolution: See base.
      rgb_topic: Topic containing RGB images of type sensor_msgs/msg/Image
      pose_topic: Topic containing poses of type geometry_msgs/msg/PoseStamped
      semseg_topic: Topic containing segmentation images of type sensor_msgs/msg/Image
        with 32SC1 encoding containing class IDs
      semantic_labels_topic: Topic containing semantic label mappings of type
        std_msgs/msg/String with JSON format
      disparity_topic: Topic containing disparity images of type
        stereo_msgs/DisparityImage.
      depth_topic: Topic containing depth images of type sensor_msgs/msg/Image 
        with 32FC1 encoding in metric scale.
      confidence_topic: (Optional) Topic containing confidence in depth values.
        Message type: sensor_msgs/msg/Image.
      point_cloud_topic: Topic containing point cloud of type
        sensor_msgs/msg/PointCloud.
      intrinsics_topic: Topic containing intrinsics information from messages
        of type sensor_msgs/msg/CameraInfo. Will be used at initialization only.
      intrinsics_file: Path to json file containing intrinsics with the
        following keys, fx, fy, cx, cy, w, h. This will be prioritized
        over the intrinsics topic.
      src_coord_system: A string of 3 letters describing the camera coordinate
        system in r/l u/d f/b in any order. (e.g, rdf, flu, rfu)
      frame_skip: See base.
      interp_mode: See base.
      load_semseg: Whether to load semantic segmentation labels or not.
    """
    super().__init__(rgb_resolution=rgb_resolution,
                     depth_resolution=depth_resolution,
                     frame_skip=frame_skip,
                     interp_mode=interp_mode)
    
    self.load_semseg = load_semseg
    
    # Initialize semantic segmentation mappings
    self._cat_id_to_name = {}
    self._semantic_labels_loaded = False
    self._semantic_labels_cond = threading.Condition()
    
    if point_cloud_topic is not None and disparity_topic is not None:
      raise ValueError("You cannot set both the point cloud topic and "
                       "disparity topic as that will lead to an ambiguous "
                       "source of depth information.")

    if intrinsics_file is None and intrinsics_topic is None:
      raise ValueError("Must provide a source for the intrinsics")

    self._shutdown_event = threading.Event()

    self.f = 0
    self.intrinsics_3x3 = None
    if intrinsics_file is not None:
      intrinsics_topic = None
      with open(intrinsics_file, "r") as f:
        int_json = json.load(f)
        self.intrinsics_3x3 = torch.tensor([
          [int_json["fx"], 0, int_json["cx"]],
          [0, int_json["fy"], int_json["cy"]],
          [0, 0, 1]
        ])
    self._intrinsics_loaded_cond = threading.Condition()
    self.src2rdf_transform = g3d.mat_3x3_to_4x4(
      g3d.get_coord_system_transform(src_coord_system, "rdf"))

    # Setup ros node
    msg_str_to_type = OrderedDict(
      rgb=Image,
      pose=Odometry,
      disp=DisparityImage,
      depth=Image,
      pc=PointCloud,
      conf=Image,
      semseg=Image,
      semlabels=String,
    )
    self._topics = [rgb_topic, pose_topic, disparity_topic, depth_topic, 
                   point_cloud_topic, confidence_topic, semseg_topic, 
                   semantic_labels_topic]
    
    if not rclpy.ok():
      rclpy.init()
    self._rosnode = Node("rayfronts_semseg_input_streamer")

    if intrinsics_topic is not None:
      self.intrinsics_sub = self._rosnode.create_subscription(
        CameraInfo, intrinsics_topic, self._set_intrinsics_from_msg,
        QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=1))

    # Subscribe to semantic labels topic separately (not synchronized)
    if self.load_semseg:
      self.semantic_labels_sub = self._rosnode.create_subscription(
        String, semantic_labels_topic, self._parse_semantic_labels,
        QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=10))

    self._subs = OrderedDict()
    # Only include topics that are not None
    topic_keys = list(msg_str_to_type.keys())
    for i, t in enumerate(self._topics):
      if t is not None and i < len(topic_keys):
        msg_str = topic_keys[i]
        if msg_str != 'semlabels':  # Handle semantic labels separately
          self._subs[msg_str] = message_filters.Subscriber(
            self._rosnode, msg_str_to_type[msg_str], t, qos_profile=10)
    
    self._frame_msgs_queue = queue.Queue()

    # Create time synchronizer for synchronized topics (excluding semantic labels)
    if len(self._subs) > 1:
      self._time_sync = message_filters.ApproximateTimeSynchronizer(
        list(self._subs.values()), queue_size=10, slop=0.01,
        allow_headerless=False)
      self._time_sync.registerCallback(self._buffer_frame_msgs)
    elif len(self._subs) == 1:
      # Single topic case
      list(self._subs.values())[0].registerCallback(self._buffer_frame_msgs)

    self._ros_executor = SingleThreadedExecutor()
    self._ros_executor.add_node(self._rosnode)
    self._spin_thread = threading.Thread(
      target=self._spin_ros, name="rayfronts_semseg_input_stream_spinner")
    self._spin_thread.daemon = True

    # Wait for intrinsics and semantic labels
    self._spin_thread.start()
    
    if intrinsics_topic is not None:
      self._intrinsics_loaded_cond.acquire()
      logger.info("Waiting for intrinsics to be published..")
      while True:
        try:
          r = self._intrinsics_loaded_cond.wait(2)
        except KeyboardInterrupt as e:
          self.shutdown()
          raise e
        if r:
          self._rosnode.destroy_subscription(self.intrinsics_sub)
          break
      self._intrinsics_loaded_cond.release()
    
    if self.load_semseg:
      self._semantic_labels_cond.acquire()
      logger.info("Waiting for semantic labels to be published..")
      while True:
        try:
          r = self._semantic_labels_cond.wait(2)
        except KeyboardInterrupt as e:
          self.shutdown()
          raise e
        if r:
          break
      self._semantic_labels_cond.release()

    logger.info("Ros2SemSegSubscriber initialized successfully.")

  def _parse_semantic_labels(self, msg):
    """Parse semantic labels from JSON string message."""
    try:
      labels_data = json.loads(msg.data)
      
      # Parse the JSON structure: {"0":{"class":"BACKGROUND"},"1":{"class":"UNLABELLED"},...}
      self._cat_id_to_name = {}
      
      for key, value in labels_data.items():
        if key == "time_stamp":
          continue
          
        class_id = int(key)
        
        if isinstance(value, dict) and "class" in value:
          # Format: {"class": "BACKGROUND"}
          class_name = value["class"]
        elif isinstance(value, str):
          # Format: "beam:beam" or just "beam"
          if ":" in value:
            class_name = value.split(":")[0]
          else:
            class_name = value
        else:
          logger.warning(f"Unknown semantic label format for ID {class_id}: {value}")
          continue

        # Remap UNLABELLED (id 1) to wall
        if class_id == 1:
          class_name = "wall"
        # Rename soil class for clarity
        if class_name.lower() == "soil":
          class_name = "soil pile"
          
        if class_id not in self._cat_id_to_name:
            self._cat_id_to_name[class_id] = class_name
            # updated = True
        elif self._cat_id_to_name[class_id] != class_name:
            logger.debug(
                "Semantic label for id %d already exists as '%s'. Incoming '%s' ignored.",
                class_id, self._cat_id_to_name[class_id], class_name)
      
      # logger.info(f"Loaded {len(self._cat_id_to_name)} semantic classes: {self._cat_id_to_name}")
      
      # Notify that semantic labels are loaded
      self._semantic_labels_cond.acquire()
      self._semantic_labels_loaded = True
      self._semantic_labels_cond.notify()
      self._semantic_labels_cond.release()
      
    except json.JSONDecodeError as e:
      logger.error(f"Failed to parse semantic labels JSON: {e}")
    except Exception as e:
      logger.error(f"Error parsing semantic labels: {e}")

  def _spin_ros(self):
    try:
      self._ros_executor.spin()
    except (KeyboardInterrupt,
            rclpy.executors.ExternalShutdownException,
            rclpy.executors.ShutdownException):
      pass

  def _set_intrinsics_from_msg(self, msg):
    self._intrinsics_loaded_cond.acquire()
    self.intrinsics_3x3 = torch.tensor(msg.k, dtype=torch.float).reshape(3,3)
    self.original_h = msg.height
    self.original_w = msg.width
    self.rgb_h = self.original_h if self.rgb_h <= 0 else self.rgb_h
    self.rgb_w = self.original_w if self.rgb_w <= 0 else self.rgb_w
    self.depth_h = self.original_h if self.depth_h <= 0 else self.depth_h
    self.depth_w = self.original_w if self.depth_w <= 0 else self.depth_w

    if self.depth_h != self.original_h or self.depth_w != self.original_w:
      h_ratio = self.depth_h / self.original_h
      w_ratio = self.depth_w / self.original_w
      self.intrinsics_3x3[0, :] = self.intrinsics_3x3[0, :] * w_ratio
      self.intrinsics_3x3[1, :] = self.intrinsics_3x3[1, :] * h_ratio

    logger.info("Loaded intrinsics: \n%s", str(self.intrinsics_3x3))
    self._intrinsics_loaded_cond.notify()
    self._intrinsics_loaded_cond.release()

  def _buffer_frame_msgs(self, *msgs):
    if self.frame_skip <= 0 or self.f % (self.frame_skip+1) == 0:
      self._frame_msgs_queue.put(msgs)
    self.f += 1

  def __iter__(self):
    c = 0 
    while True:
      msgs = None
      try:
        msgs = self._frame_msgs_queue.get(block=True, timeout=2)
      except queue.Empty:
        c+=1
        logger.info(f"Queue empty, waiting for more messages... {c}")
        if c > 5:
          return None
        if not self._shutdown_event.is_set():
          continue

      if msgs is None:
        return

      # Create message dictionary from synchronized topics
      msg_dict = dict(zip(self._subs.keys(), msgs))
      
      # Add semantic segmentation if available and loaded
      if self.load_semseg and self._semantic_labels_loaded:
        # Note: semseg messages are synchronized with other topics
        if 'semseg' in msg_dict:
          semseg_msg = msg_dict['semseg']
        else:
          # If no synchronized semseg, skip this frame or create dummy
          logger.warning("No synchronized semantic segmentation message found")
          continue

      # Parse RGB
      rgba_img = image_to_numpy(msg_dict["rgb"]).astype("float") / 255
      rgb_img = rgba_img[..., :3]
      rgb_img = torch.tensor(rgb_img, dtype=torch.float).permute(2,0,1)

      # Parse Pose
      src_pose_4x4 = torch.tensor(
        pose_to_numpy(msg_dict["pose"].pose), dtype=torch.float)
      if True:
        # Static transform from base_link to camera (CameraLeft)
        translation = np.array([0.161, 0.060, 0.712], dtype=np.float32)
        rpy = [0.0, 0.1745, 0.0]  # in radians
        rotation_matrix = R.from_euler('xyz', rpy).as_matrix()

        # Construct 4x4 transform matrix in numpy
        T_base_to_cam_np = np.eye(4, dtype=np.float32)
        T_base_to_cam_np[:3, :3] = rotation_matrix
        T_base_to_cam_np[:3, 3] = translation

        # Convert to PyTorch tensor on same device and dtype as src_pose_4x4
        T_base_to_cam = torch.from_numpy(T_base_to_cam_np).to(src_pose_4x4.device, dtype=src_pose_4x4.dtype)

        # Transform base_link pose to camera pose in the map frame
        src_pose_4x4 = src_pose_4x4 @ T_base_to_cam
      rdf_pose_4x4 = g3d.transform_pose_4x4(
        src_pose_4x4, self.src2rdf_transform)

      # Parse Depth
      if "depth" in msg_dict.keys():
        depth_img = image_to_numpy(msg_dict["depth"])
        depth_img = torch.tensor(depth_img, dtype=torch.float).unsqueeze(0)
      elif "disp" in msg_dict.keys():
        # TODO: Why is disparity negative in ros2 zedx and why is max and min
        # flipped? Not sure if this is correct ros2 zedx behaviour but will
        # correct those here for now.
        disparity_img = -image_to_numpy(msg_dict["disp"].image)
        min_disp = msg_dict["disp"].max_disparity
        max_disp = msg_dict["disp"].min_disparity

        focal_length = msg_dict["disp"].f
        stereo_baseline = msg_dict["disp"].t
        depth_img = focal_length*stereo_baseline/disparity_img
        depth_img[disparity_img < min_disp] = np.inf
        depth_img[disparity_img > max_disp] = -np.inf
        depth_img = torch.tensor(depth_img, dtype=torch.float).unsqueeze(0)

      elif "pc" in msg_dict.keys():
        # TODO: This should be more efficient than a for loop
        pc_xyz = torch.tensor([[p.x, p.y, p.z] for p in msg_dict["pc"].points])
        if len(pc_xyz) == 0:
          continue
        pc_xyz_homo = g3d.pts_to_homogen(pc_xyz)
        pc_xyz_homo = g3d.transform_points_homo(pc_xyz_homo,
                                                self.src2rdf_transform)
        pc_xyz_homo_cam = pc_xyz_homo @ torch.linalg.inv(rdf_pose_4x4).T
        pc_xyz_homo_cam /= pc_xyz_homo_cam[:, -1].unsqueeze(-1)
        pc_depth = pc_xyz_homo_cam[:, 2]
        ch_n2i = {ch.name: i for i,ch in enumerate(msg_dict["pc"].channels)}
        if "kp_u" in ch_n2i and "kp_v" in ch_n2i:
          u = torch.tensor(msg_dict["pc"].channels[ch_n2i["kp_u"]].values,
                       dtype=torch.int32)
          v = torch.tensor(msg_dict["pc"].channels[ch_n2i["kp_v"]].values,
                          dtype=torch.int32)
        else:
          uv = pc_xyz_homo_cam[:, :3] @ self.intrinsics_3x3.T
          uv /= uv[:, -1]
          u = uv[:, 0]
          v = uv[:, 1]

        depth_img = torch.ones_like(rgb_img[0:1])*torch.nan
        mask = torch.logical_and(v < rgb_img.shape[1], u < rgb_img.shape[2])
        depth_img[:, v[mask], u[mask]] = pc_depth[mask]
        depth_img[depth_img < 0] = -torch.infs
      else:
        raise ValueError("Expected at least a disparity or point cloud topic")

      # Parse confidence map if it exists
      conf_img = None
      if "conf" in msg_dict:
        conf_img = image_to_numpy(msg_dict["conf"]).astype("float")
        conf_img = 1 - (torch.tensor(conf_img, dtype=torch.float) / 100)
        conf_img = conf_img.unsqueeze(0)

      # Parse semantic segmentation
      semseg_img = None
      if self.load_semseg and 'semseg' in msg_dict:
        # Convert segmentation image to tensor
        semseg_np = image_to_numpy(semseg_msg)
        semseg_img = torch.tensor(semseg_np, dtype=torch.long).unsqueeze(0)

      # Resize images if needed
      if (self.rgb_h != rgb_img.shape[-2] or
          self.rgb_w != rgb_img.shape[-1]):
        rgb_img = torch.nn.functional.interpolate(rgb_img.unsqueeze(0),
          size=(self.rgb_h, self.rgb_w), mode=self.interp_mode,
          antialias=self.interp_mode in ["bilinear", "bicubic"]).squeeze(0)

      if (self.depth_h != depth_img.shape[-2] or
          self.depth_w != depth_img.shape[-1]):
        depth_img = torch.nn.functional.interpolate(depth_img.unsqueeze(0),
          size=(self.depth_h, self.depth_w),
          mode="nearest-exact").squeeze(0)
        if conf_img is not None:
          conf_img = torch.nn.functional.interpolate(conf_img.unsqueeze(0),
            size=(self.depth_h, self.depth_w),
            mode="nearest-exact").squeeze(0)

      # Resize semantic segmentation if needed
      if semseg_img is not None:
        if (self.rgb_h != semseg_img.shape[-2] or
            self.rgb_w != semseg_img.shape[-1]):
          semseg_img = torch.nn.functional.interpolate(
            semseg_img.unsqueeze(0).float(),
            size=(self.rgb_h, self.rgb_w),
            mode="nearest-exact").squeeze(0).long()

      if torch.sum(~depth_img.isnan()) == 0:
        logger.warning("Ignoring received depth frame with no valid values")
        continue
        
      frame_data = dict(rgb_img=rgb_img, depth_img=depth_img,
                        pose_4x4=rdf_pose_4x4)

      if conf_img is not None:
        frame_data["confidence_map"] = conf_img
        
      if semseg_img is not None:
        frame_data["semseg_img"] = semseg_img

      yield frame_data

  def shutdown(self):
    self._shutdown_event.set()
    self._rosnode.context.try_shutdown()
    logger.info("Ros2SemSegSubscriber shutdown.")

  @property
  def num_classes(self):
    """Number of semantic classes."""
    return len(self._cat_id_to_name)

  @property
  def cat_id_to_name(self):
    """Returns a mapping from class id to class name.
    
    cat_id_to_name[cat_id] gives the name of that class id.
    """
    return self._cat_id_to_name

  @property
  def scene_name(self):
    """Returns the scene name for compatibility with evaluation scripts."""
    return "isaac_sim_robot"