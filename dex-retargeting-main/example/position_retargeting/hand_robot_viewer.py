import tempfile
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import sapien
from hand_viewer import HandDatasetSAPIENViewer, look_at
from pytransform3d import rotations, transformations as pt
from tqdm import trange

from dex_retargeting import yourdfpy as urdf
from dex_retargeting.constants import (
    HandType,
    RetargetingType,
    RobotName,
    get_default_config_path,
)
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.seq_retarget import SeqRetargeting
from mano_layer import MANOLayer


class RobotHandDatasetSAPIENViewer(HandDatasetSAPIENViewer):
    def __init__(
        self,
        robot_names: List[RobotName],
        hand_type: HandType,
        headless=False,
        use_ray_tracing=False,
    ):
        super().__init__(headless=headless, use_ray_tracing=use_ray_tracing)

        self.robot_names = robot_names
        self.robots: List[sapien.Articulation] = []
        self.robot_file_names: List[str] = []
        self.retargetings: List[SeqRetargeting] = []
        self.retarget2sapien: List[np.ndarray] = []
        self.hand_type = hand_type

        # Set default URDF dir
        assets_path = Path(__file__).absolute().parent.parent.parent / "assets" / "robots" / "hands"
        RetargetingConfig.set_default_urdf_dir(assets_path)

        # Load optimizer and filter
        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = True
        loader.load_multiple_collisions_from_file = True
        for robot_name in robot_names:
            config_path = get_default_config_path(
                robot_name, RetargetingType.position, hand_type
            )

            # Add 6-DoF dummy joint at the root of each robot to make them move freely in the space
            # 添加6个自由度的虚拟关节，使机器人能在空间中自由移动
            override = dict(add_dummy_free_joint=True)
            config = RetargetingConfig.load_from_file(config_path, override=override)
            retargeting = config.build()
            robot_file_name = Path(config.urdf_path).stem
            self.robot_file_names.append(robot_file_name)
            self.retargetings.append(retargeting)

            # Build robot
            urdf_path = Path(config.urdf_path)
            if "glb" not in urdf_path.stem:
                urdf_path = urdf_path.with_stem(urdf_path.stem + "_glb")
            robot_urdf = urdf.URDF.load(
                str(urdf_path), add_dummy_free_joints=True, build_scene_graph=False
            )
            urdf_name = urdf_path.name
            temp_dir = tempfile.mkdtemp(prefix="dex_retargeting-")
            temp_path = f"{temp_dir}/{urdf_name}"
            robot_urdf.write_xml_file(temp_path)

            robot = loader.load(temp_path)
            self.robots.append(robot)
            sapien_joint_names = [joint.name for joint in robot.get_active_joints()]
            retarget2sapien = np.array(
                [retargeting.joint_names.index(n) for n in sapien_joint_names]
            ).astype(int)
            self.retarget2sapien.append(retarget2sapien)

    def load_object_hand(self, data: Dict):
        super().load_object_hand(data)
        ycb_ids = data["ycb_ids"]
        ycb_mesh_files = data["object_mesh_file"]

        # Load the same YCB objects for n times, n is the number of robots
        # So that for each robot, there will be an identical set of objects
        # 为每个机器人加载相同的 YCB 物体，确保每个机器人都有对应的物体集合
        for _ in range(len(self.robots)):
            for ycb_id, ycb_mesh_file in zip(ycb_ids, ycb_mesh_files):
                self._load_ycb_object(ycb_id, ycb_mesh_file)

    def render_dexycb_data(self, data: Dict, fps=5, y_offset=0, data_root=None, output_dir=None):
        # Set table and viewer pose for better visual effect only
        global_y_offset = -y_offset * len(self.robots) / 2
        self.table.set_pose(sapien.Pose([0.5, global_y_offset + 0.2, 0]))
        if not self.headless:
            self.viewer.set_camera_xyz(1.5, global_y_offset, 1)
        else:
            local_pose = self.camera.get_local_pose()
            local_pose.set_p(np.array([1.5, global_y_offset, 1]))
            self.camera.set_local_pose(local_pose)

        hand_pose = data["hand_pose"]
        object_pose = data["object_pose"]
        num_frame = hand_pose.shape[0]
        num_copy = len(self.robots) + 1
        num_ycb_objects = len(data["ycb_ids"])
        pose_offsets = []

        # --- Extrinsics Loading Logic ---
        import yaml
        all_extrinsics = data.get("all_extrinsics", None)
        
        # 优先使用传入的 data_root (如果 data 字典里有的话)
        # 注意：通常 data 字典只包含数据集数据，不含命令行参数
        # 假设：如果推断失败，尝试从 object_mesh_file 的绝对路径反推
        
        # Try to infer data_root for intrinsics loading
        if data_root is None and "object_mesh_file" in data:
            try:
                mesh_path = Path(data["object_mesh_file"][0])
                # 向上遍历直到找到 'models' 目录以确定 data_root
                for p in mesh_path.parents:
                    if p.name == "models":
                        data_root = p.parent
                        break
                
                # 如果未找到 'models'，尝试备用推断逻辑
                if data_root is None:
                     # 假设 mesh_path 为绝对路径且符合特定结构 (.../data/models/...)
                     # 我们可以尝试直接找 'data' 目录
                     for p in mesh_path.parents:
                        if p.name == "data": # 这是一个常见的命名
                            data_root = p
                            break
            except Exception:
                pass
        
        # Fallback: Try to load all_extrinsics if missing
        if all_extrinsics is None and "capture_name" in data and data_root:
            try:
                capture_name = data["capture_name"]
                
                if data_root:
                    capture_dir = None
                    
                    # 1. Direct path
                    p1 = data_root / capture_name
                    # 2. Nested path
                    p2 = data_root / capture_name / capture_name
                    
                    if (p1 / "meta.yml").exists():
                        capture_dir = p1
                    elif (p2 / "meta.yml").exists():
                        capture_dir = p2
                    else:
                        for child in data_root.iterdir():
                            if child.is_dir():
                                p3 = child / capture_name
                                if (p3 / "meta.yml").exists():
                                    capture_dir = p3
                                    break
                    
                    if capture_dir:
                        meta_path = capture_dir / "meta.yml"
                        with open(meta_path, 'r') as f:
                            meta = yaml.load(f, Loader=yaml.FullLoader)
                        ext_name = meta["extrinsics"]
                        
                        calib_dir = data_root / "calibration"
                        ext_file = calib_dir / f"extrinsics_{ext_name}" / "extrinsics.yml"
                        
                        if ext_file.exists():
                            with open(ext_file, 'r') as f:
                                ext_data = yaml.load(f, Loader=yaml.FullLoader)
                            all_extrinsics = ext_data["extrinsics"]
            except Exception:
                pass
        # -------------------------------------------------------------

        for i in range(len(self.robots) + 1):
            pose = sapien.Pose([0, -y_offset * i, 0])
            pose_offsets.append(pose)
            if i >= 1:
                self.robots[i - 1].set_pose(pose)

        # Skip frames where human hand is not detected in DexYCB dataset
        start_frame = 0
        for i in range(0, num_frame):
            init_hand_pose_frame = hand_pose[i]
            vertex, joint = self._compute_hand_geometry(init_hand_pose_frame)
            if vertex is not None:
                start_frame = i
                break

        writers = {}
        image_dirs = {}
        if self.headless:
            robot_names = [robot.name for robot in self.robot_names]
            robot_names_str = "_".join(robot_names)
            
            print("Initializing Output Directories:")
            if all_extrinsics is not None:
                for cam_name in all_extrinsics:
                    if cam_name == 'apriltag': continue
                    
                    if output_dir:
                        # Use provided output directory
                        out_path = Path(output_dir)
                        out_path.mkdir(parents=True, exist_ok=True)
                        
                        # Create image directory inside output_dir using camera name (e.g. 836212060125)
                        # This matches the original DexYCB structure
                        img_dir = out_path / cam_name
                        img_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Save video inside the camera folder
                        video_path = img_dir / f"{robot_names_str}_retargeting.mp4"
                    else:
                        # Default path
                        video_path = Path(__file__).parent.resolve() / f"data/{robot_names_str}_video_{cam_name}.mp4"
                        folder_name = f"{robot_names_str}_video_{cam_name}"
                        img_dir = Path(__file__).parent.resolve() / f"data/{folder_name}"
                        img_dir.mkdir(parents=True, exist_ok=True)

                    writers[cam_name] = cv2.VideoWriter(
                        str(video_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        30.0,
                        (self.camera.get_width(), self.camera.get_height()),
                    )
                    
                    image_dirs[cam_name] = img_dir
                    print(f"  [Camera {cam_name}] Output: {img_dir}")
            else:
                if output_dir:
                    out_path = Path(output_dir)
                    out_path.mkdir(parents=True, exist_ok=True)
                    video_path = out_path / f"{robot_names_str}_video.mp4"
                    img_dir = out_path / f"{robot_names_str}_video"
                else:
                    video_path = (
                        Path(__file__).parent.resolve() / f"data/{robot_names_str}_video.mp4"
                    )
                    img_dir = Path(__file__).parent.resolve() / f"data/{robot_names_str}_video"

                writers["default"] = cv2.VideoWriter(
                    str(video_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    30.0,
                    (self.camera.get_width(), self.camera.get_height()),
                )
                img_dir.mkdir(parents=True, exist_ok=True)
                image_dirs["default"] = img_dir
                print(f"  [Default Camera] Video: {video_path.name} | Images: {img_dir}")

        # Warm start
        hand_pose_start = hand_pose[start_frame]
        wrist_quat = rotations.quaternion_from_compact_axis_angle(
            hand_pose_start[0, 0:3]
        )
        vertex, joint = self._compute_hand_geometry(hand_pose_start)
        for robot, retargeting, retarget2sapien in zip(
            self.robots, self.retargetings, self.retarget2sapien
        ):
            retargeting.warm_start(
                joint[0, :],
                wrist_quat,
                hand_type=self.hand_type,
                is_mano_convention=True,
            )

        # Loop rendering
        step_per_frame = int(60 / fps)
        
        for i in trange(start_frame, num_frame):
            # Force debug print on first iteration of the loop
            is_first_render_pass = (i == start_frame)
            
            object_pose_frame = object_pose[i]
            hand_pose_frame = hand_pose[i]
            vertex, joint = self._compute_hand_geometry(hand_pose_frame)

            # Update poses for YCB objects
            for k in range(num_ycb_objects):
                pos_quat = object_pose_frame[k]

                # Quaternion convention: xyzw -> wxyz
                quat_wxyz = np.concatenate([pos_quat[3:4], pos_quat[:3]])
                obj_pose_camera = sapien.Pose(pos_quat[4:], quat_wxyz)
                
                if hasattr(self, 'opencv_camera_pose') and self.opencv_camera_pose is not None:
                    pose = self.opencv_camera_pose * obj_pose_camera
                else:
                    pose = self.camera_pose * obj_pose_camera
                
                if y_offset == 0:
                    # If no offset, only render one set of objects to avoid z-fighting
                    self.objects[k].set_pose(pose)
                    for copy_ind in range(1, num_copy):
                        # Move other copies out of view
                        self.objects[k + copy_ind * num_ycb_objects].set_pose(sapien.Pose([0, 0, 100]))
                else:
                    self.objects[k].set_pose(pose)
                    for copy_ind in range(num_copy):
                        self.objects[k + copy_ind * num_ycb_objects].set_pose(
                            pose_offsets[copy_ind] * pose
                        )

            # Update pose for human hand
            # self._update_hand(vertex)

            # Update poses for robot hands
            for robot, retargeting, retarget2sapien in zip(
                self.robots, self.retargetings, self.retarget2sapien
            ):
                indices = retargeting.optimizer.target_link_human_indices
                ref_value = joint[indices, :]
                qpos = retargeting.retarget(ref_value)[retarget2sapien]
                robot.set_qpos(qpos)

            self.scene.update_render()
            if self.headless:
                if writers:
                    if all_extrinsics is not None:
                        for cam_name, writer in writers.items():
                            # Calculate pose
                            rel_ext_tuple = all_extrinsics[cam_name]
                            rel_ext_mat = np.array(rel_ext_tuple).reshape(3, 4)
                            rel_ext_mat = np.concatenate([rel_ext_mat, np.array([[0, 0, 0, 1]])], axis=0)
                            base_ext_mat = data["extrinsics"]
                            
                            # Invert relative extrinsics as per user request
                            final_ext_mat = np.linalg.inv(rel_ext_mat) @ base_ext_mat
                            
                            camera_in_apriltag = np.linalg.inv(final_ext_mat)
                            rot_mat = camera_in_apriltag[:3, :3]
                            eye = camera_in_apriltag[:3, 3]
                            forward_world = rot_mat @ np.array([0, 0, 1])
                            up_world = rot_mat @ np.array([0, -1, 0])
                            
                            sapien_pose = look_at(eye, eye + forward_world, up_world)
                            
                            # --- Dynamic Intrinsics Loading ---
                            if data_root:
                                try:
                                    # Path: data_root/calibration/intrinsics/{cam_name}_640x480.yml
                                    int_file = data_root / "calibration" / "intrinsics" / f"{cam_name}_640x480.yml"
                                    
                                    if int_file.exists():
                                        with open(int_file, 'r') as f:
                                            int_data = yaml.load(f, Loader=yaml.FullLoader)
                                        
                                        # Scale 640x480 -> 1920x1440 (Scale = 3.0)
                                        scale = 3.0
                                        fx = int_data['color']['fx'] * scale
                                        fy = int_data['color']['fy'] * scale
                                        cx = int_data['color']['ppx'] * scale
                                        cy = int_data['color']['ppy'] * scale
                                        
                                        # Apply perspective projection
                                        self.camera.set_perspective_parameters(0.01, 100, fx, fy, cx, cy, 0.0)
                                except Exception:
                                    pass
                            
                            self.camera.set_local_pose(sapien_pose)
                            self.scene.update_render()
                            self.camera.take_picture()
                            rgb = self.camera.get_picture("Color")[..., :3]
                            rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
                            bgr = rgb[..., ::-1]
                            writer.write(bgr)
                            
                            if cam_name in image_dirs:
                                img_path = image_dirs[cam_name] / f"{i:06d}.png"
                                cv2.imwrite(str(img_path), bgr)
                    else:
                        self.camera.take_picture()
                        rgb = self.camera.get_picture("Color")[..., :3]
                        rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
                        writers["default"].write(rgb[..., ::-1])
                        if "default" in image_dirs:
                            img_path = image_dirs["default"] / f"{i:06d}.png"
                            cv2.imwrite(str(img_path), rgb[..., ::-1])
            else:
                for _ in range(step_per_frame):
                    self.viewer.render()

        if not self.headless:
            self.viewer.paused = True
            self.viewer.render()
        else:
            for writer in writers.values():
                writer.release()

