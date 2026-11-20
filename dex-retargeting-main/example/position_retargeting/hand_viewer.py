from pathlib import Path
from typing import Dict, List, Optional

import cv2
from tqdm import trange
import numpy as np
import sapien
import torch
import yaml
# from pytransform3d import transformations as pt
from scipy.spatial.transform import Rotation as R
from sapien import internal_renderer as R_internal
from sapien.asset import create_dome_envmap
from sapien.utils import Viewer

from dataset import YCB_CLASSES
from mano_layer import MANOLayer


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> sapien.Pose:
    """
    实现一个 look_at 函数来对齐相机。
    SAPIEN 的相机坐标系是：X轴朝前，Z轴朝上，Y轴朝右（右手系）。
    """
    f = target - eye
    f /= np.linalg.norm(f)
    
    # 计算右向量
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    
    # 计算真正的上向量（保证垂直）
    u = np.cross(r, f)
    
    # 拼成旋转矩阵 [Forward, -Right, Up] -> [X, -Y, Z]
    # 注意这里 SAPIEN 的 Y 是 -Right
    rot_mat = np.column_stack([f, -r, u])
    
    # 转成四元数塞给 SAPIEN
    quat_scipy = R.from_matrix(rot_mat).as_quat() # [x, y, z, w]
    quat_sapien = np.array([quat_scipy[3], quat_scipy[0], quat_scipy[1], quat_scipy[2]])
    
    return sapien.Pose(eye, quat_sapien)


def compute_smooth_shading_normal_np(vertices, indices):
    """
    计算顶点法线以实现平滑着色。
    """
    v1 = vertices[indices[:, 0]]
    v2 = vertices[indices[:, 1]]
    v3 = vertices[indices[:, 2]]
    face_normal = np.cross(v2 - v1, v3 - v1)

    vertex_normal = np.zeros_like(vertices)
    vertex_normal[indices[:, 0]] += face_normal
    vertex_normal[indices[:, 1]] += face_normal
    vertex_normal[indices[:, 2]] += face_normal
    vertex_normal /= np.linalg.norm(vertex_normal, axis=1, keepdims=True)
    return vertex_normal


class HandDatasetSAPIENViewer:
    def __init__(self, headless=False, use_ray_tracing=False):
        # 配置渲染设置
        if not use_ray_tracing:
            sapien.render.set_viewer_shader_dir("default")
            sapien.render.set_camera_shader_dir("default")
        else:
            sapien.render.set_viewer_shader_dir("rt")
            sapien.render.set_camera_shader_dir("rt")
            sapien.render.set_ray_tracing_samples_per_pixel(64)
            sapien.render.set_ray_tracing_path_depth(8)
            sapien.render.set_ray_tracing_denoiser("oidn")

        # 创建场景
        scene = sapien.Scene()
        scene.set_timestep(1 / 240)

        # 设置光照
        scene.set_environment_map(
            create_dome_envmap(sky_color=[0.2, 0.2, 0.2], ground_color=[0.2, 0.2, 0.2])
        )
        scene.add_directional_light(
            np.array([1, -1, -1]), np.array([2, 2, 2]), shadow=True
        )
        scene.add_directional_light([0, 0, -1], [1.8, 1.6, 1.6], shadow=False)
        scene.set_ambient_light(np.array([0.2, 0.2, 0.2]))

        # 添加地面
        visual_material = sapien.render.RenderMaterial()
        visual_material.set_base_color(np.array([0.5, 0.5, 0.5, 1]))
        visual_material.set_roughness(0.7)
        visual_material.set_metallic(1)
        visual_material.set_specular(0.04)
        scene.add_ground(-1, render_material=visual_material)

        # 如果非无头模式，初始化查看器窗口
        if not headless:
            viewer = Viewer()
            viewer.set_scene(scene)
            viewer.set_camera_xyz(1.5, 0, 1)
            viewer.set_camera_rpy(0, -0.8, 3.14)
            viewer.control_window.toggle_origin_frame(False)
            self.viewer = viewer
        else:
            # 使用您提供的精确内参 (840412060917_640x480.yml)
            # 原始分辨率 640x480 -> 渲染分辨率 1920x1440 (Scale = 3.0)
            scale = 3.0
            fx = 621.7969970703125 * scale
            fy = 621.4028930664062 * scale
            cx = 302.4136962890625 * scale
            cy = 236.88433837890625 * scale
            skew = 0.0
            near = 0.01
            far = 100
            
            # 先创建一个默认相机
            self.camera = scene.add_camera("cam", 1920, 1440, 1.0, near, far)
            # 然后覆盖投影矩阵，应用精确的主点偏移
            self.camera.set_perspective_parameters(near, far, fx, fy, cx, cy, skew)

        self.headless = headless

        # 创建桌子
        white_diffuse = sapien.render.RenderMaterial()
        white_diffuse.set_base_color(np.array([0.8, 0.8, 0.8, 1]))
        white_diffuse.set_roughness(0.9)
        builder = scene.create_actor_builder()
        builder.add_box_collision(
            sapien.Pose([0, 0, -0.02]), half_size=np.array([0.5, 2.0, 0.02])
        )
        builder.add_box_visual(
            sapien.Pose([0, 0, -0.02]),
            half_size=np.array([0.5, 2.0, 0.02]),
            material=white_diffuse,
        )
        # 桌腿
        builder.add_box_visual(
            sapien.Pose([0.4, 1.9, -0.51]),
            half_size=np.array([0.015, 0.015, 0.49]),
            material=white_diffuse,
        )
        builder.add_box_visual(
            sapien.Pose([-0.4, 1.9, -0.51]),
            half_size=np.array([0.015, 0.015, 0.49]),
            material=white_diffuse,
        )
        builder.add_box_visual(
            sapien.Pose([0.4, -1.9, -0.51]),
            half_size=np.array([0.015, 0.015, 0.49]),
            material=white_diffuse,
        )
        builder.add_box_visual(
            sapien.Pose([-0.4, -1.9, -0.51]),
            half_size=np.array([0.015, 0.015, 0.49]),
            material=white_diffuse,
        )
        self.table = builder.build_static(name="table")
        self.table.set_pose(sapien.Pose([0.5, 0, 0]))

        # 缓存变量以供后续使用
        sapien.render.set_log_level("error")
        self.scene = scene
        self.internal_scene: R_internal.Scene = scene.render_system._internal_scene
        self.context: R_internal.Context = sapien.render.SapienRenderer()._internal_context
        self.mat_hand = self.context.create_material(
            np.zeros(4), np.array([0.96, 0.75, 0.69, 1]), 0.0, 0.8, 0
        )

        self.mano_layer: Optional[MANOLayer] = None
        self.mano_face: Optional[np.ndarray] = None
        self.camera_pose: Optional[sapien.Pose] = None
        self.opencv_camera_pose: Optional[sapien.Pose] = None
        self.objects: List[sapien.Entity] = []
        self.nodes: List[R_internal.Node] = []

    def clear_all(self):
        while self.objects:
            actor = self.objects.pop()
            try:
                self.scene.remove_actor(actor)
            except RuntimeError:
                pass
        self.clear_node()
        self.mano_layer = None

    def clear_node(self):
        for _ in range(len(self.nodes)):
            node = self.nodes.pop()
            self.internal_scene.remove_node(node)

    def load_object_hand(self, data: Dict):
        ycb_ids = data["ycb_ids"]
        ycb_mesh_files = data["object_mesh_file"]
        hand_shape = data["hand_shape"]
        extrinsic_mat = data["extrinsics"]  # 这个矩阵是 T_{apriltag <- camera}
        
        # 加载 YCB 物体模型
        for ycb_id, ycb_mesh_file in zip(ycb_ids, ycb_mesh_files):
            self._load_ycb_object(ycb_id, ycb_mesh_file)

        # 初始化 MANO 手模型
        self.mano_layer = MANOLayer("right", hand_shape.astype(np.float32))
        self.mano_face = self.mano_layer.f.cpu().numpy()
        
        # === 相机位姿计算 ===
        # 原始数据里的外参是把相机坐标系转到 AprilTag 坐标系的矩阵
        # 我们需要反过来，求相机在 AprilTag 坐标系下的位置和姿态
        camera_in_apriltag = np.linalg.inv(extrinsic_mat)
        
        # 提取位置和旋转（OpenCV 坐标系习惯）
        position_opencv = camera_in_apriltag[:3, 3]
        quat_opencv_xyzw = R.from_matrix(camera_in_apriltag[:3, :3]).as_quat()
        # 换个顺序，SAPIEN 喜欢 w 在前 [w, x, y, z]
        quat_opencv_wxyz = np.array([quat_opencv_xyzw[3], quat_opencv_xyzw[0], quat_opencv_xyzw[1], quat_opencv_xyzw[2]])
        
        # 存一下这个 OpenCV 坐标系下的相机位姿，后面转物体坐标要用
        self.opencv_camera_pose = sapien.Pose(position_opencv, quat_opencv_wxyz)
        
        # === 坐标系转换 ===
        # OpenCV 的相机是 Z轴向前，Y轴向下
        # SAPIEN 的相机是 X轴向前，Z轴向上
        # 所以得用 look_at 重新算一下 SAPIEN 相机该怎么摆
        
        eye = camera_in_apriltag[:3, 3]
        rot_mat = camera_in_apriltag[:3, :3]
        
        # OpenCV: 前方是 +Z，上方是 -Y
        forward_world = rot_mat @ np.array([0, 0, 1])
        up_world = rot_mat @ np.array([0, -1, 0])
        
        # 算出 SAPIEN 需要的位姿
        self.camera_pose = look_at(eye, eye + forward_world, up_world)

        if self.headless:
            self.camera.set_local_pose(self.camera_pose)
    def _load_ycb_object(self, ycb_id, ycb_mesh_file):
        builder = self.scene.create_actor_builder()
        builder.add_visual_from_file(ycb_mesh_file)
        actor = builder.build_static(name=YCB_CLASSES[ycb_id])
        self.objects.append(actor)

    def _compute_hand_geometry(self, hand_pose_frame, use_camera_frame=False):
        # pose parameters all zero, no hand is detected
        if np.abs(hand_pose_frame).sum() < 1e-5:
            return None, None
        
        p = torch.from_numpy(hand_pose_frame[:, :48].astype(np.float32))
        t = torch.from_numpy(hand_pose_frame[:, 48:51].astype(np.float32))
        vertex, joint = self.mano_layer(p, t)
        vertex = vertex.cpu().numpy()[0]  # (778, 3)
        joint = joint.cpu().numpy()[0]    # (21, 3)
        
        if not use_camera_frame and self.opencv_camera_pose is not None:
            camera_mat = self.opencv_camera_pose.to_transformation_matrix()
            vertex = vertex @ camera_mat[:3, :3].T + camera_mat[:3, 3]
            vertex = np.ascontiguousarray(vertex)
            joint = joint @ camera_mat[:3, :3].T + camera_mat[:3, 3]
            joint = np.ascontiguousarray(joint)

        return vertex, joint

    def _update_hand(self, vertex):
        self.clear_node()
        normal = compute_smooth_shading_normal_np(vertex, self.mano_face)
        mesh = self.context.create_mesh_from_array(vertex, self.mano_face, normal)
        model = self.context.create_model([mesh], [self.mat_hand])
        node = self.internal_scene.add_node()
        node.set_position(np.array([0, 0, 0]))
        obj = self.internal_scene.add_object(model, node)
        obj.shading_mode = 0
        obj.cast_shadow = True
        obj.transparency = 0
        self.nodes.append(node)

    def render_dexycb_data(self, data: Dict, fps=10, data_root=None):
        """
        渲染 DexYCB 数据集里的手和物体。
        fps: 视频帧率，决定了生成的视频播放速度有多快。
        data_root: 数据集根目录，用于加载内参文件。
        """
        hand_pose = data["hand_pose"]
        object_pose = data["object_pose"]
        num_frame = hand_pose.shape[0]
        
        # --- 1. 外参加载逻辑 (Robust Extrinsics Loading) ---
        all_extrinsics = data.get("all_extrinsics", None)
        
        # 尝试推断 data_root，用于后续加载内参
        if data_root is None and "object_mesh_file" in data:
            try:
                mesh_path = Path(data["object_mesh_file"][0])
                for p in mesh_path.parents:
                    if p.name == "models":
                        data_root = p.parent
                        break
            except Exception:
                pass

        # 如果数据里没给外参，尝试手动加载
        if all_extrinsics is None and "capture_name" in data and data_root:
            try:
                print("[DEBUG] 尝试手动加载外参...")
                capture_name = data["capture_name"]
                
                if data_root:
                    print(f"[DEBUG] 推断出的数据根目录: {data_root}")
                    capture_dir = None
                    
                    # 1. 直接路径
                    p1 = data_root / capture_name
                    # 2. 嵌套路径
                    p2 = data_root / capture_name / capture_name
                    
                    if (p1 / "meta.yml").exists():
                        capture_dir = p1
                    elif (p2 / "meta.yml").exists():
                        capture_dir = p2
                    else:
                        print(f"[DEBUG] 直接路径未找到，搜索子目录...")
                        for child in data_root.iterdir():
                            if child.is_dir():
                                p3 = child / capture_name
                                if (p3 / "meta.yml").exists():
                                    capture_dir = p3
                                    break
                    
                    if capture_dir:
                        print(f"[DEBUG] 找到采集目录: {capture_dir}")
                        meta_path = capture_dir / "meta.yml"
                        with open(meta_path, 'r') as f:
                            meta = yaml.load(f, Loader=yaml.FullLoader)
                        ext_name = meta["extrinsics"]
                        
                        # 修正：calibration 只有一层
                        calib_dir = data_root / "calibration"
                        ext_file = calib_dir / f"extrinsics_{ext_name}" / "extrinsics.yml"
                        
                        if ext_file.exists():
                            with open(ext_file, 'r') as f:
                                ext_data = yaml.load(f, Loader=yaml.FullLoader)
                            all_extrinsics = ext_data["extrinsics"]
                            print(f"[DEBUG] 成功手动加载了 {len(all_extrinsics)} 个外参。")
                        else:
                            print(f"[DEBUG] 外参文件未找到: {ext_file}")
                    else:
                        print(f"[DEBUG] 在 {data_root} 下未找到 {capture_name} 的目录")
            except Exception as e:
                print(f"[DEBUG] 手动加载外参失败: {e}")
                import traceback
                traceback.print_exc()
        
        if all_extrinsics:
            cam_names = [k for k in all_extrinsics.keys() if k != 'apriltag']
            print(f"开启多视角渲染，一共 {len(cam_names)} 个相机。")

        # --- 2. 跳过没有手的帧 (Skip Empty Frames) ---
        start_frame = 0
        for i in range(0, num_frame):
            init_hand_pose_frame = hand_pose[i]
            vertex, joint = self._compute_hand_geometry(init_hand_pose_frame)
            if vertex is not None:
                start_frame = i
                break
        print(f"从第 {start_frame} 帧开始渲染 (跳过前面的空帧)。")

        # --- 3. 初始化输出目录 (Output Directories) ---
        writers = {}
        image_dirs = {}
        if self.headless:
            print("初始化输出目录:")
            if all_extrinsics is not None:
                for cam_name in all_extrinsics:
                    if cam_name == 'apriltag': continue
                    video_path = Path(__file__).parent.resolve() / f"data/video_{cam_name}.mp4"
                    writers[cam_name] = cv2.VideoWriter(
                        str(video_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        30.0,
                        (self.camera.get_width(), self.camera.get_height()),
                    )
                    
                    # 创建图片文件夹
                    folder_name = f"video_{cam_name}"
                    img_dir = Path(__file__).parent.resolve() / f"data/{folder_name}"
                    img_dir.mkdir(parents=True, exist_ok=True)
                    image_dirs[cam_name] = img_dir
                    print(f"  [相机 {cam_name}] 视频: {video_path.name} | 图片: data/{folder_name}/")
            else:
                video_path = Path(__file__).parent.resolve() / "data/human_hand_video.mp4"
                writers["default"] = cv2.VideoWriter(
                    str(video_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    30.0,
                    (self.camera.get_width(), self.camera.get_height()),
                )
                
                # 创建图片文件夹
                img_dir = Path(__file__).parent.resolve() / "data/human_hand_video"
                img_dir.mkdir(parents=True, exist_ok=True)
                image_dirs["default"] = img_dir
                print(f"  [默认相机] 视频: {video_path.name} | 图片: data/human_hand_video/")
            print("-" * 30 + "\n")

        # --- 4. 渲染循环 (Render Loop) ---
        step_per_frame = int(60 / fps)
        for i in trange(start_frame, num_frame):
            object_pose_frame = object_pose[i]
            hand_pose_frame = hand_pose[i]
            
            # 更新手部几何信息
            vertex, joint = self._compute_hand_geometry(hand_pose_frame)
            if vertex is not None:
                self._update_hand(vertex)
            
            # 更新物体位姿
            for k in range(len(self.objects)):
                pos_quat = object_pose_frame[k]
                
                quat_xyzw = pos_quat[:4]
                trans = pos_quat[4:]
                
                # 转成 SAPIEN 格式 [w, x, y, z]
                quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
                
                # 数据是在相机坐标系下的，需要转到世界坐标系
                obj_pose_camera = sapien.Pose(trans, quat_wxyz)
                
                if self.opencv_camera_pose is not None:
                    obj_pose_world = self.opencv_camera_pose * obj_pose_camera
                    self.objects[k].set_pose(obj_pose_world)
                else:
                    self.objects[k].set_pose(obj_pose_camera)
            
            self.scene.update_render()
            
            if self.headless:
                if writers:
                    if all_extrinsics is not None:
                        # 多视角渲染
                        for cam_name, writer in writers.items():
                            # 1. 获取相对外参 (Camera <- Master)
                            rel_ext_tuple = all_extrinsics[cam_name]
                            rel_ext_mat = np.array(rel_ext_tuple).reshape(3, 4)
                            rel_ext_mat = np.concatenate([rel_ext_mat, np.array([[0, 0, 0, 1]])], axis=0)
                            
                            # 2. 获取基础外参 (Master <- World/Apriltag)
                            base_ext_mat = data["extrinsics"]
                            
                            # 3. 计算最终外参 (Camera <- World)
                            # 注意：文件里的外参可能是 T_{master <- camera}
                            # 我们需要 T_{camera <- master} 来乘以 T_{master <- world}
                            # 所以这里用了逆矩阵
                            final_ext_mat = np.linalg.inv(rel_ext_mat) @ base_ext_mat
                            
                            # 4. 计算 SAPIEN 相机位姿 (Look-At 方法)
                            camera_in_apriltag = np.linalg.inv(final_ext_mat)
                            
                            rot_mat = camera_in_apriltag[:3, :3]
                            eye = camera_in_apriltag[:3, 3]
                            forward_world = rot_mat @ np.array([0, 0, 1]) # OpenCV +Z 前
                            up_world = rot_mat @ np.array([0, -1, 0])    # OpenCV -Y 上
                            
                            sapien_pose = look_at(eye, eye + forward_world, up_world)
                            
                            # --- 动态加载内参 (Dynamic Intrinsics Loading) ---
                            if data_root:
                                try:
                                    # 构造内参文件路径: data_root/calibration/intrinsics/{cam_name}_640x480.yml
                                    int_file = data_root / "calibration" / "intrinsics" / f"{cam_name}_640x480.yml"
                                    if int_file.exists():
                                        with open(int_file, 'r') as f:
                                            int_data = yaml.load(f, Loader=yaml.FullLoader)
                                        
                                        # 原始分辨率 640x480 -> 渲染分辨率 1920x1440 (Scale = 3.0)
                                        scale = 3.0
                                        fx = int_data['color']['fx'] * scale
                                        fy = int_data['color']['fy'] * scale
                                        cx = int_data['color']['ppx'] * scale
                                        cy = int_data['color']['ppy'] * scale
                                        
                                        # 应用透视投影
                                        self.camera.set_perspective_parameters(0.01, 100, fx, fy, cx, cy, 0.0)
                                        if i == 0: # 只在第一帧打印一次调试信息
                                            print(f"[DEBUG] Camera {cam_name} loaded intrinsics: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
                                except Exception as e:
                                    print(f"[WARN] Failed to load intrinsics for {cam_name}: {e}")
                            # -------------------------------------------------
                            
                            self.camera.set_local_pose(sapien_pose)
                            self.scene.update_render()
                            self.camera.take_picture()
                            rgb = self.camera.get_picture("Color")[..., :3]
                            rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
                            bgr = rgb[..., ::-1]
                            writer.write(bgr)
                            
                            # 保存图片
                            if cam_name in image_dirs:
                                img_path = image_dirs[cam_name] / f"{i:06d}.png"
                                cv2.imwrite(str(img_path), bgr)
                    else:
                        # 单视角（默认相机位姿）
                        self.camera.take_picture()
                        rgb = self.camera.get_picture("Color")[..., :3]
                        rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
                        bgr = rgb[..., ::-1]
                        writers["default"].write(bgr)
                        
                        # 保存图片
                        if "default" in image_dirs:
                            img_path = image_dirs["default"] / f"{i:06d}.png"
                            cv2.imwrite(str(img_path), bgr)
            else:
                for _ in range(step_per_frame):
                    self.viewer.render()

        if not self.headless:
            self.viewer.paused = True
            self.viewer.render()
        else:
            for writer in writers.values():
                writer.release()
