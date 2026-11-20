"""Prepare a right Ability Hand clip from one DexYCB camera view."""

import argparse
import shutil
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
from frames import new_output

# DexYCB annotation IDs are 1-based and follow this subset of the YCB models.
YCB_MODELS = (
    "002_master_chef_can", "003_cracker_box", "004_sugar_box", "005_tomato_soup_can",
    "006_mustard_bottle", "007_tuna_fish_can", "008_pudding_box", "009_gelatin_box",
    "010_potted_meat_can", "011_banana", "019_pitcher_base", "021_bleach_cleanser",
    "024_bowl", "025_mug", "035_power_drill", "036_wood_block", "037_scissors",
    "040_large_marker", "051_large_clamp", "052_extra_large_clamp", "061_foam_brick",
)


def read_pose(view, t, object_count):
    with np.load(view / f"labels_{t:06d}.npz", allow_pickle=False) as labels:
        joints = labels["joint_3d"][0]
        mano_pose = labels["pose_m"][0]
        objects = labels["pose_y"]
    if (joints.shape != (21, 3) or not np.isfinite(joints).all() or np.all(joints == -1)
            or mano_pose.shape != (51,) or not np.isfinite(mano_pose).all() or not np.any(mano_pose)):
        raise ValueError(f"No valid right-hand annotation in frame {t}")
    if objects.shape != (object_count, 3, 4) or not np.isfinite(objects).all():
        raise ValueError(f"Object poses do not match the capture metadata in frame {t}")
    return joints, mano_pose[:3], objects


def make_camera(scene, size, intrinsics):
    import sapien

    camera = scene.add_camera("rgb", *size, fovy=1.0, near=0.01, far=10.0)
    camera.set_perspective_parameters(0.01, 10.0, intrinsics["fx"], intrinsics["fy"],
                                      intrinsics["ppx"], intrinsics["ppy"], 0.0)
    # SAPIEN camera axes (forward, left, up) expressed in DexYCB's (right, down, forward).
    camera.set_local_pose(sapien.Pose(q=[0.5, 0.5, -0.5, 0.5]))
    return camera


def render_clip(view, frames, models, object_ids, hands, size, intrinsics, output):
    import sapien
    from dex_retargeting import yourdfpy
    from dex_retargeting.constants import (
        HandType,
        RetargetingType,
        RobotName,
        get_default_config_path,
    )
    from dex_retargeting.retargeting_config import RetargetingConfig
    from pytransform3d.rotations import quaternion_from_compact_axis_angle, quaternion_from_matrix

    scene = sapien.Scene()
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, 1], [1, 1, 1])
    camera = make_camera(scene, size, intrinsics)
    objects = []
    for object_id in object_ids:
        if not 1 <= object_id <= len(YCB_MODELS):
            raise ValueError(f"Unknown DexYCB object ID: {object_id}")
        name = YCB_MODELS[object_id - 1]
        mesh = models / name / "textured_simple.obj"
        if not mesh.is_file():
            raise FileNotFoundError(mesh)
        builder = scene.create_actor_builder()
        builder.add_visual_from_file(str(mesh))
        objects.append(builder.build_kinematic(name=name))

    side = HandType.right
    RetargetingConfig.set_default_urdf_dir(hands.resolve())
    config_path = get_default_config_path(RobotName.ability, RetargetingType.position, side)
    config = RetargetingConfig.load_from_file(config_path)
    solver = config.build()
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    # The renderer and solver must both include the six movable wrist joints.
    urdf = yourdfpy.URDF.load(config.urdf_path, add_dummy_free_joints=True, build_scene_graph=False)
    with TemporaryDirectory() as temp:
        model = Path(temp) / "ability.urdf"
        urdf.write_xml_file(str(model))
        robot = loader.load(str(model))

    order = [solver.joint_names.index(joint.name) for joint in robot.get_active_joints()]
    fingertips = solver.optimizer.target_link_human_indices
    output.mkdir()
    for t in frames:
        joints, wrist, poses = read_pose(view, t, len(objects))
        if t == frames.start:
            solver.warm_start(joints[0], quaternion_from_compact_axis_angle(wrist),
                              hand_type=side, is_mano_convention=True)
        robot.set_qpos(solver.retarget(joints[fingertips])[order])
        for actor, pose in zip(objects, poses):
            actor.set_pose(sapien.Pose(p=pose[:, 3], q=quaternion_from_matrix(pose[:, :3])))
        scene.update_render()
        camera.take_picture()
        color = np.clip(camera.get_picture("Color")[..., :3], 0, 1)
        Image.fromarray((255 * color).astype(np.uint8)).save(output / f"{t:06d}.png")


def prepare_clip(view, first, last, hands, output):
    if not 0 <= first < last:
        raise ValueError("Select at least two frames with 0 <= FIRST < LAST")
    view = view.resolve()
    capture = view.parent
    dataset = capture.parent.parent
    meta = yaml.safe_load((capture / "meta.yml").read_text(encoding="utf-8"))
    if meta["mano_sides"] != ["right"]:
        raise ValueError("Select a single right-hand capture for the Ability Hand example")
    frames = range(first, last + 1)

    with new_output(output) as clip:
        rgb, renders = clip / "rgb", clip / "rendered"
        rgb.mkdir()
        for t in frames:
            shutil.copyfile(view / f"color_{t:06d}.jpg", rgb / f"{t:06d}.jpg")
        with Image.open(rgb / f"{first:06d}.jpg") as image:
            width, height = image.size
        cal = dataset / "calibration" / "intrinsics" / f"{view.name}_{width}x{height}.yml"
        intrinsics = yaml.safe_load(cal.read_text(encoding="utf-8"))["color"]
        render_clip(view, frames, dataset / "models", meta["ycb_ids"], hands,
                    (width, height), intrinsics, renders)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", type=Path, required=True, help="DexYCB camera directory")
    parser.add_argument("--frames", type=int, nargs=2, metavar=("FIRST", "LAST"), required=True,
                        help="Inclusive range with a visible, annotated right hand")
    parser.add_argument("--hands", type=Path,
                        default=Path(__file__).resolve().parents[2] / "assets" / "robots" / "hands")
    parser.add_argument("--output", type=Path, required=True, help="New clip directory")
    args = parser.parse_args()
    prepare_clip(args.view, *args.frames, args.hands, args.output)


if __name__ == "__main__":
    main()
