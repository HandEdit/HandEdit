from pathlib import Path
from typing import Optional, List
import os
import numpy as np
import tyro

from dataset import DexYCBVideoDataset
from dex_retargeting.constants import RobotName, HandType
from dex_retargeting.retargeting_config import RetargetingConfig
from hand_robot_viewer import RobotHandDatasetSAPIENViewer
from hand_viewer import HandDatasetSAPIENViewer

# For numpy version compatibility
np.bool = bool
np.int = int
np.float = float
np.str = str
np.complex = complex
np.object = object
np.unicode = np.str_

def ensure_output_dir_exists():
    """Ensures the output directory for videos exists."""
    output_dir = Path(__file__).parent.resolve() / "data"
    if not output_dir.exists():
        print(f"Creating output directory for videos at: {output_dir}")
        os.makedirs(output_dir)

def viz_hand_object(
    robots: Optional[List[RobotName]],
    data_root: Path,
    fps: int,
    data_id: int,
    headless: bool,
):
    """
    Initializes the viewer and renders the specified hand-object interaction.

    Args:
        robots: A list of robot names to render. If None, renders the human hand.
        data_root: The root directory of the DexYCB dataset.
        fps: The frames per second for rendering the trajectory.
        data_id: The ID of the data sample to visualize from the dataset.
        headless: If True, run in headless mode and save output as a video.
    """
    if headless:
        ensure_output_dir_exists()
        print("Running in headless mode. The output will be saved as a video.")
    else:
        print("Running in interactive mode. A viewer window will open.")

    dataset = DexYCBVideoDataset(data_root, hand_type="right")
    
    if data_id >= len(dataset):
        print(f"Error: data_id {data_id} is out of range. The dataset has {len(dataset)} samples.")
        print(f"Using data_id = 0 instead.")
        data_id = 0

    if robots is None:
        viewer = HandDatasetSAPIENViewer(headless=headless)
    else:
        # The viewer expects a list, not a tuple, for robots.
        viewer = RobotHandDatasetSAPIENViewer(
            list(robots), HandType.right, headless=headless
        )

    print(f"Loading data sample with ID: {data_id}")
    sampled_data = dataset[data_id]
    print("Data sample loaded successfully. Details:")
    for key, value in sampled_data.items():
        if "pose" not in key:
            # For lists/arrays, print their shape instead of all values
            if isinstance(value, (list, np.ndarray)):
                shape = np.array(value).shape
                print(f"- {key}: shape{shape}")
            else:
                print(f"- {key}: {value}")

    viewer.load_object_hand(sampled_data)

    viewer.render_dexycb_data(sampled_data, fps)

    if headless:
        print("\nVideo rendering complete!")
        # The detailed output paths are now printed within the render_dexycb_data method.
        print("Check the 'data' directory for output videos and frame images.")


def main(
    dexycb_dir: str,
    robots: Optional[List[RobotName]] = None,
    fps: int = 10,
    data_id: int = 30,
    headless: bool = False,
):

    data_root = Path(dexycb_dir).absolute()
    robot_dir = (
        Path(__file__).absolute().parent.parent.parent / "assets" / "robots" / "hands"
    )
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    if not data_root.exists():
        raise ValueError(f"Path to DexYCB dir: {data_root} does not exist.")
    else:
        print(f"Using DexYCB dir: {data_root}")
    
    viz_hand_object(robots, data_root, fps, data_id, headless)


if __name__ == "__main__":
    tyro.cli(main)
