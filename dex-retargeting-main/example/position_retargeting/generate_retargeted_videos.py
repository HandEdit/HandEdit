import argparse
import shutil
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
import yaml
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import DexYCBVideoDataset
from hand_robot_viewer import RobotHandDatasetSAPIENViewer
from dex_retargeting.constants import RobotName, HandType
from dex_retargeting.retargeting_config import RetargetingConfig

def main():
    parser = argparse.ArgumentParser(description="Generate retargeted videos from DexYCB")
    parser.add_argument("--dexycb-dir", type=str, required=True, help="Path to DexYCB dataset root")
    parser.add_argument("--output", type=str, default="./result",  required=True, help="Path to save retargeted videos")
    parser.add_argument("--robots", nargs='+', default=["inspire"], help="List of robot names")
    parser.add_argument("--hand-type", type=str, default="right", choices=["right", "left"])
    parser.add_argument("--subjects", nargs='+', help="List of subjects to process (optional)")
    
    args = parser.parse_args()

    data_root = Path(args.dexycb_dir)
    output_root = Path(args.output)
    
    # Initialize dataset
    print(f"Loading dataset from {data_root}...")
    dataset = DexYCBVideoDataset(data_root, hand_type=args.hand_type, subjects=args.subjects)
    print(f"Found {len(dataset)} captures.")
    if len(dataset) == 0:
        raise ValueError("No captures match the selected subjects and hand side")

    # Initialize Viewer (Headless)
    # RobotName keys are lowercase (e.g. 'inspire', 'allegro')
    robot_name_enums = [RobotName[name.lower()] for name in args.robots]
    hand_type_enum = HandType[args.hand_type.lower()]
    
    print(f"Processing with robots: {args.robots}")

    # Iterate and process
    for i in tqdm(range(len(dataset))):
        viewer = None
        capture_name = str(i)
        try:
            data = dataset[i]
            capture_name = data['capture_name']
            subject_name = data['subject_name']
            
            print(f"\nProcessing {subject_name}/{capture_name} ({i+1}/{len(dataset)})...")
            
            # Construct output directory
            save_dir = output_root / subject_name / capture_name
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize Viewer for EACH video to prevent memory leaks
            # Re-creating the scene ensures all resources are freed by the OS/GC
            viewer = RobotHandDatasetSAPIENViewer(
                robot_names=robot_name_enums,
                hand_type=hand_type_enum,
                headless=True,
                use_ray_tracing=False
            )

            # Load objects and hand
            viewer.load_object_hand(data)
            
            # Render and save video
            viewer.render_dexycb_data(data, fps=30, data_root=data_root, output_dir=str(save_dir))
            
            # Cleanup
            viewer.clear_all()
            
        except Exception as e:
            raise RuntimeError(f"Failed to render capture {capture_name}") from e
        finally:
            # Force destruction of the viewer and garbage collection
            if viewer is not None:
                viewer.scene = None
                del viewer
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
    print("Done.")

if __name__ == "__main__":
    main()
