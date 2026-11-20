import os
import argparse
import torch
from tqdm import tqdm
from inference import load_models, inference_single, get_device

def main():
    parser = argparse.ArgumentParser(description='Batch Inference for ProPainter')
    parser.add_argument('--input_root', type=str, required=False, help='(Deprecated) Root directory of the input dataset. Provide either this OR both --photo_root and --mask_root')
    parser.add_argument('--photo_root', type=str, required=False, help='Root directory containing photo folders (mirrors mask_root structure)')
    parser.add_argument('--mask_root', type=str, required=False, help='Root directory containing mask folders (mirrors photo_root structure)')
    parser.add_argument('--output_root', type=str, required=True, help='Root directory for the output')
    
    parser.add_argument("--resize_ratio", type=float, default=1.0, help='Resize scale for processing video.')
    parser.add_argument('--height', type=int, default=-1, help='Height of the processing video.')
    parser.add_argument('--width', type=int, default=-1, help='Width of the processing video.')
    parser.add_argument('--mask_dilation', type=int, default=4, help='Mask dilation for video and flow masking.')
    parser.add_argument("--ref_stride", type=int, default=10, help='Stride of global reference frames.')
    parser.add_argument("--neighbor_length", type=int, default=10, help='Length of local neighboring frames.')
    parser.add_argument("--subvideo_length", type=int, default=80, help='Length of sub-video for long video inference.')
    parser.add_argument("--raft_iter", type=int, default=20, help='Iterations for RAFT inference.')
    parser.add_argument('--mode', default='video_inpainting', choices=['video_inpainting', 'video_outpainting'], help="Modes: video_inpainting / video_outpainting")
    parser.add_argument('--scale_h', type=float, default=1.0, help='Outpainting scale of height for video_outpainting mode.')
    parser.add_argument('--scale_w', type=float, default=1.2, help='Outpainting scale of width for video_outpainting mode.')
    parser.add_argument('--save_fps', type=int, default=24, help='Frame per second. Default: 24')
    parser.add_argument('--fp16', action='store_true', help='Use fp16 (half precision) during inference. Default: fp32 (single precision).')

    args = parser.parse_args()

    if bool(args.photo_root) != bool(args.mask_root):
        parser.error('--photo_root and --mask_root must be provided together')
    if not args.input_root and not args.photo_root:
        parser.error('Provide --photo_root and --mask_root, or --input_root')
    for root in (args.photo_root, args.mask_root, args.input_root):
        if root and not os.path.isdir(root):
            parser.error(f'Input directory not found: {root}')

    device = get_device()
    models = load_models(device)

    # Two modes supported:
    # 1) (New) Provide both --photo_root and --mask_root with identical directory structure; script will walk mask_root and match by relative path
    # 2) (Legacy) Provide --input_root with subfolders containing 'photo' and 'mask' like before

    if args.photo_root and args.mask_root:
        # New mode: walk mask_root and find corresponding photo directories by relative path
        print(f"Walking mask_root: {args.mask_root} and matching with photo_root: {args.photo_root}")
        processed = 0
        for dirpath, dirnames, filenames in os.walk(args.mask_root):
            # Skip dirs without image files
            img_files = [f for f in filenames if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if not img_files:
                continue

            rel = os.path.relpath(dirpath, args.mask_root)
            photo_dir = os.path.join(args.photo_root, rel)
            if not os.path.exists(photo_dir):
                raise FileNotFoundError(f"No photo folder corresponding to masks: {rel}")

            save_path = os.path.join(args.output_root, rel)
            os.makedirs(save_path, exist_ok=True)

            print(f"Processing: {rel} -> saving to {save_path}")
            inference_single(models, device, args, photo_dir, dirpath, save_path)
            processed += 1
        if processed == 0:
            raise ValueError("No mask clips found")
        print(f"Batch finished. Processed {processed} folders.")

    elif args.input_root:
        # Legacy behavior: traverse input_root structure
        sub_folders = sorted([d for d in os.listdir(args.input_root) if os.path.isdir(os.path.join(args.input_root, d))])
        print(f"Found {len(sub_folders)} subfolders in {args.input_root}")

        for sub in sub_folders:
            sub_path = os.path.join(args.input_root, sub)
            item_folders = sorted([d for d in os.listdir(sub_path) if os.path.isdir(os.path.join(sub_path, d))])
            
            print(f"Processing subfolder: {sub} ({len(item_folders)} items)")
            
            for item in item_folders:
                item_path = os.path.join(sub_path, item)
                mask_root = os.path.join(item_path, 'mask')
                photo_root = os.path.join(item_path, 'photo')
                
                if not os.path.exists(mask_root) or not os.path.exists(photo_root):
                    print(f"Skipping {item}: mask or photo folder missing")
                    continue
                    
                # Find the mask subfolder
                mask_subs = [d for d in os.listdir(mask_root) if os.path.isdir(os.path.join(mask_root, d))]
                if len(mask_subs) != 1:
                    print(f"Skipping {item}: Expected 1 mask subfolder, found {len(mask_subs)}")
                    continue
                
                mask_sub_name = mask_subs[0]
                mask_path = os.path.join(mask_root, mask_sub_name)
                
                # Find the matching photo subfolder (same name)
                photo_path = os.path.join(photo_root, mask_sub_name)
                
                if not os.path.exists(photo_path):
                    print(f"Skipping {item}: Matching photo folder '{mask_sub_name}' not found in {photo_root}")
                    continue
                
                # Preserve exact mask directory structure: include 'mask' level so output mirrors input mask path
                save_path = os.path.join(args.output_root, sub, item, 'mask', mask_sub_name)
                
                print(f"Processing: {sub}/{item}/{mask_sub_name}")
                inference_single(models, device, args, photo_path, mask_path, save_path)

    else:
        parser.error('Please provide either --photo_root and --mask_root, or --input_root')

if __name__ == '__main__':
    main()

