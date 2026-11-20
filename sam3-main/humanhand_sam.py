import os
import sys
import argparse
import torch
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading
import re
from pathlib import Path
from typing import Optional

# Ensure local package import works when running from this script
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sam3.model_builder import build_sam3_video_predictor
from sam3.model.io_utils import load_image_as_single_frame_video
import sam3.model.io_utils

# Restrict SAM3 to only load JPGs to avoid loading depth maps/masks as frames
sam3.model.io_utils.IMAGE_EXTS = ['.jpg', '.jpeg']

def natural_sort_key(s):
    """
    Sorts strings that contain numbers in a human-friendly way.
    e.g. [file1, file2, file10] instead of [file1, file10, file2]
    """
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]


class SAM3VideoSegmenter:
    def __init__(self, checkpoint_path: Optional[str] = None, device: str = "cuda"):
        self.device = device
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        print(f"Building SAM 3 video predictor on {device}...")
        self.predictor = build_sam3_video_predictor(checkpoint_path=checkpoint_path)
        self.session_id = None
        self.video_path = None
        self.temp_video_path = None

    def start_session(self, video_path: str):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video path not found: {video_path}")
        
        self.video_path = video_path

        response = self.predictor.handle_request(
            request={
                "type": "start_session",
                "resource_path": video_path
            }
        )
        self.session_id = response["session_id"]
        return self.session_id

    def add_prompt(self, frame_index, points=None, labels=None, box=None, text=None, obj_id=None):
        if self.session_id is None:
            raise RuntimeError("Session not started.")
        
        request = {
            "type": "add_prompt",
            "session_id": self.session_id,
            "frame_index": frame_index,
        }

        # Get dimensions
        if os.path.isdir(self.video_path):
            # It's a directory, try to find an image to get dims
            valid_exts = ('.jpg',)
            files = [f for f in os.listdir(self.video_path) if f.lower().endswith(valid_exts)]
            # Sort to match backend logic
            files.sort()
                
            if files:
                img = cv2.imread(os.path.join(self.video_path, files[0]))
                if img is not None:
                    height, width = img.shape[:2]
                else:
                    width, height = 1920, 1080
            else:
                width, height = 1920, 1080
        else:
            cap = cv2.VideoCapture(self.video_path)
            if cap.isOpened():
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
            else:
                # Fallback
                width, height = 1920, 1080 

        if text:
            request["text"] = text
        
        if points is not None and labels is not None:
            points_normalized = [[x / width, y / height] for x, y in points]
            request["points"] = torch.tensor(points_normalized, dtype=torch.float32)
            request["point_labels"] = torch.tensor(labels, dtype=torch.int32)
            
        if box is not None:
            x1, y1, x2, y2 = box
            # Convert xyxy to xywh
            w = x2 - x1
            h = y2 - y1
            box_normalized = [x1 / width, y1 / height, w / width, h / height]
            request["bounding_boxes"] = torch.tensor([box_normalized], dtype=torch.float32)
            request["bounding_box_labels"] = torch.tensor([1], dtype=torch.int32)

        if obj_id is not None:
            request["obj_id"] = obj_id

        # Use autocast to handle BFloat16/Float32 mismatch in SAM3 model
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return self.predictor.handle_request(request=request)

    def propagate(self, start_frame_index=0):
        """Generator that yields (frame_idx, mask) tuples."""
        # Use autocast to handle BFloat16/Float32 mismatch in SAM3 model
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for result in self.predictor.handle_stream_request(
                request={
                    "type": "propagate_in_video",
                    "session_id": self.session_id,
                    "propagation_direction": "forward",
                    "start_frame_index": start_frame_index,
                }
            ):
                yield result

    def close(self):
        if self.session_id:
            try:
                self.predictor.handle_request(request={"type": "close_session", "session_id": self.session_id})
            except:
                pass
            self.session_id = None

class SAM3VideoGUI:
    def __init__(self, root, checkpoint=None):
        self.root = root
        self.checkpoint = checkpoint or Path(__file__).resolve().parent.parent / "weights" / "sam3.pt"
        self.root.title("SAM3 Video Segmentation Tool")
        self.root.geometry("1400x900")
        
        # Variables
        self.segmenter = None
        self.current_video_path = None
        self.current_frame_index = 0
        self.total_frames = 0
        self.fps = 30
        self.video_width = 0
        self.video_height = 0
        self.scale_factor = 1.0
        
        self.frame_files = [] # List of absolute paths to frames
        
        self.video_list = [] # List of paths for batch processing
        self.root_input_folder = None
        self.batch_data = {} # path -> {'masks': {}, 'annotations': {}, 'history': []}

        # Annotations: frame_idx -> {'pos': [], 'neg': [], 'bbox': []}
        self.annotations = {}
        
        self.temp_bbox = None
        self.bbox_start = None
        self.operation_history = [] # List of (frame_idx, type, data)
        self.text_prompt = tk.StringVar()
        
        # Results
        self.video_masks = {}  # {frame_idx: mask_numpy}
        
        # Display
        self.display_image = None
        self.current_mode = "positive"
        
        # Fast Mode toggle
        self.fast_mode_var = tk.BooleanVar(value=True)
        self.view_mode_var = tk.StringVar(value="overlay")
        
        self.setup_ui()
        # Defer model loading to allow UI to show up
        self.root.after(100, self.load_model)
        
    def setup_ui(self):
        # Main Layout
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left Control Panel
        control_container = ttk.Frame(main_frame, width=300)
        control_container.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        control_container.pack_propagate(False)
        
        # Scrollable Control Panel
        control_canvas = tk.Canvas(control_container, width=280, highlightthickness=0)
        control_scrollbar = ttk.Scrollbar(control_container, orient=tk.VERTICAL, command=control_canvas.yview)
        control_frame = ttk.Frame(control_canvas)
        control_canvas.configure(yscrollcommand=control_scrollbar.set)
        
        control_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        control_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        control_canvas.create_window((0, 0), window=control_frame, anchor=tk.NW)
        
        control_frame.bind("<Configure>", lambda e: control_canvas.configure(scrollregion=control_canvas.bbox("all")))
        
        # 1. Video Loading
        load_frame = ttk.LabelFrame(control_frame, text="1. Video Loading", padding=10)
        load_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(load_frame, text="Load Image Folder", command=self.browse_image_folder).pack(fill=tk.X, pady=2)

        # Combobox to select among discovered subfolders (recursive)
        self.video_selector = ttk.Combobox(load_frame, state='readonly', values=[])
        self.video_selector.pack(fill=tk.X, pady=2)
        self.video_selector.bind('<<ComboboxSelected>>', self.on_video_selection_change)
        
        # 2. Navigation
        nav_frame = ttk.LabelFrame(control_frame, text="2. Navigation", padding=10)
        nav_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.frame_scale = ttk.Scale(nav_frame, from_=0, to=100, orient=tk.HORIZONTAL, command=self.on_frame_scale_change)
        self.frame_scale.pack(fill=tk.X, pady=5)
        
        nav_btn_frame = ttk.Frame(nav_frame)
        nav_btn_frame.pack(fill=tk.X)
        ttk.Button(nav_btn_frame, text="<<", width=4, command=self.first_frame).pack(side=tk.LEFT, padx=1)
        ttk.Button(nav_btn_frame, text="<", width=4, command=self.prev_frame).pack(side=tk.LEFT, padx=1)
        ttk.Button(nav_btn_frame, text=">", width=4, command=self.next_frame).pack(side=tk.LEFT, padx=1)
        ttk.Button(nav_btn_frame, text=">>", width=4, command=self.last_frame).pack(side=tk.LEFT, padx=1)
        
        self.frame_info_label = ttk.Label(nav_frame, text="Frame: 0/0")
        self.frame_info_label.pack(pady=5)
        
        # View Options
        view_frame = ttk.LabelFrame(control_frame, text="View Options", padding=10)
        view_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Radiobutton(view_frame, text="Overlay Mask (Yellow)", variable=self.view_mode_var, value="overlay", command=self.display_current_frame).pack(anchor=tk.W)
        ttk.Radiobutton(view_frame, text="Show Cutout (Black BG)", variable=self.view_mode_var, value="cutout", command=self.display_current_frame).pack(anchor=tk.W)
        
        # 3. Annotation
        mode_frame = ttk.LabelFrame(control_frame, text="3. Annotation", padding=10)
        mode_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.mode_var = tk.StringVar(value="positive")
        ttk.Radiobutton(mode_frame, text="Positive Point (Green)", variable=self.mode_var, value="positive", command=self.change_mode).pack(anchor=tk.W)
        ttk.Radiobutton(mode_frame, text="Negative Point (Red)", variable=self.mode_var, value="negative", command=self.change_mode).pack(anchor=tk.W)
        ttk.Radiobutton(mode_frame, text="Bounding Box (Blue)", variable=self.mode_var, value="bbox", command=self.change_mode).pack(anchor=tk.W)
        
        ttk.Label(mode_frame, text="Text Prompt:").pack(anchor=tk.W, pady=(5,0))
        ttk.Entry(mode_frame, textvariable=self.text_prompt).pack(fill=tk.X, pady=2)
        
        # Info & Actions
        info_frame = ttk.LabelFrame(control_frame, text="Annotation Info (Current Frame)", padding=10)
        info_frame.pack(fill=tk.X, pady=(0, 10))
        self.points_info_label = ttk.Label(info_frame, text="Pos: 0 | Neg: 0 | Box: 0")
        self.points_info_label.pack()
        
        action_frame = ttk.Frame(control_frame)
        action_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(action_frame, text="Undo Last", command=self.undo_operation).pack(fill=tk.X, pady=2)
        ttk.Button(action_frame, text="Clear Current Frame", command=self.clear_current_annotations).pack(fill=tk.X, pady=2)
        ttk.Button(action_frame, text="Clear All Frames", command=self.clear_all_annotations).pack(fill=tk.X, pady=2)
        
        # 4. Inference
        run_frame = ttk.LabelFrame(control_frame, text="4. Inference", padding=10)
        run_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Checkbutton(run_frame, text="Fast Mode (Skip Pre-run)", variable=self.fast_mode_var).pack(anchor=tk.W, pady=(0, 5))
        ttk.Button(run_frame, text="▶ Run SAM 3 Inference", command=self.run_video_inference_thread).pack(fill=tk.X, pady=5)
        ttk.Button(run_frame, text="▶ Batch Process All (Text Prompt)", command=self.run_batch_processing_thread).pack(fill=tk.X, pady=5)
        
        # 5. Export
        save_frame = ttk.LabelFrame(control_frame, text="5. Export", padding=10)
        save_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(save_frame, text="Save Masks (PNG)", command=self.save_all_masks).pack(fill=tk.X, pady=2)
        ttk.Button(save_frame, text="Save Cutouts (PNG)", command=self.save_all_cutouts).pack(fill=tk.X, pady=2)
        ttk.Button(save_frame, text="Save Batch Masks (PNG)", command=self.save_batch_masks).pack(fill=tk.X, pady=2)
        ttk.Button(save_frame, text="Save Batch Cutouts (PNG)", command=self.save_cutout_images).pack(fill=tk.X, pady=2)

        # Right Display Area
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        canvas_container = ttk.Frame(right_frame)
        canvas_container.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(canvas_container, bg="#333333")
        v_bar = ttk.Scrollbar(canvas_container, orient=tk.VERTICAL, command=self.canvas.yview)
        h_bar = ttk.Scrollbar(canvas_container, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=v_bar.set, xscrollcommand=h_bar.set)
        
        v_bar.pack(side=tk.RIGHT, fill=tk.Y)
        h_bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.canvas.bind("<Configure>", self.on_canvas_resize)
        
        # Status Bar
        self.status_var = tk.StringVar(value="Ready - Please load a video")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, padding=2)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def load_model(self):
        try:
            self.status_var.set("Loading SAM 3 model...")
            self.root.update()
            self.segmenter = SAM3VideoSegmenter(checkpoint_path=str(self.checkpoint))
            self.status_var.set("SAM 3 Model Loaded.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {e}")
            self.status_var.set("Model Load Failed")

    def browse_image_folder(self):
        folder_path = filedialog.askdirectory(title="Select Image Folder")
        if not folder_path: return
        
        self.root_input_folder = folder_path
        self.video_list = []
        valid_exts = ('.jpg',)

        # Walk tree recursively
        for dirpath, dirnames, filenames in os.walk(folder_path):
            # Check for meta.yml in current directory to filter subdirectories
            meta_path = os.path.join(dirpath, "meta.yml")
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    import re
                    match = re.search(r"mano_sides.*?(?:['\"]|(?<=\s))(right|left)(?:['\"]|(?=\s)|$)", content, re.IGNORECASE | re.DOTALL)
                    
                    target_subdir = None
                    if match:
                        side = match.group(1).lower()
                        if side == "right":
                            target_subdir = "840412060917"
                        elif side == "left":
                            target_subdir = "841412060263"
                        
                        print(f"Found meta.yml in {dirpath}: side={side} -> target={target_subdir}")

                    if target_subdir:
                        # If the target subdirectory exists, prune the walk to ONLY visit that one
                        if target_subdir in dirnames:
                            dirnames[:] = [target_subdir]
                        else:
                            # Target folder expected but not found? 
                            # Don't traverse other folders if we are strict about the rule
                            # But maybe the meta.yml is one level up? 
                            # The user said "meta.yml file in the same directory as the 8 folders"
                            # So if we are here, dirnames contains the 8 folders.
                            # If target_subdir is not in dirnames, we probably shouldn't visit the others either if we want to be strict.
                            # Let's clear dirnames to avoid loading wrong data.
                            dirnames[:] = []
                            
                except Exception as e:
                    print(f"Error reading meta.yml in {dirpath}: {e}")

            # Standard image finding logic
            imgs = [f for f in filenames if f.lower().endswith(valid_exts)]
            if imgs:
                self.video_list.append(dirpath)

        # Deduplicate and sort using natural sort order on the full path
        self.video_list = sorted(list(dict.fromkeys(self.video_list)), key=natural_sort_key)

        if not self.video_list:
            msg = "No images found in selected folder or its subfolders."
            messagebox.showerror("Error", msg)
            return

        self.batch_data = {} # Reset batch data

        # Populate combobox with basenames and auto-select first
        names = []
        for p in self.video_list:
            if self.root_input_folder and p.startswith(self.root_input_folder):
                names.append(os.path.relpath(p, self.root_input_folder))
            else:
                names.append(os.path.basename(p))
        
        self.video_selector['values'] = names
        if names:
            self.video_selector.current(0)

        # Load the first available video/folder
        self.load_video_source(self.video_list[0])

    # Video file loading removed per user request
    pass

    def on_video_selection_change(self, event):
        idx = self.video_selector.current()
        if idx < 0: return
        
        new_path = self.video_list[idx]
        if new_path == self.current_video_path: return
        
        # Save current state
        if self.current_video_path:
            self.batch_data[self.current_video_path] = {
                'masks': self.video_masks,
                'annotations': self.annotations,
                'history': self.operation_history,
                'text_prompt': self.text_prompt.get()
            }
            
        self.load_video_source(new_path)

    def load_video_source(self, path):
        try:
            self.status_var.set(f"Loading {os.path.basename(path)}...")
            self.root.update()
            
            # Preserve text prompt across video switches
            kept_text = self.text_prompt.get()
            
            # Close previous video session but keep batch data
            if self.segmenter and self.segmenter.session_id:
                self.segmenter.close()
            
            # Clear current display data
            self.video_masks = {}
            self.clear_all_annotations()
            self.canvas.delete("all")
            
            # Restore text prompt
            self.text_prompt.set(kept_text)
            
            # Only support folder-based image sequences
            if not os.path.isdir(path):
                # Silently fail (do not mention video support)
                self.status_var.set("Load failed")
                return

            # Folder Logic
            valid_exts = ('.jpg',)
            files = [f for f in os.listdir(path) if f.lower().endswith(valid_exts)]

            # Sort files using standard sort to match backend
            files.sort()

            self.frame_files = [os.path.join(path, f) for f in files]
            self.total_frames = len(self.frame_files)

            if self.frame_files:
                img = cv2.imread(self.frame_files[0])
                if img is not None:
                    self.video_height, self.video_width = img.shape[:2]

            self.current_video_path = path
            self.current_frame_index = 0
            self.frame_scale.configure(to=self.total_frames - 1)
            self.frame_scale.set(0)
            
            # Start SAM session
            self.segmenter.start_session(path)
            
            # Restore state if exists
            if path in self.batch_data:
                data = self.batch_data[path]
                self.video_masks = data.get('masks', {})
                self.annotations = data.get('annotations', {})
                self.operation_history = data.get('history', [])
                
                saved_text = data.get('text_prompt', "")
                if saved_text:
                    self.text_prompt.set(saved_text)
                
                self.status_var.set(f"Restored state for: {os.path.basename(path)}")
            else:
                self.status_var.set(f"Loaded: {os.path.basename(path)}")
            
            self.update_video_info()
            self.display_current_frame()
            
        except Exception as e:
            messagebox.showerror("Error", f"Load failed: {e}")
            self.status_var.set("Load failed")
            import traceback
            traceback.print_exc()

    # Deprecated methods mapped to new ones
    def load_image_folder(self): self.browse_image_folder()
    def load_video(self):
        # No-op: video file load option removed
        pass

    def close_video(self):
        # No temp video frames to clean up (video loading removed)
        self.frame_files = []
        self.current_video_path = None
        self.video_masks = {}
        self.clear_all_annotations()
        self.canvas.delete("all")
        if self.segmenter:
            self.segmenter.close()
        self.update_video_info()
        self.video_list = []
        self.video_selector['values'] = []
        self.video_selector.set('')
        self.batch_data = {}

    def on_canvas_resize(self, event):
        if self.current_video_path:
            self.display_current_frame()

    def display_current_frame(self):
        if not self.frame_files: return
        
        try:
            if self.current_frame_index >= len(self.frame_files): return
            frame_path = self.frame_files[self.current_frame_index]
            if not os.path.exists(frame_path): return
            
            frame = cv2.imread(frame_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Overlay mask
            if self.current_frame_index in self.video_masks:
                mask = self.video_masks[self.current_frame_index]
                if mask is not None:
                    # Resize mask if needed
                    if mask.shape != frame.shape[:2]:
                        mask = cv2.resize(mask.astype(np.uint8), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                    
                    if self.view_mode_var.get() == "cutout":
                        # Apply mask (black background)
                        frame = cv2.bitwise_and(frame, frame, mask=mask)
                    else:
                        # Create overlay
                        color = np.array([0, 255, 255], dtype=np.uint8) # Yellow
                        colored_mask = np.zeros_like(frame)
                        colored_mask[mask > 0] = color
                        
                        alpha = 0.4
                        mask_bool = mask > 0
                        
                        # Blend
                        if mask_bool.any():
                            frame[mask_bool] = cv2.addWeighted(frame[mask_bool], 1-alpha, colored_mask[mask_bool], alpha, 0).squeeze()

            # Draw annotations for current frame
            self.draw_annotations(frame)
            
            pil_image = Image.fromarray(frame)
            
            # Scale to fit canvas
            c_w = self.canvas.winfo_width()
            c_h = self.canvas.winfo_height()
            if c_w > 1 and c_h > 1:
                i_w, i_h = pil_image.size
                # Allow scaling up or down to fit
                scale = min(c_w/i_w, c_h/i_h) 
                self.scale_factor = scale
                new_size = (int(i_w*scale), int(i_h*scale))
                pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
            
            self.display_image = ImageTk.PhotoImage(pil_image)
            self.canvas.delete("all")
            # Center the image
            x_pos = (c_w - new_size[0]) // 2
            y_pos = (c_h - new_size[1]) // 2
            self.canvas.create_image(x_pos, y_pos, anchor=tk.NW, image=self.display_image)
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
            self.update_frame_info()
            self.update_points_info()
            
        except Exception as e:
            print(f"Display error: {e}")

    def draw_annotations(self, image):
        if self.current_frame_index not in self.annotations:
            return

        ann = self.annotations[self.current_frame_index]
        
        for pt in ann['pos']:
            cv2.circle(image, pt, 5, (0, 255, 0), -1) # Green
            cv2.circle(image, pt, 6, (0, 0, 0), 1)
        for pt in ann['neg']:
            cv2.circle(image, pt, 5, (255, 0, 0), -1) # Red
            cv2.circle(image, pt, 6, (0, 0, 0), 1)
        for box in ann['bbox']:
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2) # Blue
            
        if self.temp_bbox:
            cv2.rectangle(image, (self.temp_bbox[0], self.temp_bbox[1]), (self.temp_bbox[2], self.temp_bbox[3]), (255, 255, 0), 2)

    def canvas_to_image_coords(self, cx, cy):
        if self.scale_factor <= 0: return int(cx), int(cy)
        
        # Adjust for centered image
        c_w = self.canvas.winfo_width()
        c_h = self.canvas.winfo_height()
        
        # Calculate current image size on canvas
        img_w = int(self.video_width * self.scale_factor)
        img_h = int(self.video_height * self.scale_factor)
        
        x_offset = (c_w - img_w) // 2
        y_offset = (c_h - img_h) // 2
        
        return int((cx - x_offset) / self.scale_factor), int((cy - y_offset) / self.scale_factor)

    def on_canvas_click(self, event):
        if not self.current_video_path: return
        
        # Use canvas coordinates to handle scrolling correctly
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)
        
        ix, iy = self.canvas_to_image_coords(cx, cy)
        
        # Bounds check
        if ix < 0 or ix >= self.video_width or iy < 0 or iy >= self.video_height:
            return
        
        # Initialize annotations for this frame if needed
        if self.current_frame_index not in self.annotations:
            self.annotations[self.current_frame_index] = {'pos': [], 'neg': [], 'bbox': []}
        
        mode = self.mode_var.get()
        if mode == "positive":
            self.annotations[self.current_frame_index]['pos'].append((ix, iy))
            self.operation_history.append((self.current_frame_index, "pos", (ix, iy)))
            self.status_var.set(f"Added Positive Point at ({ix}, {iy})")
        elif mode == "negative":
            self.annotations[self.current_frame_index]['neg'].append((ix, iy))
            self.operation_history.append((self.current_frame_index, "neg", (ix, iy)))
            self.status_var.set(f"Added Negative Point at ({ix}, {iy})")
        elif mode == "bbox":
            self.bbox_start = (ix, iy)
        
        self.display_current_frame()

    def on_canvas_drag(self, event):
        if not self.current_video_path or self.mode_var.get() != "bbox" or not self.bbox_start: return
        
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)
        ix, iy = self.canvas_to_image_coords(cx, cy)
        
        ix = max(0, min(ix, self.video_width-1))
        iy = max(0, min(iy, self.video_height-1))
        
        self.temp_bbox = (min(self.bbox_start[0], ix), min(self.bbox_start[1], iy), max(self.bbox_start[0], ix), max(self.bbox_start[1], iy))
        self.display_current_frame()

    def on_canvas_release(self, event):
        if not self.current_video_path or self.mode_var.get() != "bbox" or not self.bbox_start: return
        
        if self.temp_bbox:
            if self.current_frame_index not in self.annotations:
                self.annotations[self.current_frame_index] = {'pos': [], 'neg': [], 'bbox': []}
                
            # Only allow one bbox per frame for simplicity in this version, or append
            # self.annotations[self.current_frame_index]['bbox'] = [self.temp_bbox] # Replace
            self.annotations[self.current_frame_index]['bbox'].append(self.temp_bbox) # Append
            
            self.operation_history.append((self.current_frame_index, "bbox", self.temp_bbox))
            self.status_var.set(f"Added Bounding Box")
            
            self.temp_bbox = None
            self.bbox_start = None
            self.display_current_frame()

    def undo_operation(self):
        if not self.operation_history: return
        frame_idx, op, data = self.operation_history.pop()
        
        if frame_idx in self.annotations:
            if op == "pos" and data in self.annotations[frame_idx]['pos']:
                self.annotations[frame_idx]['pos'].remove(data)
            elif op == "neg" and data in self.annotations[frame_idx]['neg']:
                self.annotations[frame_idx]['neg'].remove(data)
            elif op == "bbox" and data in self.annotations[frame_idx]['bbox']:
                self.annotations[frame_idx]['bbox'].remove(data)
                
        self.display_current_frame()
        self.status_var.set("Undid last operation")

    def clear_current_annotations(self):
        if self.current_frame_index in self.annotations:
            del self.annotations[self.current_frame_index]
            # Also clean up history for this frame
            self.operation_history = [x for x in self.operation_history if x[0] != self.current_frame_index]
            self.display_current_frame()
            self.status_var.set("Cleared annotations for current frame")

    def clear_all_annotations(self):
        self.annotations = {}
        self.operation_history = []
        self.text_prompt.set("")
        self.display_current_frame()
        self.status_var.set("Cleared all annotations")

    def update_points_info(self):
        if self.current_frame_index in self.annotations:
            ann = self.annotations[self.current_frame_index]
            self.points_info_label.config(text=f"Pos: {len(ann['pos'])} | Neg: {len(ann['neg'])} | Box: {len(ann['bbox'])}")
        else:
            self.points_info_label.config(text="Pos: 0 | Neg: 0 | Box: 0")

    def update_video_info(self):
        pass

    def update_frame_info(self):
        self.frame_info_label.config(text=f"Frame: {self.current_frame_index}/{self.total_frames-1}")

    def on_frame_scale_change(self, value):
        idx = int(float(value))
        if idx != self.current_frame_index:
            self.current_frame_index = idx
            self.display_current_frame()

    def first_frame(self): self.frame_scale.set(0)
    def prev_frame(self): self.frame_scale.set(max(0, self.current_frame_index-1))
    def next_frame(self): self.frame_scale.set(min(self.total_frames-1, self.current_frame_index+1))
    def last_frame(self): self.frame_scale.set(self.total_frames-1)
    def change_mode(self): pass

    def save_current_state(self):
        if self.current_video_path:
            self.batch_data[self.current_video_path] = {
                'masks': self.video_masks,
                'annotations': self.annotations,
                'history': self.operation_history,
                'text_prompt': self.text_prompt.get()
            }

    def run_video_inference_thread(self):
        threading.Thread(target=self.run_video_inference, daemon=True).start()

    def run_video_inference(self):
        # Save current state first
        self.save_current_state()
        
        # Identify tasks (videos with annotations or text prompts)
        tasks = []
        for path in self.video_list:
            if path in self.batch_data:
                data = self.batch_data[path]
                # Only include folders that have manual annotations (pos/neg points or bbox)
                has_ann = False
                anns = data.get('annotations', {})
                for f_idx, ann in anns.items():
                    if ann.get('pos') or ann.get('neg') or ann.get('bbox'):
                        has_ann = True
                        break

                if has_ann:
                    tasks.append(path)
        
        if not tasks:
            messagebox.showwarning("Warning", "No manual annotations found in any folder. (Folders with only text prompts will be skipped.)")
            return

        self.status_var.set(f"Starting inference on {len(tasks)} folders...")
        
        total_processed = 0
        error_count = 0
        
        for i, path in enumerate(tasks):
            self.status_var.set(f"Processing {i+1}/{len(tasks)}: {os.path.basename(path)}")
            self.root.update()
            
            data = self.batch_data[path]
            anns = data.get('annotations', {})
            text = data.get('text_prompt', '')
            
            try:
                masks = self._run_single_video_inference(path, anns, text)
                self.batch_data[path]['masks'] = masks
                
                # If this is the currently viewed video, update the display immediately
                if path == self.current_video_path:
                    self.video_masks = masks
                    self.root.after(0, self.display_current_frame)
                    
                total_processed += 1
            except Exception as e:
                print(f"Failed on {path}: {e}")
                error_count += 1
                import traceback
                traceback.print_exc()

        # Restore session for current video if needed
        if self.current_video_path and self.segmenter.video_path != self.current_video_path:
             self.status_var.set(f"Restoring session for {os.path.basename(self.current_video_path)}...")
             if self.segmenter.session_id:
                 self.segmenter.close()
             self.segmenter.start_session(self.current_video_path)

        self.status_var.set(f"Completed. Processed {total_processed} folders. Errors: {error_count}")
        messagebox.showinfo("Inference Complete", f"Processed {total_processed} folders.\nErrors: {error_count}")

    def _run_single_video_inference(self, video_path, annotations, text_prompt):
        # Close previous session
        if self.segmenter.session_id:
            self.segmenter.close()
            
        self.segmenter.start_session(video_path)
        
        # Get dimensions
        width, height = 1920, 1080
        if os.path.isdir(video_path):
            valid_exts = ('.jpg',)
            files = [f for f in os.listdir(video_path) if f.lower().endswith(valid_exts)]
            files.sort()
            if files:
                img = cv2.imread(os.path.join(video_path, files[0]))
                if img is not None:
                    height, width = img.shape[:2]
        
        video_masks = {}
        
        # Strategy for "One Object" segmentation:
        # 1. Initialization: We need ONE "Starter Prompt" (Text or Box) to initialize the object.
        # 2. Pre-Propagation: If there are refinement points on other frames, we MUST propagate first 
        #    to populate the cache for those frames.
        # 3. Refinement: Send points for all frames (converting secondary boxes to points).
        # 4. Final Propagation: Generate the final masks.
        
        prompts_added = False
        target_obj_id = 0 # Default ID
        starter_frame_idx = -1
        
        # --- Step 1: Find and Send Initializer (Text, Box, or Points) ---
        
        # Check for Text
        text = text_prompt.strip()
        
        if text:
            # Case A: Text Prompt
            print(f"Initializing with Text: {text}")
            resp = self.segmenter.add_prompt(frame_index=0, text=text)
            prompts_added = True
            starter_frame_idx = 0 # Text effectively starts at 0 for our purposes
            
            # Try to capture ID from text response
            if resp and 'outputs' in resp and resp['outputs']:
                outputs = resp['outputs']
                if isinstance(outputs, dict) and 'out_obj_ids' in outputs:
                    ids = outputs['out_obj_ids']
                    if len(ids) > 0:
                        target_obj_id = int(ids[0])
                        print(f"Text initialized object ID: {target_obj_id}")
            
        else:
            # Case B: Box Prompt (Find the first frame with a box)
            sorted_frames = sorted(annotations.keys())
            found_box = False
            for frame_idx in sorted_frames:
                ann = annotations[frame_idx]
                if ann['bbox']:
                    # Found a box! Use this as the initializer.
                    box = ann['bbox'][0] # Use the first box
                    print(f"Initializing with Box on frame {frame_idx}")
                    
                    resp = self.segmenter.add_prompt(
                        frame_index=frame_idx,
                        box=box
                    )
                    prompts_added = True
                    starter_frame_idx = frame_idx
                    found_box = True
                    
                    # Try to capture the assigned ID
                    if resp and 'outputs' in resp and resp['outputs']:
                        outputs = resp['outputs']
                        if isinstance(outputs, dict) and 'out_obj_ids' in outputs:
                            ids = outputs['out_obj_ids']
                            if len(ids) > 0:
                                target_obj_id = int(ids[0])
                                print(f"Box initialized object ID: {target_obj_id}")
                    break # Stop after the first box
            
            # Case C: Points Only (Construct Box from Points)
            if not found_box:
                # Find first frame with points
                for frame_idx in sorted_frames:
                    ann = annotations[frame_idx]
                    points = ann['pos'] + ann['neg']
                    if points:
                        print(f"Initializing with Points on frame {frame_idx} (via constructed box)")
                        # Construct bounding box from points
                        xs = [p[0] for p in points]
                        ys = [p[1] for p in points]
                        x1, y1 = min(xs), min(ys)
                        x2, y2 = max(xs), max(ys)
                        
                        # Add some padding to the box (e.g. 0 pixels - user requested no expansion)
                        pad = 0
                        x1 = max(0, x1 - pad)
                        y1 = max(0, y1 - pad)
                        x2 = min(width, x2 + pad)
                        y2 = min(height, y2 + pad)
                        
                        # If box is too small (single point), make it a small square
                        if x2 - x1 < 5: x1 -= 5; x2 += 5
                        if y2 - y1 < 5: y1 -= 5; y2 += 5
                        
                        box = (x1, y1, x2, y2)
                        
                        resp = self.segmenter.add_prompt(
                            frame_index=frame_idx,
                            box=box
                        )
                        prompts_added = True
                        starter_frame_idx = frame_idx
                        
                        # Try to capture the assigned ID
                        if resp and 'outputs' in resp and resp['outputs']:
                            outputs = resp['outputs']
                            if isinstance(outputs, dict) and 'out_obj_ids' in outputs:
                                ids = outputs['out_obj_ids']
                                if len(ids) > 0:
                                    target_obj_id = int(ids[0])
                                    print(f"Points initialized object ID: {target_obj_id}")
                        break

        # --- Step 2: Collect Refinement Requests ---
        refinement_requests = []
        sorted_frames = sorted(annotations.keys())
        
        for frame_idx in sorted_frames:
            ann = annotations[frame_idx]
            
            # Start with explicit points
            current_points = list(ann['pos'] + ann['neg'])
            current_labels = [1]*len(ann['pos']) + [0]*len(ann['neg'])
            
            # Check if we need to convert boxes to points on this frame
            if ann['bbox']:
                should_convert = False
                if text:
                    # If initialized by text, ALL boxes are refinement
                    should_convert = True
                elif starter_frame_idx != -1 and frame_idx != starter_frame_idx:
                    # If initialized by box, only SUBSEQUENT boxes are refinement
                    should_convert = True
                    
                if should_convert:
                    print(f"Converting Box on frame {frame_idx} to Point for refinement.")
                    for b in ann['bbox']:
                        # b is (x1, y1, x2, y2)
                        cx = (b[0] + b[2]) / 2
                        cy = (b[1] + b[3]) / 2
                        current_points.append((cx, cy))
                        current_labels.append(1) # Treat box center as positive point
            
            if current_points:
                refinement_requests.append({
                    'frame_idx': frame_idx,
                    'points': current_points,
                    'labels': current_labels
                })

        # --- Step 3: Pre-Propagate (Conditional) ---
        # If we have refinement requests on later frames, we MUST pre-propagate
        # to ensure the model's cache is populated. SAM3 requires this for adding points to existing tracks.
        needs_pre_propagation = len(refinement_requests) > 0
        
        if needs_pre_propagation:
            print("Pre-propagating to populate cache for refinement...")
            # Run propagation to populate cache.
            for _ in self.segmenter.propagate(start_frame_index=0):
                pass
            print("Pre-propagation complete.")
        else:
            # No refinements needed
            pass

        # --- Step 4: Apply Refinements ---
        for req in refinement_requests:
            print(f"Adding refinement points on frame {req['frame_idx']} for ID {target_obj_id}")
            self.segmenter.add_prompt(
                frame_index=req['frame_idx'],
                points=req['points'],
                labels=req['labels'],
                obj_id=target_obj_id
            )
            prompts_added = True

        if not prompts_added:
             pass 

        # --- Step 5: Final Propagation ---
        print("Running final propagation...")
        
        for result in self.segmenter.propagate(start_frame_index=0):
            frame_idx = result['frame_index']
            outputs = result.get('outputs', None)
            mask = None
            if outputs:
                if isinstance(outputs, dict) and 'out_binary_masks' in outputs:
                    m = outputs['out_binary_masks']
                    if isinstance(m, torch.Tensor): m = m.detach().cpu().numpy()
                    if m.size > 0: mask = m[0]
                elif isinstance(outputs, (list, tuple)) and len(outputs) >= 3:
                    m = outputs[2]
                    if isinstance(m, torch.Tensor): m = m.detach().cpu().numpy()
                    if m.size > 0: mask = m[0]

            if mask is not None:
                if mask.ndim == 3: mask = mask[0]
                if mask.ndim == 2:
                    binary_mask = (mask > 0).astype(np.uint8)
                    kernel = np.ones((3, 3), np.uint8)
                    binary_mask = cv2.dilate(binary_mask, kernel, iterations=1)
                    video_masks[frame_idx] = binary_mask
                    
        return video_masks

    def run_batch_processing_thread(self):
        threading.Thread(target=self.run_batch_processing, daemon=True).start()

    def run_batch_processing(self):
        if not self.video_list: return
        text = self.text_prompt.get().strip()
        if not text:
            messagebox.showwarning("Warning", "Please enter a Text Prompt for batch processing.")
            return
            
        save_root = filedialog.askdirectory(title="Select Output Directory for Batch Results")
        if not save_root: return
        
        self.status_var.set("Starting Batch Processing...")
        
        total_videos = len(self.video_list)
        
        # Save current state to restore later
        original_video_path = self.current_video_path
        
        for idx, video_path in enumerate(self.video_list):
            try:
                self.status_var.set(f"Batch Processing {idx+1}/{total_videos}: {os.path.basename(video_path)}")
                self.root.update()
                
                # Close existing session to free memory
                if self.segmenter.session_id:
                    self.segmenter.close()
                    
                self.segmenter.start_session(video_path)
                
                # Add Prompt
                self.segmenter.add_prompt(frame_index=0, text=text)
                
                # Propagate
                video_masks = {}
                for result in self.segmenter.propagate(start_frame_index=0):
                    frame_idx = result['frame_index']
                    outputs = result.get('outputs', None)
                    mask = None
                    if outputs:
                        if isinstance(outputs, dict) and 'out_binary_masks' in outputs:
                            m = outputs['out_binary_masks']
                            if isinstance(m, torch.Tensor): m = m.detach().cpu().numpy()
                            if m.size > 0: mask = m[0]
                        elif isinstance(outputs, (list, tuple)) and len(outputs) >= 3:
                            m = outputs[2]
                            if isinstance(m, torch.Tensor): m = m.detach().cpu().numpy()
                            if m.size > 0: mask = m[0]
                    
                    if mask is not None:
                        if mask.ndim == 3: mask = mask[0]
                        if mask.ndim == 2:
                            binary_mask = (mask > 0).astype(np.uint8)
                            video_masks[frame_idx] = binary_mask
                
                # Save Results
                self.save_results_for_video(video_path, video_masks, save_root)
                
                # Store in batch_data for visualization
                self.batch_data[video_path] = {
                    'masks': video_masks,
                    'annotations': {},
                    'history': [],
                    'text_prompt': text
                }
                
                # Cleanup
                self.segmenter.close()
                
            except Exception as e:
                print(f"Error processing {video_path}: {e}")
                # Continue to next
                
        self.status_var.set("Batch Processing Complete.")
        messagebox.showinfo("Success", f"Batch processing finished for {total_videos} videos.")
        
        # Restore UI state
        if original_video_path:
            self.load_video_source(original_video_path)
        else:
            self.video_masks = {}
            self.clear_all_annotations()
            self.canvas.delete("all")

    def save_results_for_video(self, video_path, masks, save_root, save_cutout=True, save_mask=True):
        if not masks: return 0
        
        # Determine output folder name
        if self.root_input_folder and video_path.startswith(self.root_input_folder):
            rel_path = os.path.relpath(video_path, self.root_input_folder)
            out_dir = os.path.join(save_root, rel_path)
        else:
            # Fallback
            folder_name = os.path.basename(video_path)
            if os.path.isfile(video_path):
                folder_name = os.path.splitext(folder_name)[0]
            out_dir = os.path.join(save_root, folder_name)
        
        os.makedirs(out_dir, exist_ok=True)
        
        # Get frame files
        current_frame_files = []
        if video_path == self.current_video_path:
            current_frame_files = self.frame_files
        elif os.path.isdir(video_path):
            valid_exts = ('.jpg',)
            files = [f for f in os.listdir(video_path) if f.lower().endswith(valid_exts)]
            # Use standard sort to ensure frame order matches backend
            files.sort()
            current_frame_files = [os.path.join(video_path, f) for f in files]
        else:
            # For video files not loaded, we can't easily get frames without extracting
            return 0

        saved_count = 0
        for i, frame_path in enumerate(current_frame_files):
            if i in masks:
                if not os.path.exists(frame_path): continue
                
                frame = cv2.imread(frame_path)
                if frame is None: continue
                
                mask = masks[i]
                if mask.shape != frame.shape[:2]:
                    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                
                original_name = os.path.basename(frame_path)
                name_stem = os.path.splitext(original_name)[0]
                
                if save_cutout:
                    # 1. Save Cutout
                    frame_bgra = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
                    frame_bgra[:, :, 3] = (mask * 255).astype(np.uint8)
                    
                    cutout_name = f"{name_stem}.png"
                    cv2.imwrite(os.path.join(out_dir, cutout_name), frame_bgra)
                
                if save_mask:
                    # 2. Save Mask
                    mask_name = f"{name_stem}.png"
                    cv2.imwrite(os.path.join(out_dir, mask_name), mask * 255)
                
                saved_count += 1
        return saved_count

    def save_all_masks(self):
        if not self.video_masks: return
        save_dir = filedialog.askdirectory(title="Select Save Directory")
        if not save_dir: return
        
        self.status_var.set("Saving masks...")
        for idx, mask in self.video_masks.items():
            # Use original filename if available
            if idx < len(self.frame_files):
                original_name = os.path.basename(self.frame_files[idx])
                name_stem = os.path.splitext(original_name)[0]
                
                mask_name = f"{name_stem}.png"
                cv2.imwrite(os.path.join(save_dir, mask_name), mask * 255)
                
        self.status_var.set("Masks saved.")
        messagebox.showinfo("Success", f"Saved {len(self.video_masks)} masks.")

    def save_batch_masks(self):
        # Ensure current state is saved to batch_data
        if self.current_video_path:
            self.batch_data[self.current_video_path] = {
                'masks': self.video_masks,
                'annotations': self.annotations,
                'history': self.operation_history,
                'text_prompt': self.text_prompt.get()
            }
            
        if not self.batch_data: return
        
        save_root = filedialog.askdirectory(title="Select Root Save Directory for Masks")
        if not save_root: return
        
        self.status_var.set("Saving batch masks...")
        total_saved = 0
        
        for video_path, data in self.batch_data.items():
            masks = data.get('masks', {})
            if not masks: continue
            
            print(f"Saving masks for {video_path}...")
            # Only save masks, skip cutouts
            count = self.save_results_for_video(video_path, masks, save_root, save_cutout=False, save_mask=True)
            total_saved += count
            self.root.update()
            
        self.status_var.set("Batch save complete.")
        messagebox.showinfo("Success", f"Saved {total_saved} masks across {len(self.batch_data)} folders.")

    def save_cutout_images(self):
        # Ensure current state is saved to batch_data
        if self.current_video_path:
            self.batch_data[self.current_video_path] = {
                'masks': self.video_masks,
                'annotations': self.annotations,
                'history': self.operation_history,
                'text_prompt': self.text_prompt.get()
            }
            
        if not self.batch_data: return
        
        save_root = filedialog.askdirectory(title="Select Root Save Directory for Cutouts")
        if not save_root: return
        
        self.status_var.set("Saving batch cutout images...")
        total_saved = 0
        
        for video_path, data in self.batch_data.items():
            masks = data.get('masks', {})
            if not masks: continue
            
            print(f"Saving for {video_path}...")
            # Only save cutouts, skip masks
            count = self.save_results_for_video(video_path, masks, save_root, save_cutout=True, save_mask=False)
            total_saved += count
            self.root.update()
            
        self.status_var.set("Batch save complete.")
        messagebox.showinfo("Success", f"Saved {total_saved} cutout images across {len(self.batch_data)} folders.")

    def save_all_cutouts(self):
        if not self.video_masks: return
        save_dir = filedialog.askdirectory(title="Select Save Directory for Cutouts")
        if not save_dir: return
        
        self.status_var.set("Saving cutouts...")
        for idx, mask in self.video_masks.items():
            # Use original filename if available
            if idx < len(self.frame_files):
                original_name = os.path.basename(self.frame_files[idx])
                name_stem = os.path.splitext(original_name)[0]
                
                frame_path = self.frame_files[idx]
                frame = cv2.imread(frame_path)
                if frame is None: continue
                
                if mask.shape != frame.shape[:2]:
                    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                
                # Create cutout
                frame_bgra = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
                frame_bgra[:, :, 3] = (mask * 255).astype(np.uint8)
                
                cutout_name = f"{name_stem}.png"
                cv2.imwrite(os.path.join(save_dir, cutout_name), frame_bgra)
                
        self.status_var.set("Cutouts saved.")
        messagebox.showinfo("Success", f"Saved {len(self.video_masks)} cutouts.")

def main():
    parser = argparse.ArgumentParser(description="Interactively segment human hands with SAM 3.")
    parser.add_argument("--checkpoint", type=Path,
                        default=Path(__file__).resolve().parent.parent / "weights" / "sam3.pt")
    args = parser.parse_args()
    root = tk.Tk()
    app = SAM3VideoGUI(root, checkpoint=args.checkpoint)
    root.mainloop()

if __name__ == "__main__":
    main()
