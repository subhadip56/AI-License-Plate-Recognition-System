"""
detector.py

This is the core logic file for the ALPR system. It contains the SimpleALPR
class which handles:
- Loading the YOLO and EasyOCR models.
- Initializing the DeepSORT tracker.
- Processing a video frame-by-frame.
- Running detection, tracking, and OCR.
- Applying visual effects (spotlight, HUD, enlarged preview).
- Saving the final annotated video to a temporary file.
"""

import warnings
import cv2
from ultralytics import YOLO
import easyocr
from utils import enhance, clean_text
from deep_sort_realtime.deepsort_tracker import DeepSort
import torch
import os
import time
import numpy as np
import tempfile  # <-- IMPORT tempfile

# --- Suppress specific warnings for a cleaner console output ---
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="'pin_memory' argument is set as true but not supported on MPS",
    category=UserWarning,
)

# --- Global Device Auto-Detection ---
def get_best_device():
    """
    Auto-detects and returns the best available compute device (GPU or CPU).
    Prioritizes CUDA, then MPS (Apple Silicon), then CPU.
    """
    if torch.cuda.is_available():
        print("[INFO] Using CUDA (NVIDIA GPU)")
        return 'cuda'
    if torch.backends.mps.is_available():
        print("[INFO] Using MPS (Apple Silicon GPU)")
        return 'mps'
    print("[INFO] Using CPU")
    return 'cpu'

DEVICE = get_best_device()

# --- Main ALPR Class ---
class SimpleALPR:
    """
    A class to encapsulate the entire ALPR (Automatic License Plate Recognition)
    pipeline, from model loading to video processing.
    """
    def __init__(self, model_path="models/best.torchscript"):
        """
        Initializes the ALPR system.
        
        Args:
            model_path (str): Path to the exported .torchscript YOLO model.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        try:
            # Load the exported YOLO model
            self.model = YOLO(model_path, task="detect")
            print(f"[INFO] YOLO model loaded. Will run on {DEVICE} during inference.")
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model: {e}")

        # Load EasyOCR (use GPU if available)
        use_gpu = (DEVICE != 'cpu')
        self.reader = easyocr.Reader(["en"], gpu=use_gpu, verbose=False)
        print(f"[INFO] EasyOCR loaded (GPU: {use_gpu}).")

        # Initialize DeepSORT tracker with optimized parameters
        self.tracker = DeepSort(max_age=30,           # Max frames to keep a track without detection
                                n_init=3,             # Min. detections to start a track
                                nms_max_overlap=1.0,  # Max. overlap for non-max suppression
                                max_cosine_distance=0.2)
        
        # In-memory store for stable OCR results (Track ID -> [list of texts])
        self.ocr_memory = {}

    def _get_best_plate(self, track_id):
        """
        Performs a majority vote from the OCR memory for a given track ID.
        This stabilizes flickering OCR results.
        
        Args:
            track_id (int): The track ID to get the best plate for.
            
        Returns:
            str: The most frequently seen (and thus most stable) plate text.
        """
        if track_id in self.ocr_memory and self.ocr_memory[track_id]:
            texts = self.ocr_memory[track_id]
            # Find and return the most common text in the list
            return max(set(texts), key=texts.count)
        return ""

    def process_video(self, video_path, cb=None):
        """
        Processes an entire video file frame by frame, applying detection,
        tracking, OCR, and annotations.
        
        Args:
            video_path (str): The path to the input video file.
            cb (callable, optional): A callback function for updating Streamlit's
                                     progress bar, e.g., cb(current_frame, total_frames).
                                     
        Returns:
            tuple: (output_path, list_of_plates)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # --- ★★★ CHANGE: Create a Temporary File for the output video ★★★ ---
        # We no longer save to the 'Outputs/' folder.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_out:
            output_path = temp_out.name
        # --- ★★★ END OF CHANGE ★★★ ---
        
        # Use 'avc1' (H.264) for web compatibility, fall back to 'mp4v'
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))
        if not out.isOpened():
            print("[WARNING] avc1 codec failed. Falling back to mp4v.")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))
        
        frame_index = 0

        # --- Main processing loop ---
        while True:
            ret, frame = cap.read()
            if not ret:
                break # End of video
            
            frame_index += 1
            # We draw on the original color frame
            annotated_frame = frame.copy()
            # We detect on an enhanced (grayscale, high-contrast) frame
            enhanced_frame = enhance(frame)
            
            # Run YOLO model
            results = self.model(enhanced_frame, verbose=False, device=DEVICE)[0]
            
            # Prepare detections for DeepSORT
            detections = []
            if hasattr(results, "boxes") and results.boxes is not None:
                for box in results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    class_name = self.model.names.get(cls_id, "unknown")
                    
                    # Filter for 'license_plate' class with a confidence threshold
                    if class_name.lower() == "license_plate" and conf > 0.3:
                        w, h = x2 - x1, y2 - y1
                        detections.append(([x1, y1, w, h], conf, class_name))

            # Update tracker with current detections
            tracks = self.tracker.update_tracks(detections, frame=annotated_frame)
            
            # Process each active track
            for track in tracks:
                if not track.is_confirmed():
                    continue # Skip tracks that are not yet confirmed

                track_id = track.track_id
                bbox = track.to_tlbr() # Get [x1, y1, x2, y2]
                x1, y1, x2, y2 = map(int, bbox)
                
                # --- FEATURE: Plate Spotlight Effect ---
                # 1. Clip coordinates to be safely inside the frame
                y1_c = max(0, y1)
                y2_c = min(frame_h, y2)
                x1_c = max(0, x1)
                x2_c = min(frame_w, x2)
                
                # 2. Get the crop from the *original color frame*
                color_crop = annotated_frame[y1_c:y2_c, x1_c:x2_c]
                
                if color_crop.size > 0:
                    # 3. Add brightness to create the "spotlight"
                    brightness = 40 # Brightness intensity
                    bright_matrix = np.ones(color_crop.shape, dtype="uint8") * brightness
                    
                    # Use cv2.add for safe addition (clips at 255)
                    brightened_crop = cv2.add(color_crop, bright_matrix)
                    
                    # 4. Paste the brightened crop back onto the frame
                    annotated_frame[y1_c:y2_c, x1_c:x2_c] = brightened_crop
                
                # --- OCR (on enhanced frame for better accuracy) ---
                crop_ocr = enhanced_frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
                raw_text = ""
                if crop_ocr.size > 0:
                    ocr_res = self.reader.readtext(crop_ocr, detail=0)
                    if ocr_res:
                        raw_text = ocr_res[0]

                # Clean the text using global rules
                cleaned_text = clean_text(raw_text)
                
                # Add to memory for majority voting
                if cleaned_text:
                    if track_id not in self.ocr_memory:
                        self.ocr_memory[track_id] = []
                    self.ocr_memory[track_id].append(cleaned_text)
                
                # Get the most stable text for this track
                display_text = self._get_best_plate(track_id)
                
                # --- Draw the green bounding box (on top of the spotlight) ---
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # --- Draw the Floating HUD Label ---
                if display_text:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.9
                    thickness = 2
                    
                    (text_w, text_h), baseline = cv2.getTextSize(display_text, font, font_scale, thickness)
                    
                    # Center the label horizontally above the green box
                    label_x = x1 + (x2 - x1) // 2 - text_w // 2
                    label_y = y1 - 20 # Position 20px above the green box
                    
                    # --- Clip HUD box to stay within frame boundaries ---
                    if label_y - text_h < 10:
                        label_y = y2 + text_h + 20 # Move it below the plate
                    
                    if label_x < 0: label_x = 0
                    if label_x + text_w > frame_w: label_x = frame_w - text_w
                    
                    # Get background coordinates
                    bg_x1, bg_y1 = label_x - 10, label_y - text_h - (baseline // 2) - 5
                    bg_x2, bg_y2 = label_x + text_w + 10, label_y + (baseline // 2) + 5
                    bg_x1 = max(0, bg_x1); bg_y1 = max(0, bg_y1)
                    bg_x2 = min(frame_w, bg_x2); bg_y2 = min(frame_h, bg_y2)

                    # Draw semi-transparent background for text
                    overlay = annotated_frame.copy()
                    cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
                    alpha = 0.6 # 60% opacity
                    cv2.addWeighted(overlay, alpha, annotated_frame, 1 - alpha, 0, annotated_frame)
                    
                    # Draw the white text
                    cv2.putText(annotated_frame, display_text, (label_x, label_y), 
                                font, font_scale, (255, 255, 255), thickness)

                    # --- FEATURE: Enlarged Preview Window ---
                    if color_crop.size > 0:
                        crop_h, crop_w = color_crop.shape[:2]
                        # Scale the preview window
                        scale_factor = 2.5
                        preview_w = int(max(180, crop_w * scale_factor))
                        preview_h = int(max(80, crop_h * scale_factor))
                        
                        # Constrain preview size
                        preview_w = min(preview_w, frame_w - 20)
                        preview_h = min(preview_h, frame_h - 20)
                        
                        if preview_w > 0 and preview_h > 0:
                            # Resize the *original color* crop
                            enlarged = cv2.resize(color_crop, (preview_w, preview_h), interpolation=cv2.INTER_CUBIC)

                            # Calculate a stable position for the preview
                            preview_x = max(10, min(frame_w - preview_w - 10, x1 - preview_w // 2))
                            preview_y = y1 - preview_h - 30 # Position above the HUD
                            
                            # If it goes off-screen, move it below
                            if preview_y < 10:
                                preview_y = max(10, min(frame_h - preview_h - 10, y2 + 30))

                            # Paste the enlarged preview onto the frame
                            overlay_roi = annotated_frame[preview_y:preview_y + preview_h, preview_x:preview_x + preview_w]
                            if overlay_roi.shape[:2] == enlarged.shape[:2]:
                                overlay_roi[:] = enlarged
                                # Add border and text
                                cv2.rectangle(annotated_frame, (preview_x, preview_y), (preview_x + preview_w, preview_y + preview_h), (0, 255, 0), 2)
                                text_pos = (preview_x + 12, preview_y + preview_h - 12)
                                cv2.putText(annotated_frame, display_text, text_pos, font, 0.9, (255, 255, 255), 2)
                # --- End of HUD & Preview ---

            # Clean up old tracks from OCR memory
            active_track_ids = {t.track_id for t in tracks}
            for tid in list(self.ocr_memory.keys()):
                if tid not in active_track_ids:
                    del self.ocr_memory[tid]
            
            # Write the annotated frame to the output video
            out.write(annotated_frame)

            # Update Streamlit progress bar if callback is provided
            if cb:
                cb(frame_index, total_frames)

        # Release all resources
        cap.release()
        out.release() 
        self.ocr_memory.clear()

        # Return the path to the saved video
        return output_path, []