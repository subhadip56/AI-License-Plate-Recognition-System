"""
streamlit_app.py

This file creates the web interface for the ALPR system using Streamlit.
It allows a user to:
- Upload a video file.
- See a progress bar as the video is processed.
- View the final annotated video output.
"""

import streamlit as st
from detector import SimpleALPR
import os
import tempfile
import pandas as pd

# --- Page Configuration ---
st.set_page_config(
    page_title="Global ALPR System",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("Global License Plate Recognition System 🌍")
st.write("Upload a video to detect, track, and read license plates in real-time.")

# --- Constants ---
VIDEO_TYPES = ["mp4", "avi", "mov", "mkv"]
MODEL_PATH = "models/best.torchscript"

# --- Model Check ---
# Verify that the exported model file exists
if not os.path.exists(MODEL_PATH):
    st.error(f"FATAL ERROR: Model file not found at '{MODEL_PATH}'. "
             "Please ensure 'best.torchscript' is in the 'models' folder.")
    st.stop()

# --- File Uploader ---
video_file = st.file_uploader(
    "Drag and drop a video file",
    type=VIDEO_TYPES
)

if video_file:
    # Save uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp:
        temp.write(video_file.read())
        video_path = temp.name

    st.info("Processing video... This may take a moment.")

    # --- Progress Bar & Status ---
    progress_bar = st.progress(0)
    progress_text = st.empty()

    def progress_cb(current_frame, total_frames):
        """Callback function to update Streamlit's progress bar."""
        if total_frames > 0:
            percent_complete = current_frame / total_frames
            progress_bar.progress(percent_complete)
            progress_text.text(f"Processing frame {current_frame} of {total_frames} "
                               f"({int(percent_complete * 100)}%)")
        else:
            # For streams or videos where total_frames is unknown
            progress_text.text(f"Processing frame {current_frame}...")

    try:
        # --- Run the ALPR Pipeline ---
        alpr = SimpleALPR(MODEL_PATH)

        with st.spinner("Running detection, tracking, and OCR..."):
            # The 'plates' list is returned but not used, per request
            output_path, _ = alpr.process_video(video_path, cb=progress_cb)

        st.success("🎉 Processing Complete!")
        progress_text.empty() # Clear progress text
        
        # --- Display Final Video ---
        st.subheader("Annotated Video Output")
        st.video(output_path)

    except Exception as e:
        st.error(f"An error occurred during processing: {e}")
        print(f"Error details: {e}") # Also print to console for debugging
    
    finally:
        # Clean up the temporary video file
        if os.path.exists(video_path):
            os.remove(video_path)