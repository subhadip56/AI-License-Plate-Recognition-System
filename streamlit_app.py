"""
streamlit_app.py

This file creates the web interface for the ALPR system using Streamlit.
It allows a user to:
- Upload a video file.
- See a progress bar as the video is processed.
- View the final annotated video output.
- Features a custom "frosted glass" UI with an embedded background.
"""

import streamlit as st
from detector import SimpleALPR
import os
import tempfile
import pandas as pd
import base64 # Import base64

# --- Page Configuration ---
st.set_page_config(
    page_title="Global ALPR",  # Sets the browser tab title
    page_icon="🚗",             # Sets the browser tab icon
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- ★★★ EMBEDDED BACKGROUND IMAGE & STYLING ★★★ ---

# NOTE: A full base64 string is too long to paste here.
# 1. Go to https://www.base64-image.de/
# 2. Upload a dark, abstract background image you like.
# 3. Copy the (long) string it generates and paste it inside the quotes below.
#
# I have used a simple gradient as a placeholder.
# Replace the `background-image` line with:
# background-image: url("data:image/jpeg;base64,PASTE_YOUR_LONG_STRING_HERE");

PAGE_STYLING = f"""
    <style>
    /* Main app container */
    [data-testid="stAppViewContainer"] > .main {{
        /* Fallback background color */
        background-color: #0a0a0a; 
        
        /* This is the placeholder gradient. Replace this! */
        background-image: linear-gradient(180deg, #1E1E1E, #0a0a0a);
        
        /* --- UNCOMMENT THIS LINE AND PASTE YOUR BASE64 STRING ---
        background-image: url("data:image/jpeg;base64,YOUR_BASE64_STRING_GOES_HERE");
        */
        
        background-size: cover;
        background-attachment: fixed;
    }}
    
    /* Main content area styling ("frosted glass") */
    .st-emotion-cache-16txtl3 {{
        background-color: rgba(30, 30, 30, 0.65); /* 65% opaque dark grey */
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }}

    /* Title text */
    [data-testid="stTitle"] {{
        color: #FAFAFA;
        text-align: center;
    }}
    
    /* Subheader text */
    .st-emotion-cache-16txtl3 p {{
        color: #A0A0A0;
        text-align: center;
    }}

    /* Video player */
    [data-testid="stVideo"] {{
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }}

    /* Hide default Streamlit branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    
    </style>
    """
st.markdown(PAGE_STYLING, unsafe_allow_html=True)


# --- App Title and Description ---
# Use columns to center the title and uploader
_, col2, _ = st.columns([1, 2, 1])

with col2:
    st.title("Global License Plate Recognition System 🌍")
    st.write(
        "Upload a video to detect, track, and read license plates. "
        "The system uses YOLOv8, DeepSORT, and EasyOCR."
    )

    # --- Constants ---
    VIDEO_TYPES = ["mp4", "avi", "mov", "mkv"]
    MODEL_PATH = "models/best.torchscript"

    # --- Model Check ---
    if not os.path.exists(MODEL_PATH):
        st.error(f"FATAL ERROR: Model file not found at '{MODEL_PATH}'.")
        st.stop()

    # --- File Uploader ---
    video_file = st.file_uploader(
        "Upload your video file",
        type=VIDEO_TYPES,
        label_visibility="collapsed"
    )

    if video_file:
        # We need to keep track of both temp files
        input_video_path = None
        output_video_path = None
        
        try:
            # Save uploaded file to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_in:
                temp_in.write(video_file.read())
                input_video_path = temp_in.name

            st.info("Processing video... This may take a moment.")

            progress_bar = st.progress(0)
            progress_text = st.empty()

            def progress_cb(current_frame, total_frames):
                """Callback function to update Streamlit's progress bar."""
                if total_frames > 0:
                    percent = current_frame / total_frames
                    progress_bar.progress(percent)
                    progress_text.text(f"Processing frame {current_frame} of {total_frames} "
                                       f"({int(percent * 100)}%)")
                else:
                    progress_text.text(f"Processing frame {current_frame}...")

            
            alpr = SimpleALPR(MODEL_PATH)

            with st.spinner("Running detection, tracking, and OCR..."):
                # The detector returns the path to its *own* temp file
                output_video_path, _ = alpr.process_video(input_video_path, cb=progress_cb)

            st.success("🎉 Processing Complete!")
            progress_text.empty() 
            
            st.subheader("Annotated Video Output")
            
            # --- ★★★ CHANGE: Read the video file as bytes ★★★ ---
            # This is the fix for Streamlit Cloud
            if os.path.exists(output_video_path):
                video_f = open(output_video_path, 'rb')
                video_bytes = video_f.read()
                st.video(video_bytes) # Pass the bytes, not the path
                video_f.close()
            else:
                st.error("Error: Processed video file could not be found.")
            # --- ★★★ END OF CHANGE ★★★ ---

        except Exception as e:
            st.error(f"An error occurred during processing: {e}")
            print(f"Error details: {e}")
        
        finally:
            # --- Clean up BOTH temporary files ---
            if input_video_path and os.path.exists(input_video_path):
                os.remove(input_video_path)
            if output_video_path and os.path.exists(output_video_path):
                os.remove(output_video_path)