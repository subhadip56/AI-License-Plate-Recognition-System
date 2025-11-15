"""
streamlit_app.py

Streamlit interface for the ALPR system.

Features:
- Upload a video
- Show processing progress
- Download the processed video
"""

import streamlit as st
from detector import SimpleALPR
import os
import tempfile


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
if not os.path.exists(MODEL_PATH):
    st.error(
        f"FATAL ERROR: Model file not found at '{MODEL_PATH}'. "
        "Please upload 'best.torchscript' to the 'models' folder."
    )
    st.stop()


# --- File Upload ---
video_file = st.file_uploader(
    "Select or drag-and-drop a video file",
    type=VIDEO_TYPES
)


if video_file:

    # Save uploaded file to a temp path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp:
        temp.write(video_file.read())
        input_video_path = temp.name

    st.info("Processing your video... Please wait.")

    # Progress components
    progress_bar = st.progress(0)
    progress_text = st.empty()

    def progress_cb(current_frame, total_frames):
        """Callback updating the progress bar."""
        if total_frames > 0:
            percent = current_frame / total_frames
            progress_bar.progress(percent)
            progress_text.text(
                f"Processing frame {current_frame}/{total_frames} "
                f"({int(percent * 100)}%)"
            )
        else:
            progress_text.text(f"Processing frame {current_frame}...")

    try:
        # Initialize ALPR processor
        alpr = SimpleALPR(MODEL_PATH)

        with st.spinner("Running detection, tracking, and OCR..."):
            output_path, _ = alpr.process_video(input_video_path, cb=progress_cb)

        st.success("🎉 Processing Complete!")
        progress_text.empty()

        st.subheader("Download Processed Video")

        with open(output_path, "rb") as f:
            st.download_button(
                label="⬇️ Download Annotated Video",
                data=f,
                file_name="processed_video.mp4",
                mime="video/mp4"
            )

    except Exception as e:
        st.error(f"Error during processing: {e}")
        print("ERROR:", e)

    finally:
        if os.path.exists(input_video_path):
            os.remove(input_video_path)


# --- Footer ---
st.markdown(
    """
    <div style="text-align: center; opacity: 0.6; padding-top: 30px; font-size: 14px;">
        Made with ❤️ by <b>Subhadip Malakar</b>
    </div>
    """,
    unsafe_allow_html=True
)
