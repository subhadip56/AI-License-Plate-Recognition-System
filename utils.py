"""
utils.py

This file contains helper functions for the ALPR system.
- enhance(): Pre-processes video frames to improve model accuracy.
- clean_text(): Cleans raw OCR output for a global (non-specific) format.
"""

import cv2
import re

def enhance(frame):
    """
    Applies image enhancement to a video frame for better OCR/detection.
    Converts to grayscale and uses CLAHE (Contrast Limited Adaptive
    Histogram Equalization) for high-contrast, shadow-reduced output.
    
    Args:
        frame: The input color frame (NumPy array).
        
    Returns:
        The enhanced frame (NumPy array), still in 3-channel BGR format.
    """
    if frame is None:
        return None
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Use CLAHE for better contrast enhancement than simple equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)
        
        # Convert back to BGR so it can be processed by models expecting 3 channels
        return cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
    except cv2.error:
        # Fallback if conversion fails
        return frame

def clean_text(text: str) -> str:
    """
    Cleans raw OCR text for a global format.
    Removes all non-alphanumeric characters and converts to uppercase.
    
    Args:
        text (str): The raw text output from EasyOCR.
        
    Returns:
        str: The cleaned, uppercase text.
    """
    if not text:
        return ""
    # This regex keeps only letters (A-Z) and numbers (0-9)
    return re.sub(r'[^A-Z0-9]', '', text).upper()