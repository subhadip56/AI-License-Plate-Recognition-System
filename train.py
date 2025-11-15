"""
train.py

YOLOv8 Training & Fine-Tuning Script
This script provides a reusable, command-line interface for fine-tuning
a YOLOv8 model on a custom dataset from Roboflow.

It handles:
1. Google Drive mounting (if in Colab).
2. Dependency installation.
3. Roboflow API key validation.
4. Dataset download and setup.
5. YOLOv8 model training with specified parameters.

Example Usage (from terminal):
    python train.py --api-key YOUR_API_KEY --epochs 50 --project /path/to/outputs
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional


def _running_in_colab() -> bool:
    """Checks if the script is running in a Google Colab environment."""
    try:
        import google.colab
        return True
    except ImportError:
        return False

def _mount_drive_if_available(mount_point: str = "/content/drive") -> None:
    """Mounts Google Drive at the specified path if running in Colab."""
    if not _running_in_colab():
        print("[INFO] Not in Colab. Skipping Google Drive mount.")
        return

    from google.colab import drive  #type: ignore
    print("[INFO] In Colab. Mounting Google Drive...")
    drive.mount(mount_point, force_remount=False)
    print(f"[INFO] Google Drive mounted at {mount_point}")

# --- Dependency Management ---

def _ensure_packages_installed() -> None:
    """
    Ensures the required Python packages (roboflow, ultralytics)
    are installed using pip.
    """
    required = ["roboflow", "ultralytics"]
    print(f"[INFO] Ensuring required packages are installed: {', '.join(required)}")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", *required]
        )
        print("[INFO] Dependencies verified.")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to install dependencies: {e}", file=sys.stderr)
        sys.exit(1)

# --- API Key Resolution ---

def _resolve_api_key(explicit_key: Optional[str]) -> str:
    """
    Resolves the Roboflow API key.
    Prefers the key passed via --api-key, falls back to ROBOFLOW_API_KEY
    environment variable.
    
    Args:
        explicit_key: The API key provided as a command-line argument.
        
    Returns:
        The validated API key.
        
    Raises:
        RuntimeError: If no API key is found.
    """
    key = explicit_key or os.getenv("ROBOFLOW_API_KEY")
    if not key:
        raise RuntimeError(
            "Roboflow API key not provided. "
            "Pass with --api-key or set ROBOFLOW_API_KEY."
        )
    return key

# --- Core Training Function ---

def train_yolov8(
    *,
    api_key: Optional[str],
    project_dir: Path,
    epochs: int,
    imgsz: int,
    batch: int,
    freeze: int,
    base_weights: str,
) -> None:
    """
    Executes the main training pipeline.
    
    Args:
        api_key: The Roboflow API key.
        project_dir: The directory to save training outputs (runs, weights).
        epochs: The number of epochs to train for.
        imgsz: The input image size (e.g., 640).
        batch: The batch size for training.
        freeze: Number of backbone layers to freeze.
        base_weights: The starting model checkpoint (e.g., 'yolov8s.pt').
    """
    from roboflow import Roboflow
    from ultralytics import YOLO

    project_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Training artifacts will be stored in: {project_dir}")

    # 1. Download Dataset
    key = _resolve_api_key(api_key)
    print("[INFO] Initializing Roboflow client...")
    rf = Roboflow(api_key=key)
    project = rf.workspace("roboflow-universe-projects").project("license-plate-recognition-rxg4e")
    dataset = project.version(11).download("yolov8")
    data_yaml = Path(dataset.location) / "data.yaml"
    print(f"[INFO] Dataset downloaded. YAML configuration at: {data_yaml}")

    # 2. Load Base Model
    print(f"[INFO] Loading base YOLOv8 checkpoint: {base_weights}")
    model = YOLO(base_weights)

    # 3. Start Training
    print(
        f"[INFO] Starting fine-tuning run: "
        f"epochs={epochs} batch={batch} imgsz={imgsz} freeze={freeze}"
    )
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=0,  
        project=str(project_dir),
        name="alpr_yolov8s_finetune",
        freeze=freeze,
        patience=20 
    )

    print("[INFO] Training completed successfully.")
    print(f"[INFO] Model and results saved to: {results.save_dir}")
    print(f"[INFO] Best weights at: {results.save_dir / 'weights/best.pt'}")

# --- Argument Parsing & Main Execution ---

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parses command-line arguments for the training script."""
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 for ALPR detection")
    
    # Dataset & API
    parser.add_argument(
        "--api-key",
        dest="api_key",
        default=None,
        help="Roboflow API key (optional if ROBOFLOW_API_KEY env var is set)"
    )
    parser.add_argument(
        "--project",
        default="/content/drive/MyDrive/ALPR_Project",
        help="Directory where training outputs (runs/) will be written"
    )
    
    # Training Hyperparameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Training image size (square)"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size per iteration"
    )
    parser.add_argument(
        "--freeze",
        type=int,
        default=10,
        help="Number of model layers to freeze (first N layers)"
    )
    parser.add_argument(
        "--weights",
        default="yolov8s.pt",
        help="Path or alias of the base YOLOv8 checkpoint to fine-tune"
    )
    
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    """Main execution function."""
    args = parse_args(argv)
    
    _mount_drive_if_available("/content/drive")
    _ensure_packages_installed()
    
    train_yolov8(
        api_key=args.api_key,
        project_dir=Path(args.project),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        freeze=args.freeze,
        base_weights=args.weights,
    )


if __name__ == "__main__":
    main()