"""
image_utils.py
==============
Utilities for loading PNG chest X-ray images.
Replaces dicom_utils.py — no pydicom needed.
"""

import os
from PIL import Image


def load_png(image_path: str) -> Image.Image:
    """
    Load a PNG image and return as PIL RGB image.
    Falls back to a blank image if the file is missing or corrupt.
    """
    try:
        img = Image.open(image_path).convert("RGB")
        return img
    except Exception:
        return Image.new("RGB", (224, 224))


def build_image_index(train_dir: str) -> dict:
    """
    Scan train_dir for all .png files and return:
        { image_id (str, no extension): full_path (str) }
    """
    index = {}
    try:
        for entry in os.scandir(train_dir):
            if entry.is_file() and entry.name.lower().endswith(".png"):
                image_id = os.path.splitext(entry.name)[0]
                index[image_id] = entry.path
    except FileNotFoundError:
        print(f"[WARNING] train_dir not found: {train_dir}")
    return index
