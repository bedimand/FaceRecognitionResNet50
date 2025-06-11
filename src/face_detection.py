#!/usr/bin/env python3
import cv2
import os
import re  # for extracting sequence numbers
import numpy as np
from insightface.app import FaceAnalysis
import torch
from src.train_model import load_config

# from src.preprocessing import preprocess_face  # Not used anymore

def initialize_face_analyzer():
    """
    Initialize the InsightFace analyzer for face detection and landmark detection
    """
    # No need to check for model_dir or instruct to run download_models.py
    try:
        # Initialize the FaceAnalysis application with specific configs
        model_dir = "models/insightface"  # Still used for root, but don't check existence
        app = FaceAnalysis(name="buffalo_s", root=os.path.dirname(model_dir), 
                          allowed_modules=['detection', 'landmark_2d_106'])
        # Choose GPU if available, else CPU
        ctx_id = 0 if torch.cuda.is_available() else -1
        app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        print("InsightFace model loaded successfully!")
        return app
    except Exception as e:
        print(f"Error initializing InsightFace: {e}")
        print("Make sure you have installed insightface and onnxruntime packages")
        return None

def detect_faces(frame, face_analyzer):
    """
    Detect faces using InsightFace
    
    Args:
        frame: Input frame
        face_analyzer: InsightFace FaceAnalysis object
        
    Returns:
        List of face objects containing detection and landmark data
    """
    if face_analyzer is None:
        return []
    
    # Detect faces (returns a list of Face objects with bbox, kps, landmark etc.)
    faces = face_analyzer.get(frame)
    
    return faces


def center_crop_around_nose(frame, face, crop_size=(128, 128)):
    """
    Crop the image to include the entire face but centered around the nose
    with a tighter focus on the face and less background
    
    Args:
        frame: Input frame
        face: InsightFace face object with landmarks
        crop_size: Tuple of (width, height) for the final image
        
    Returns:
        Cropped face image centered on the nose but showing the entire face
    """
    # Get frame dimensions
    frame_h, frame_w = frame.shape[:2]
    
    # Get face bounding box
    bbox = face.bbox.astype(int)
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    
    # Calculate face dimensions
    face_width = x2 - x1
    face_height = y2 - y1
    
    # Get nose position from keypoints
    if hasattr(face, 'kps') and face.kps is not None:
        nose = face.kps[2]  # Nose is the third keypoint in InsightFace
        nose_x, nose_y = int(nose[0]), int(nose[1])
        
        # Get eye positions to better estimate face size
        left_eye = face.kps[0]
        right_eye = face.kps[1]
        eye_distance = np.sqrt((left_eye[0] - right_eye[0])**2 + (left_eye[1] - right_eye[1])**2)
        
        # Use eye distance to determine crop size (typically faces are ~2.5-3x eye distance in width)
        face_scale = 2.7  # This controls how tight the crop will be (lower = tighter)
        ideal_size = int(eye_distance * face_scale)
    else:
        # If no keypoints, estimate nose position from the center of bounding box
        nose_x = (x1 + x2) // 2
        nose_y = (y1 + y2) // 2
        
        # Fall back to using face bbox dimensions with a small margin
        ideal_size = max(face_width, face_height)
        
    # Calculate square region around nose using the ideal size
    half_size = ideal_size // 2
    square_x1 = max(0, nose_x - half_size)
    square_y1 = max(0, nose_y - half_size)
    square_x2 = min(frame_w, nose_x + half_size)
    square_y2 = min(frame_h, nose_y + half_size)
    
    # Check if we hit the image boundaries and adjust
    if square_x1 == 0:
        square_x2 = min(frame_w, square_x1 + ideal_size)
    if square_y1 == 0:
        square_y2 = min(frame_h, square_y1 + ideal_size)
    if square_x2 == frame_w:
        square_x1 = max(0, square_x2 - ideal_size)
    if square_y2 == frame_h:
        square_y1 = max(0, square_y2 - ideal_size)
    
    # Crop the square region
    if square_x2 <= square_x1 or square_y2 <= square_y1:
        # Handle invalid crop (shouldn't happen but just in case)
        face_crop = frame[y1:y2, x1:x2]
    else:
        face_crop = frame[square_y1:square_y2, square_x1:square_x2]
    
    # Resize to target size
    if face_crop.size > 0:  # Check if crop is not empty
        face_resized = cv2.resize(face_crop, crop_size)
    else:
        # If crop failed, use original face bbox
        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size > 0:
            face_resized = cv2.resize(face_crop, crop_size)
        else:
            # Last resort fallback
            face_resized = np.zeros((crop_size[1], crop_size[0], 3), dtype=np.uint8)
    
    return face_resized


def extract_face(frame, face):
    """
    Extract face from frame using bounding-box crop.
    Args:
        frame: Input frame
        face: InsightFace face object
    Returns:
        Cropped and resized face image
    """
    # Load target size from config
    cfg = load_config()
    target_size = tuple(cfg.get('image_processing.target_size', [224, 224]))
    # Clamp bounding-box coordinates
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = face.bbox.astype(int)
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))
    # If invalid box, fallback to center_crop_around_nose
    if x2 <= x1 or y2 <= y1:
        return center_crop_around_nose(frame, face, crop_size=target_size)
    # Perform bounding-box crop
    face_crop = frame[y1:y2, x1:x2]
    if face_crop.size == 0:
        return center_crop_around_nose(frame, face, crop_size=target_size)
    # Resize to target size
    return cv2.resize(face_crop, target_size)


def save_face(frame, face, output_dir, person_name=None, collection_ts=None):
    """
    Save a detected face as a raw cropped image (no preprocessing)
    Args:
        frame: Source frame
        face: InsightFace face object
        output_dir: Base directory to save faces
        person_name: Optional name of the person (for creating subdirectory)
        collection_ts: Optional timestamp for naming the file
    Returns:
        Path to saved face image
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    save_dir = output_dir

    # Get cropped face centered around nose
    face_img = extract_face(frame, face)

    # Determine the next sequential number by extracting leading digits
    existing_files = [f for f in os.listdir(save_dir) if f.endswith('.png')]
    numbers = []
    for fname in existing_files:
        m = re.search(r"(\d+)", fname)
        if m:
            try:
                numbers.append(int(m.group(1)))
            except:
                pass
    next_num = max(numbers) + 1 if numbers else 1
    # Build filename: sequence-person-timestamp if provided, else default face_{num}
    if collection_ts:
        # Use folder basename as default label if none
        label = person_name if person_name else os.path.basename(os.path.normpath(output_dir))
        filename = os.path.join(save_dir, f"{next_num:03d}-{label}-{collection_ts}.png")
    else:
        filename = os.path.join(save_dir, f"face_{next_num:03d}.png")
    # Save image
    cv2.imwrite(filename, face_img)
    return filename 