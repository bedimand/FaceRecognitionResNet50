#!/usr/bin/env python3
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image
import time
import threading
import queue

from src.face_detection import initialize_face_analyzer, detect_faces, extract_face, save_face
from src.face_recognition import build_embedding_model, build_targets_db
from src.train_model import ClassifierNet, load_config

# Build transform for embeddings
transform_embed = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

def main():
    cfg = load_config()
    device = torch.device(cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))

    # Initialize face detector
    face_analyzer = initialize_face_analyzer()
    # Load embedding model
    model = build_embedding_model(cfg, device)
    # Precompute target embeddings
    embeddings_db, class_names = build_targets_db(model, cfg, device)
    # Base directory for target images
    base_targets_dir = cfg.get('targets_dir', 'targets')
    threshold = float(cfg.get('threshold', 1.0))

    # Setup for background target collection
    frame_lock = threading.Lock()
    last_frame = None
    last_faces = []
    rebuild_queue = queue.Queue()

    # Start video capture
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    if not cap.isOpened():
        print('Cannot open camera')
        return

    print("Starting real-time recognition. Press 'c' to capture, 'q' to quit, 'a' to add target.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect faces first
        faces = detect_faces(frame, face_analyzer)
        # Update shared frame and faces for background thread
        with frame_lock:
            last_frame = frame.copy()
            last_faces = faces.copy()
        # Overlay user instructions onto the frame
        cv2.putText(frame, "Press 'c' to capture, 'q' to quit, 'a' to add target", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

        for face in faces:
            # Bounding box
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            # Crop and preprocess face
            face_crop = extract_face(frame, face)
            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            face_pil = Image.fromarray(face_rgb)
            face_tensor = transform_embed(face_pil).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = model.embed(face_tensor)
                emb = F.normalize(emb, p=2, dim=1)
            # Compute distances to target embeddings
            db = embeddings_db.to(device)
            dists = torch.norm(db - emb, dim=1).cpu().numpy()
            idx = np.argmin(dists)
            dist_min = dists[idx]
            if dist_min < threshold:
                name = class_names[idx]
            else:
                name = 'Unknown'
            # Draw box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f'{name} {dist_min:.2f}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        cv2.imshow('Real-Time Recognition', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('a'):
            # Start background collection thread
            def background_collect():
                person_name = input("Enter new target name: ").strip()
                if not person_name:
                    print("Empty name, aborting.")
                    return
                count_str = input("How many images to collect? ").strip()
                try:
                    count = int(count_str)
                except ValueError:
                    print("Invalid count, aborting.")
                    return
                out_dir = os.path.join(base_targets_dir, person_name)
                os.makedirs(out_dir, exist_ok=True)
                collected_bg = 0
                print(f"Background collecting {count} images for '{person_name}'")
                while collected_bg < count:
                    time.sleep(0.1)
                    with frame_lock:
                        frame_bg = last_frame.copy() if last_frame is not None else None
                        faces_bg = last_faces.copy() if last_faces else []
                    if frame_bg is None or not faces_bg:
                        continue
                    largest_bg = max(faces_bg, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
                    ts = str(int(time.time()*1000))
                    fname = save_face(frame_bg, largest_bg, out_dir, person_name=person_name, collection_ts=ts)
                    collected_bg += 1
                    print(f"[{person_name}] Saved {fname} ({collected_bg}/{count})")
                print(f"Background collection done for '{person_name}'")
                rebuild_queue.put(True)
            threading.Thread(target=background_collect, daemon=True).start()
        elif key == ord('c'):
            if faces:
                # Capture only the largest face
                largest_face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
                # Recognize this face to get its label
                face_crop = extract_face(frame, largest_face)
                face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                face_pil = Image.fromarray(face_rgb)
                face_tensor = transform_embed(face_pil).unsqueeze(0).to(device)
                with torch.no_grad():
                    emb = model.embed(face_tensor)
                    emb = F.normalize(emb, p=2, dim=1)
                dists = torch.norm(embeddings_db.to(device) - emb, dim=1).cpu().numpy()
                idx = int(np.argmin(dists))
                name = class_names[idx] if dists[idx] < threshold else 'Unknown'
                if name != 'Unknown':
                    out_dir = os.path.join(base_targets_dir, name)
                    os.makedirs(out_dir, exist_ok=True)
                    ts = str(int(time.time()*1000))
                    filename = save_face(frame, largest_face, out_dir, person_name=name, collection_ts=ts)
                    print(f"Captured {filename} (label: {name})")
                    # Rebuild database with new sample
                    embeddings_db, class_names = build_targets_db(model, cfg, device)
                else:
                    print("Unknown face, skipping capture.")
            else:
                print("No face detected, cannot capture.")

        # If any background collection completed, rebuild embeddings DB
        try:
            rebuild_queue.get_nowait()
            embeddings_db, class_names = build_targets_db(model, cfg, device)
            print("Updated embeddings DB with new collected data.")
        except queue.Empty:
            pass

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 