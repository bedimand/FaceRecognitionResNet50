#!/usr/bin/env python3
import cv2
import os
import time
import argparse
from src.face_detection import initialize_face_analyzer, detect_faces, save_face
from src.train_model import load_config

def main():
    parser = argparse.ArgumentParser(description="Interactive target collection: press 'c' to capture the largest face, 'q' to quit.")
    parser.add_argument("--name", "-n", required=True, help="Name of the person to collect images for")
    parser.add_argument("--output", "-o", default=None, help="Targets directory (default from config)")
    args = parser.parse_args()

    person_name = args.name
    # Determine targets directory from config if not provided
    cfg = load_config()
    base_output = args.output or cfg.get("targets_dir", "targets")

    # Prepare output directory
    out_dir = os.path.join(base_output, person_name)
    os.makedirs(out_dir, exist_ok=True)

    # Initialize face analyzer
    face_analyzer = initialize_face_analyzer()
    if face_analyzer is None:
        print("Failed to initialize face analyzer. Exiting.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Unable to open camera. Exiting.")
        return

    collected = 0
    print(f"Starting interactive collection for '{person_name}'. Press 'c' to capture, 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from camera.")
            break

        faces = detect_faces(frame, face_analyzer)
        # Draw rectangles around all detected faces
        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow("Collecting Faces", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Quitting interactive collection.")
            break
        elif key == ord('c'):
            if not faces:
                print("No faces detected, skipping capture.")
            else:
                # Capture only the largest face by area
                largest_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                ts = str(int(time.time() * 1000))
                filename = save_face(frame, largest_face, out_dir, person_name=person_name, collection_ts=ts)
                collected += 1
                print(f"Captured {filename} (#{collected})")

    cap.release()
    cv2.destroyAllWindows()
    print(f"Finished collection. Total images saved: {collected}")

if __name__ == "__main__":
    main() 