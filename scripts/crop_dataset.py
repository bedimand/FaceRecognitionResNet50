import os
import cv2
from train_model import load_config
from face_detection import initialize_face_analyzer, detect_faces, extract_face

def main():
    cfg = load_config()
    data_dir = cfg['data_dir']  # e.g. 'dataset'
    analyzer = initialize_face_analyzer()
    out_root = 'dataset_crops'
    # Process each subset (train, val, etc.)
    for subset in os.listdir(data_dir):
        in_subset = os.path.join(data_dir, subset)
        if not os.path.isdir(in_subset):
            continue
        out_subset = os.path.join(out_root, subset)
        for class_name in os.listdir(in_subset):
            in_class_dir = os.path.join(in_subset, class_name)
            if not os.path.isdir(in_class_dir):
                continue
            out_class_dir = os.path.join(out_subset, class_name)
            os.makedirs(out_class_dir, exist_ok=True)
            for fname in os.listdir(in_class_dir):
                if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.ppm')):
                    continue
                in_path = os.path.join(in_class_dir, fname)
                frame = cv2.imread(in_path)
                if frame is None:
                    continue
                faces = detect_faces(frame, analyzer)
                if not faces:
                    continue
                # pick the largest detected face
                face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
                crop = extract_face(frame, face)
                out_path = os.path.join(out_class_dir, fname)
                cv2.imwrite(out_path, crop)
    print('Face cropping complete.')

if __name__ == '__main__':
    main() 