#!/usr/bin/env python3
import os
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import time

from src.face_detection import initialize_face_analyzer, detect_faces, extract_face, save_face
from src.train_model import ClassifierNet, load_config

# Build transform for embeddings
transform_embed = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

def build_embedding_model(cfg, device):
    # Instantiate classifier net to access backbone and embedding layers
    train_dir = cfg['train_dir']
    num_classes = len([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir,d))])
    model = ClassifierNet(num_classes, cfg['embedding_dim'], pretrained=False)
    # Load trained weights for the full classification model
    state = torch.load(os.path.join(cfg['checkpoint_dir'], 'embed_model.pth'), map_location=device)
    # state contains backbone and embedding state_dicts
    model.backbone.load_state_dict(state['backbone'])
    model.embedding.load_state_dict(state['embedding'])
    model.to(device).eval()
    return model


def build_targets_db(model, cfg, device):
    targets_dir = cfg.get('targets_dir', 'targets')
    ds = ImageFolder(targets_dir, transform_embed)
    # Use single-process loading to avoid worker crashes on Windows paging file errors
    loader = DataLoader(ds, batch_size=cfg['batch_size'], shuffle=False, num_workers=0)
    embeddings = []
    labels = []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            emb = model.embed(imgs)
            emb = F.normalize(emb, p=2, dim=1)
            embeddings.append(emb.cpu())
            labels.extend(lbls.tolist())
    embeddings = torch.cat(embeddings, dim=0)  # shape (N, D)
    # Convert labels list to tensor and compute per-class centroids
    labels_tensor = torch.tensor(labels)
    class_names = ds.classes
    centroids = []
    for idx in range(len(class_names)):
        mask = labels_tensor == idx
        class_embs = embeddings[mask]
        centroids.append(class_embs.mean(dim=0))
    embeddings_db = torch.stack(centroids, dim=0)
    return embeddings_db, class_names


def interactive_add_target(cap, face_analyzer, base_targets_dir, model, cfg, device):
    """Interactive addition of a new target without restarting detection."""
    new_name = input("Enter new target name: ").strip()
    if not new_name:
        print("Empty name, aborting.")
        return None, None
    out_dir = os.path.join(base_targets_dir, new_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Adding targets for '{new_name}'. Press 'c' to capture largest face, 'q' to finish.")
    collected = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame.")
            break
        faces = detect_faces(frame, face_analyzer)
        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.imshow("Add Target", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            if faces:
                largest_face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
                ts = str(int(time.time()*1000))
                filename = save_face(frame, largest_face, out_dir, person_name=new_name, collection_ts=ts)
                collected += 1
                print(f"Saved {filename} (#{collected})")
            else:
                print("No face detected.")
    cv2.destroyWindow("Add Target")
    embeddings_db, class_names = build_targets_db(model, cfg, device)
    print(f"Updated targets: {class_names}")
    return embeddings_db, class_names

