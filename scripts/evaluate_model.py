import os
import torch
import yaml
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve

class PairDataset(Dataset):
    def __init__(self, root_dir, same_list, diff_list, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.pairs = []
        with open(same_list, 'r') as f:
            for line in f:
                a, b = line.strip().split()
                self.pairs.append((a, b, 1))
        with open(diff_list, 'r') as f:
            for line in f:
                a, b = line.strip().split()
                self.pairs.append((a, b, 0))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        a, b, label = self.pairs[idx]
        path_a = os.path.join(self.root_dir, a + '.ppm')
        path_b = os.path.join(self.root_dir, b + '.ppm')
        img_a = Image.open(path_a).convert('RGB')
        img_b = Image.open(path_b).convert('RGB')
        if self.transform:
            img_a = self.transform(img_a)
            img_b = self.transform(img_b)
        return img_a, img_b, label

class EmbeddingNet(nn.Module):
    def __init__(self, embedding_dim, pretrained=True):
        super().__init__()
        base = models.resnet50(pretrained=pretrained)
        in_features = base.fc.in_features
        base.fc = nn.Identity()
        self.backbone = base
        self.fc = nn.Linear(in_features, embedding_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1)


def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)


def compute_distances(model, loader, device):
    distances = []
    labels = []
    model.eval()
    with torch.no_grad():
        for img_a, img_b, label in loader:
            img_a = img_a.to(device)
            img_b = img_b.to(device)
            emb_a = model(img_a)
            emb_b = model(img_b)
            dist = F.pairwise_distance(emb_a, emb_b)
            distances.append(dist.cpu().numpy())
            labels.append(np.array(label))
    return np.concatenate(distances), np.concatenate(labels)


def main():
    cfg = load_config()
    device = torch.device(cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    root_dir = os.path.join(cfg['data_dir'], 'faces')
    test_ds = PairDataset(root_dir, cfg['val_same'], cfg['val_diff'], transform)
    test_loader = DataLoader(test_ds, batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'])

    model = EmbeddingNet(cfg['embedding_dim'], pretrained=False).to(device)
    best_path = os.path.join(cfg['checkpoint_dir'], 'best.pth')
    model.load_state_dict(torch.load(best_path, map_location=device))

    distances, labels = compute_distances(model, test_loader, device)
    thresh = float(cfg.get('threshold', 1.0))
    preds = (distances < thresh).astype(int)
    labels = labels.astype(int)

    acc = accuracy_score(labels, preds)
    cm = confusion_matrix(labels, preds)
    auc = roc_auc_score(labels, -distances)
    fpr, tpr, thr = roc_curve(labels, -distances)
    eer_idx = np.nanargmin(np.abs(fpr + tpr - 1))
    eer = (fpr[eer_idx] + (1 - tpr[eer_idx])) / 2

    print(f'Accuracy: {acc:.4f}')
    print('Confusion Matrix:')
    print(cm)
    print(f'ROC AUC: {auc:.4f}')
    print(f'EER: {eer:.4f} at threshold {thr[eer_idx]:.4f}')
    print(f'Configured threshold: {thresh:.4f}')

if __name__ == '__main__':
    main() 