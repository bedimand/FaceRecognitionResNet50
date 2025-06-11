import os
import torch
import yaml
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset, random_split
from tqdm import tqdm
import random
import numpy as np
import math
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

class ArcMarginProduct(nn.Module):
    """
    Implement of large margin arc distance: classic ArcFace
    """
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s  # norm of input feature
        self.m = m  # margin
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # input is normalized in forward; weight is normalized
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))  # [N, out_features]
        sine = torch.sqrt(1.0 - cosine.pow(2).clamp(0,1))
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(theta + m)
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # one-hot encode
        one_hot = torch.zeros_like(cosine, device=input.device)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s
        return output

class ClassifierNet(nn.Module):
    def __init__(self, num_classes, embedding_dim, pretrained=True, arcface=False, s=30.0, m=0.50, easy_margin=False):
        super().__init__()
        base = models.resnet50(pretrained=pretrained)
        in_features = base.fc.in_features
        base.fc = nn.Identity()
        self.backbone = base
        self.embedding = nn.Linear(in_features, embedding_dim)
        self.arcface = arcface
        if arcface:
            self.margin = ArcMarginProduct(embedding_dim, num_classes, s=s, m=m, easy_margin=easy_margin)
        else:
            self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x, label=None):
        x = self.backbone(x)
        x = self.embedding(x)
        x = torch.relu(x)
        if self.arcface:
            assert label is not None, "ArcFace forward requires labels"
            return self.margin(x, label)
        return self.classifier(x)

    def embed(self, x):
        x = self.backbone(x)
        x = self.embedding(x)
        return x


def get_loaders(train_dir, batch_size, num_workers, val_split, seed):
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        transforms.RandomErasing(p=0.1)
    ])
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    # Load full dataset and split
    ds_train_full = ImageFolder(train_dir, transform_train)
    ds_val_full = ImageFolder(train_dir, transform_val)
    total = len(ds_train_full)
    val_len = int(total * val_split)
    train_len = total - val_len
    # Random split reproducibly
    g = torch.Generator()
    g.manual_seed(seed)
    train_subset, val_subset = random_split(ds_train_full, [train_len, val_len], generator=g)
    # Apply val transform to val subset
    val_subset = Subset(ds_val_full, val_subset.indices)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, ds_train_full.classes


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total = 0
    for inputs, labels in tqdm(loader, desc='Train', leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs, labels)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)
        total += inputs.size(0)
    return running_loss/total, running_corrects.double()/total


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc='Val', leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs, labels)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            total += inputs.size(0)
    return running_loss/total, running_corrects.double()/total


def main():
    cfg = load_config()
    device = torch.device(cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    # Set seeds for reproducibility
    seed = cfg.get('seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Build data loaders
    train_loader, val_loader, class_names = get_loaders(
        cfg['train_dir'], cfg['batch_size'], cfg['num_workers'], cfg['val_split'], cfg['seed']
    )
    num_classes = len(class_names)
    print(f"Training on {num_classes} classes: {class_names}")
    # Build model with ArcFace if configured
    model = ClassifierNet(
        num_classes,
        cfg['embedding_dim'],
        pretrained=cfg.get('pretrained', True),
        arcface=cfg.get('arcface', False),
        s=float(cfg.get('arc_s', 30.0)),
        m=float(cfg.get('arc_m', 0.50)),
        easy_margin=cfg.get('arc_easy_margin', False)
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    lr = float(cfg['learning_rate'])
    wd = float(cfg.get('weight_decay', 0))
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    # LR scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'], eta_min=1e-6)
    # Early stopping
    patience = cfg.get('early_stopping', {}).get('patience', 3)
    no_improve = 0
    best_acc = 0.0
    ckpt_dir = cfg.get('checkpoint_dir', 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    for epoch in range(1, cfg['epochs'] + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f'Epoch {epoch}/{cfg["epochs"]} - '
              f'Train loss: {train_loss:.4f}, acc: {train_acc:.4f} | '
              f'Val loss: {val_loss:.4f}, acc: {val_acc:.4f}')
        # Checkpoint epoch
        torch.save(model.state_dict(), os.path.join(ckpt_dir, f'epoch_{epoch}.pth'))
        # Early stopping and best model
        if val_acc > best_acc:
            best_acc = val_acc
            no_improve = 0
            torch.save(model.state_dict(), os.path.join(ckpt_dir, 'best.pth'))
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"No improvement for {patience} epochs, stopping training.")
                break
        # Step LR scheduler
        scheduler.step()
    # Save embedding-only model
    embed_path = os.path.join(ckpt_dir, 'embed_model.pth')
    torch.save({
        'backbone': model.backbone.state_dict(),
        'embedding': model.embedding.state_dict()
    }, embed_path)
    print(f'Training complete. Best Val Acc: {best_acc:.4f}')
    print(f'Embedding model saved to {embed_path}')

if __name__ == '__main__':
    main()
