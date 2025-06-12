#!/usr/bin/env python3
"""
evaluate_model.py

Replica o split de treino e validação (80/20) usado em src/train_model.py e avalia a acurácia do modelo salvo.
"""
import os
import torch
from src.train_model import load_config, ClassifierNet, get_loaders
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # Carrega configurações e dispositivo
    cfg = load_config()
    device = torch.device(cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))

    # Replica split de treino/val usando get_loaders
    train_loader, val_loader, class_names = get_loaders(
        cfg['train_dir'], cfg['batch_size'], cfg['num_workers'], cfg['val_split'], cfg['seed']
    )
    num_classes = len(class_names)
    print(f'Número de classes: {num_classes}')
    print(f'Tamanho do conjunto de validação: {len(val_loader.dataset)} imagens')

    # Monta o modelo com mesma arquitetura e parâmetros de treinamento
    model = ClassifierNet(
        num_classes,
        cfg['embedding_dim'],
        pretrained=False,
        arcface=cfg.get('arcface', False),
        s=float(cfg.get('arc_s', 30.0)),
        m=float(cfg.get('arc_m', 0.50)),
        easy_margin=cfg.get('arc_easy_margin', False)
    ).to(device)

    # Carrega pesos do melhor checkpoint
    best_ckpt = os.path.join(cfg['checkpoint_dir'], 'best.pth')
    if not os.path.isfile(best_ckpt):
        print(f'Checkpoint não encontrado em {best_ckpt}')
        return
    state = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Coleta de verdadeiros e previstos para matriz de confusão
    y_true = []
    y_pred = []

    # Avalia acurácia no conjunto de validação
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            # Forward (arcface requer label no forward)
            if cfg.get('arcface', False):
                outputs = model(inputs, labels)
            else:
                outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            # Registra labels para matriz de confusão
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total if total > 0 else 0.0
    print(f'Acurácia na validação: {acc:.4f} ({correct}/{total})')

    # Gera matriz de confusão para as K classes mais frequentes
    cm = confusion_matrix(y_true, y_pred)
    class_counts = np.bincount(y_true)
    K = 20  # número de classes a plotar (ajuste conforme necessário)
    topk_idx = np.argsort(class_counts)[-K:]
    cm_sub = cm[np.ix_(topk_idx, topk_idx)]
    sub_labels = [class_names[i] for i in topk_idx]

    plt.figure(figsize=(12,10))
    sns.heatmap(cm_sub, annot=True, fmt='d', xticklabels=sub_labels, yticklabels=sub_labels, cmap='Blues')
    plt.title(f'Matriz de Confusão das {K} classes mais frequentes')
    plt.ylabel('Verdadeiro')
    plt.xlabel('Previsto')
    plt.tight_layout()
    plt.savefig('confusion_matrix_topk.png')
    print('Salvou matriz de confusão em confusion_matrix_topk.png')

if __name__ == '__main__':
    main() 