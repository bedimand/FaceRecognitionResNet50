# Recognition-CNN
**Acurácia de Validação: 91,43% (32256/35279)**

Um sistema de reconhecimento facial em tempo real em Python, construído com PyTorch. Este repositório oferece duas funcionalidades principais:

1. **Treinamento do Modelo** via `src/train_model.py`
2. **Reconhecimento em Tempo Real** via `main.py`

---

## Sumário

- [Visão Geral](#visão-geral)
- [Pré-requisitos](#pré-requisitos)
- [Instalação](#instalação)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Configuração](#configuração)
- [Preparação do Dataset](#preparação-do-dataset)
- [Processo de Treinamento](#processo-de-treinamento)
  - [Arquitetura do Modelo](#arquitetura-do-modelo)
  - [Carregamento e Aumento de Dados](#carregamento-e-aumento-de-dados)
  - [Loop de Treinamento & Avaliação](#loop-de-treinamento--avaliação)
  - [Checkpoint e Exportação de Embeddings](#checkpoint-e-exportação-de-embeddings)
- [Aplicação de Reconhecimento em Tempo Real](#aplicação-de-reconhecimento-em-tempo-real)
  - [Executando o App](#executando-o-app)
  - [Controles e Uso](#controles-e-uso)
  - [Adicionando Novos Alvos](#adicionando-novos-alvos)
- [Avaliação do Modelo (`evaluate_model.py`)](#avaliação-do-modelo-evaluate_modelpy)
- [Resolução de Problemas](#resolução-de-problemas)
- [Licença](#licença)

---

## Visão Geral

Este projeto implementa um pipeline completo de reconhecimento facial:

- **Treinamento**: Treina um modelo de embedding baseado em ResNet-50 com opção de margem ArcFace em um conjunto de imagens rotulado.
- **Inferência**: Utiliza o modelo treinado para detecção e identificação de faces em tempo real a partir de webcam.

---

## Pré-requisitos

- Python 3.8 ou superior
- PyTorch 1.10+ com suporte a CUDA (opcional, mas recomendado)
- OpenCV
- Webcam para reconhecimento ao vivo

---

## Instalação

1. **Clonar o repositório**
   ```bash
   git clone https://github.com/yourusername/Recognition-CNN.git
   cd Recognition-CNN
   ```

2. **Instalar dependências**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verificar o ambiente de GPU** (se estiver usando CUDA)
   ```python
   import torch
   print(torch.cuda.is_available())  # deve retornar True se GPU for detectada
   ```

---

## Estrutura do Projeto

```text
Recognition-CNN/
├── config.yaml            # Hiperparâmetros e caminhos
├── dataset/               # Imagens brutas organizadas por classe
│   └── train/
│       ├── Alice/
│       │   ├── alice1.jpg
│       │   └── ...
│       └── Bob/
├── src/
│   ├── train_model.py     # Script de treinamento do modelo de embedding
│   ├── face_detection.py  # Utilitários de detecção de faces
│   └── face_recognition.py# Rotinas de embedding e banco de dados
├── targets/               # Faces capturadas em tempo real
├── checkpoints/           # Checkpoints de treinamento e modelo de embedding
├── main.py                # Demonstração em tempo real
├── README.md              # Este arquivo
└── requirements.txt       # Dependências Python
``` 

---

## Configuração (`config.yaml`)

Parâmetros principais:

```yaml
train_dir: 'dataset/train'    # Pasta raiz com imagens rotuladas
val_split: 0.2                # Fração dos dados de treino para validação
batch_size: 64                # Tamanho do batch de treinamento
num_workers: 8                # Número de processos para DataLoader
epochs: 20                    # Número total de épocas
learning_rate: 1e-4           # Taxa de aprendizado do Adam
weight_decay: 1e-5            # Decaimento de peso do Adam
device: 'cuda'                # 'cuda' ou 'cpu'
pretrained: true              # Usar backbone ResNet-50 pré-treinado
checkpoint_dir: 'checkpoints'
embedding_dim: 512            # Dimensão do vetor de embedding
arcface: true                 # Ativar margem ArcFace
arc_s: 30.0                   # Escala do ArcFace
arc_m: 0.50                   # Margem do ArcFace
arc_easy_margin: false        # Flag de margem fácil no ArcFace
threshold: 0.55               # Limiar de distância para reconhecimento
targets_dir: 'targets'        # Onde salvar faces capturadas
seed: 42                      # Semente aleatória
use_amp: true                 # Precisão mista (não utilizada atualmente)
``` 

Ajuste conforme necessário para melhorar performance, trocar dispositivo ou alterar caminhos.

---

## Preparação do Dataset

Neste projeto, utilizamos uma versão filtrada do dataset VGGFace2 (disponível em https://www.kaggle.com/datasets/hearfool/vggface2). Removemos a pasta de validação original e reorganizamos o diretório de treinamento para conter 500 identidades, cada uma com pelo menos 250 imagens, totalizando mais de 170.000 imagens.

Estrutura de pastas após filtragem:
```text
dataset/train/
├── Pessoa1/
│   ├── img001.jpg
│   └── imgXXX.jpg
├── Pessoa2/
└── ...
```

Para garantir consistência, dividimos manualmente esse conjunto em:
- **Treino**: 80% das imagens
- **Validação**: 20% das imagens

O script `src/train_model.py` também oferece uma divisão programática usando `val_split: 0.2` em `config.yaml` via `torch.utils.data.random_split`, reproduzindo a mesma proporção em tempo de execução.

O processo de treinamento com esse conjunto levou aproximadamente **7 horas** em GPU para completar todas as 20 épocas.

---

## Processo de Treinamento (`src/train_model.py`)

O script executa as seguintes etapas:

### 1. Configuração e Dispositivo

- Carrega hiperparâmetros de `config.yaml`.
- Define dispositivo como CUDA (se disponível) ou CPU.
- Seta sementes para Python, NumPy e PyTorch para reprodutibilidade.

### 2. Carregamento e Aumento de Dados

- **Transformações (Treino)**:
  - RandomResizedCrop(224)
  - RandomHorizontalFlip
  - ColorJitter
  - RandomGrayscale
  - ToTensor & Normalize
  - RandomErasing

- **Transformações (Validação)**:
  - Resize(256)
  - CenterCrop(224)
  - ToTensor & Normalize

- Utiliza `ImageFolder` para carregar imagens e `random_split` para criar subconjuntos de treino/validação.
- `DataLoader` gerencia batching e shuffle.

### 3. Arquitetura do Modelo (`ClassifierNet`)

- **Backbone**: ResNet-50 (pré-treinado no ImageNet se configurado).
- **Camada de Embedding**: Linear(2048 → `embedding_dim`).
- **Cabeça de Classificação**:
  - **ArcFace**: se `arcface: true`, utiliza `ArcMarginProduct` para margem.
  - **Padrão**: caso contrário, uma camada linear simples.

### 4. Loop de Treinamento & Avaliação

- **Loss**: `CrossEntropyLoss` (com margem via ArcFace se ativado).
- **Otimiza**dor: Adam com weight decay.
- **Scheduler**: CosineAnnealingLR ao longo das épocas.
- **Early Stopping**: interrompe se a acurácia de validação não melhorar em `patience` épocas.

Em cada época:
1. **Treino**: forward → calcula loss → backward → step do otimizador.
2. **Validação**: avaliação em `torch.no_grad()`.
3. **Log**: exibe loss e acurácia de treino/validação.
4. **Checkpoint**:
   - Salva `epoch_{n}.pth` a cada época.
   - Atualiza `best.pth` se a acurácia de validação melhorar.
5. **Step do Scheduler**.

### 5. Checkpoint e Exportação de Embeddings

Ao fim do treinamento, salva o modelo de embedding em:

```bash
checkpoints/embed_model.pth
```

Contendo:
- `backbone`: pesos da ResNet-50 até antes do fc.
- `embedding`: pesos da camada de embedding.

Esse modelo é carregado por `main.py` para inferência rápida.

---

## Aplicação de Reconhecimento em Tempo Real (`main.py`)

Fluxo principal:

1. **Inicialização**:
   - Carrega `config.yaml`, define `device`.
   - Inicia detector de faces (`initialize_face_analyzer`).
   - Carrega modelo de embedding de `checkpoints/embed_model.pth`.
   - Constrói banco de embeddings a partir de pastas em `targets/`.

2. **Loop da Webcam**:
   - Captura quadros da câmera (1920×1080).
   - Detecta faces e desenha retângulos.
   - Pré-processa cada face e calcula embedding normalizado.
   - Compara contra centroides e rotula se distância < `threshold`.
   - Exibe janela com resultados ao vivo.

3. **Controles Interativos**:
   - **`a`**: adiciona novo alvo em segundo plano (coleta N amostras em `targets/<Nome>/` e atualiza o banco).
   - **`c`**: captura uma amostra do rosto reconhecido e reconstrói o banco.
   - **`q`**: encerra a aplicação.

---

## Avaliação do Modelo (`evaluate_model.py`)

Para avaliar a acurácia do modelo treinado, execute o script de avaliação:
```bash
python evaluate_model.py
```

O modelo alcançou uma acurácia de validação de **91,43%** (32256/35279).

---

## Resolução de Problemas

- **Câmera não detectada**: experimente usar outro backend do OpenCV:
  ```python
  cv2.VideoCapture(0, cv2.CAP_DSHOW)
  ```

- **Baixa acurácia**:
  - Colete imagens mais variadas para cada pessoa.
  - Reduza `threshold` em `config.yaml`.
  - Treine novamente com mais dados ou épocas.

- **CUDA OOM**: diminua `batch_size` ou use `device: 'cpu'`.

- **Problemas de instalação**: verifique `requirements.txt` e a compatibilidade de versão.

---

## Licença

Este projeto está licenciado sob a Licença MIT. Veja [LICENSE](LICENSE) para mais detalhes. 