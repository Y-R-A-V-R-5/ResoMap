1. configs\config.yaml

a)
# ResoMap Global Configuration
project_name: "ResoMap"
description: "Resolution Sensitivity Under CPU Constraints"

# The Experimental Axis (The Independent Variable)
resolutions: [64, 128, 224, 256, 320]

# Hardware & Profiling Settings
system:
  device: "cpu"
  num_threads: 4              # Fixed to ensure reproducible latency variance
  warmup_runs: 5              # To stabilize CPU frequency/cache
  num_profiling_runs: 30      # For meaningful mean/variance calculation
  track_activation_memory: true

data:
  raw_path: "data/"
  num_workers: 2              # CPU-bound; don't over-subscribe
  pin_memory: false           # Not needed for CPU-only
  val_split: 0.2

training:
  batch_size: 16              # Deployment-realistic for CPU inference
  epochs: 20                  # Sufficient to detect representation saturation
  learning_rate: 0.001
  optimizer: "adam"
  criterion: "cross_entropy"

mlflow:
  tracking_uri: "https://dagshub.com/Y-R-A-V-R-5/ResoMap.mlflow"
  experiment_name: "ResoMap"

# Minimal augmentation to prevent "hallucinated" detail
augmentation:
  horizontal_flip: true
  rotation: 10

b)
project_name: "ResoMap"
system:
  device: "cpu"
  num_threads: 4  # Locked for latency consistency
  pin_memory: false

sweep:
  resolutions: [64, 128, 224, 256, 320]
  models: ["LN5", "LN10", "VGG11", "VGG13"]

data:
  raw_path: "data/"
  dataset_name: "unified-skin-cancer"
  num_workers: 2
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1

hyperparameters:
  batch_size: 16 # Realistic for edge-CPU inference
  learning_rate: 0.001
  epochs: 15
  optimizer: "adam"

mlflow:
  tracking_uri: "https://dagshub.com/Y-R-A-V-R-5/ResoMap.mlflow"
  experiment_name: "ResoMap"

2. configs/LN5.yaml

model_name: "LN5"
type: "classic_cnn"
stages:
  - stage: "stem"
    layers: [[3, 6, 5, 1, 0]] # [in, out, kernel, stride, padding]
    pool: [2, 2] # [kernel, stride]
  - stage: "middle"
    layers: [[6, 16, 5, 1, 0]]
    pool: [2, 2]
  - stage: "bottleneck"
    layers: [[16, 120, 5, 1, 0]]
classifier:
  fc_layers: [84]
  dropout: 0.0
  pooling: "adaptive_avg" # Keeps parameter count constant across resolutions

3. configs/LN10.yaml
model_name: "LN10"
type: "classic_cnn"
stages:
  - stage: "entry"
    layers: [[3, 16, 3, 1, 1], [16, 16, 3, 1, 1]]
    pool: [2, 2]
  - stage: "mid_transition"
    layers: [[16, 32, 3, 1, 1], [32, 32, 3, 1, 1]]
    pool: [2, 2]
  - stage: "deep_feature"
    layers: [[32, 64, 3, 1, 1], [64, 64, 3, 1, 1], [64, 64, 3, 1, 1]]
    pool: [2, 2]
classifier:
  fc_layers: [128, 64]
  dropout: 0.1
  pooling: "adaptive_avg"

4. configs/VGG11.yaml

model_name: "VGG11"
type: "vgg"
stages:
  - stage: "stage1" # High resolution, low channels
    layers: [[3, 64, 3, 1, 1]]
    pool: [2, 2]
  - stage: "stage2"
    layers: [[64, 128, 3, 1, 1]]
    pool: [2, 2]
  - stage: "stage3"
    layers: [[128, 256, 3, 1, 1], [256, 256, 3, 1, 1]]
    pool: [2, 2]
  - stage: "stage4"
    layers: [[256, 512, 3, 1, 1], [512, 512, 3, 1, 1]]
    pool: [2, 2]
  - stage: "stage5" # Low resolution, high channels
    layers: [[512, 512, 3, 1, 1], [512, 512, 3, 1, 1]]
    pool: [2, 2]
classifier:
  fc_layers: [4096, 4096]
  dropout: 0.5
  pooling: "adaptive_avg"

5. configs/VGG13.yaml

model_name: "VGG13"
type: "vgg"
stages:
  - stage: "stage1"
    layers: [[3, 64, 3, 1, 1], [64, 64, 3, 1, 1]]
    pool: [2, 2]
  - stage: "stage2"
    layers: [[64, 128, 3, 1, 1], [128, 128, 3, 1, 1]]
    pool: [2, 2]
  - stage: "stage3"
    layers: [[128, 256, 3, 1, 1], [256, 256, 3, 1, 1]]
    pool: [2, 2]
  - stage: "stage4"
    layers: [[256, 512, 3, 1, 1], [512, 512, 3, 1, 1]]
    pool: [2, 2]
  - stage: "stage5"
    layers: [[512, 512, 3, 1, 1], [512, 512, 3, 1, 1]]
    pool: [2, 2]
classifier:
  fc_layers: [4096, 4096]
  dropout: 0.5
  pooling: "adaptive_avg"

 requirements.txt
-------------------------
Core Numerical Stack
-------------------------
numpy==1.26.4
scipy==1.11.4
-------------------------
PyTorch (CPU-only)
-------------------------
torch==2.2.2
torchvision==0.17.2
-------------------------
Experiment Tracking
-------------------------
mlflow==2.11.3
dagshub==0.3.28
-------------------------
Data Handling & Utilities
-------------------------
pandas==2.2.1
pyyaml==6.0.1
tqdm==4.66.2
-------------------------
Visualization
-------------------------
matplotlib==3.8.3
seaborn==0.13.2
-------------------------
Profiling & System Metrics
-------------------------
psutil==5.9.8
memory-profiler==0.61.0
-------------------------
Image I/O
-------------------------
Pillow==10.2.0
opencv-python==4.9.0.80
-------------------------
Dataset Download (KaggleHub)
-------------------------
kagglehub==0.2.7
-------------------------
Notebooks (optional)
-------------------------
jupyter==1.0.0
ipykernel==6.29.3
Dataset is from Kaggle.
import kagglehub
Download latest version
path = kagglehub.dataset_download("vinline/unified-dataset-for-skin-cancer-classification")
print("Path to dataset files:", path)
import mlflow
mlflow.set_tracking_uri("https://dagshub.com/Y-R-A-V-R-5/ResoMap.mlflow")
mlflow.set_experiment("ResoMap")
I need only 5 files of configs*.yaml. Not anything else.