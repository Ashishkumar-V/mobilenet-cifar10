# MobileNet-v2 on CIFAR-10 with Custom Quantization

CS6886W - Systems Engineering for Deep Learning, Assignment 2  
IIT Madras

## Repository Structure

```
├── model.py      — MobileNet-v2 adapted for CIFAR-10 (32×32 input)
├── dataset.py    — CIFAR-10 data loading with augmentation
├── engine.py     — Training and evaluation utilities
├── train.py      — Training script with W&B logging
├── quantize.py   — Custom PTQ: weight + activation quantization (no APIs)
├── test.py       — Evaluate with compression, log to W&B
├── sweep.py      — W&B sweep over bit-width configurations
└── requirements.txt
```

## Environment

```
Python:      3.10+
torch:       2.0+
torchvision: 0.15+
wandb:       0.15+
OS:          Linux
```

## Seed Configuration

```python
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False
```

## Install

```bash
pip install -r requirements.txt
```

## Train

```bash
python train.py --epochs 100 --lr 0.01 --optimizer sgd --batch_size 128
```

## Evaluate with Compression

```bash
# 8-bit weight + 8-bit activation quantization
python test.py --weight_quant_bits 8 --activation_quant_bits 8

# 4-bit weight + 4-bit activation quantization
python test.py --weight_quant_bits 4 --activation_quant_bits 4

# No compression (baseline)
python test.py --weight_quant_bits 32 --activation_quant_bits 32
```

## Run W&B Sweep (compression experiments)

```bash
python sweep.py --count 24
```

## GitHub

https://github.com/Ashishkumar-V/mobilenet-cifar10
