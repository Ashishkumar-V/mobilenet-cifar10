"""
train.py — Train MobileNet-v2 on CIFAR-10 with W&B logging.

Usage:
    python train.py --epochs 100 --lr 0.01 --optimizer sgd --batch_size 128
"""

import os
import random
import argparse
import numpy as np
import torch
import torch.optim.lr_scheduler as lr_sched
import wandb

from model   import mobilenet_v2
from dataset import get_loaders
from engine  import train_one_epoch, evaluate, get_optimizer


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs',      type=int,   default=100)
    p.add_argument('--batch_size',  type=int,   default=128)
    p.add_argument('--lr',          type=float, default=0.01)
    p.add_argument('--optimizer',   type=str,   default='sgd',
                   choices=['sgd', 'adam', 'adamw'])
    p.add_argument('--momentum',    type=float, default=0.9)
    p.add_argument('--weight_decay',type=float, default=1e-4)
    p.add_argument('--width_mult',  type=float, default=1.0)
    p.add_argument('--dropout',     type=float, default=0.2)
    p.add_argument('--seed',        type=int,   default=42)
    p.add_argument('--data_dir',    type=str,   default='./data')
    p.add_argument('--project',     type=str,   default='mobilenet-cifar10')
    p.add_argument('--num_workers', type=int,   default=2)
    return p.parse_args()


def train():
    args = parse_args()
    set_seed(args.seed)

    wandb.init(project=args.project, config=vars(args))
    cfg = wandb.config

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    train_loader, val_loader, _ = get_loaders(
        batch_size=cfg.batch_size, num_workers=args.num_workers, data_dir=args.data_dir
    )

    model = mobilenet_v2(width_mult=cfg.width_mult, dropout=cfg.dropout).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    model_size_mb = total_params * 4 / 1e6
    print(f"Params: {total_params:,}  |  Size: {model_size_mb:.2f} MB")

    optimizer = get_optimizer(model, cfg.optimizer, cfg.lr, cfg.momentum, cfg.weight_decay)
    scheduler = lr_sched.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    best_val_acc = 0.0
    run_name = f"mobilenet_bs{cfg.batch_size}_opt{cfg.optimizer}_lr{cfg.lr}"

    for epoch in range(cfg.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss,   val_acc   = evaluate(model, val_loader, device)

        wandb.log({
            'epoch':      epoch,
            'train_loss': train_loss,
            'train_acc':  train_acc,
            'val_loss':   val_loss,
            'val_acc':    val_acc,
            'lr':         optimizer.param_groups[0]['lr'],
        })
        print(f"Epoch {epoch:3d} | train_loss={train_loss:.4f} train_acc={train_acc:.2f}% "
              f"| val_loss={val_loss:.4f} val_acc={val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch':      epoch,
                'model_state_dict': model.state_dict(),
                'val_acc':    val_acc,
                'args':       vars(args),
            }, f"best_{run_name}.pth")

    wandb.log({'best_val_acc': best_val_acc, 'model_size_mb': model_size_mb})
    wandb.finish()
    print(f"Best val_acc: {best_val_acc:.2f}%")


if __name__ == '__main__':
    train()
