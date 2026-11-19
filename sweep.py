"""
sweep.py — W&B sweep over weight and activation bit-widths (≥8 runs).

Usage:
    python sweep.py --count 16
"""

import os
import sys
import copy
import argparse
import torch
import wandb

from model    import mobilenet_v2
from dataset  import get_loaders
from engine   import evaluate
from quantize import quantize_weights, ActivationQuantizer, compute_model_sizes


def find_checkpoint():
    for f in sorted(os.listdir('.')):
        if f.endswith('.pth') and 'best' in f:
            return f
    for f in sorted(os.listdir('.')):
        if f.endswith('.pth'):
            return f
    return None


SWEEP_CONFIG = {
    'method': 'grid',
    'metric': {'name': 'quantized_acc', 'goal': 'maximize'},
    'parameters': {
        'weight_quant_bits':     {'values': [2, 4, 6, 8, 16, 32]},
        'activation_quant_bits': {'values': [2, 4, 8, 32]},
    },
}


def run_one(data_dir, checkpoint_path, project):
    wandb.init(project=project)
    cfg = wandb.config

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt = torch.load(checkpoint_path, map_location=device)
    width_mult = ckpt.get('args', {}).get('width_mult', 1.0)

    base_model = mobilenet_v2(width_mult=width_mult).to(device)
    base_model.load_state_dict(ckpt['model_state_dict'])

    _, val_loader, test_loader = get_loaders(
        batch_size=128, num_workers=2, data_dir=data_dir
    )

    _, baseline_acc = evaluate(base_model, test_loader, device)

    model = copy.deepcopy(base_model)
    quantize_weights(model, cfg.weight_quant_bits)

    aq = ActivationQuantizer()
    act_stats = aq.calibrate(model, val_loader, device, num_batches=20)
    aq.apply(model, cfg.activation_quant_bits)

    _, quant_acc = evaluate(model, test_loader, device)
    aq.remove()

    sizes = compute_model_sizes(model, cfg.weight_quant_bits,
                                cfg.activation_quant_bits, act_stats)

    wandb.log({
        'weight_quant_bits':      cfg.weight_quant_bits,
        'activation_quant_bits':  cfg.activation_quant_bits,
        'baseline_acc':           baseline_acc,
        'quantized_acc':          quant_acc,
        'accuracy_drop':          baseline_acc - quant_acc,
        'compression_ratio':      sizes['model_ratio'],
        'weight_compression':     sizes['weight_ratio'],
        'act_compression':        sizes['act_ratio'],
        'model_size_mb':          sizes['quant_model_mb'],
        'orig_model_mb':          sizes['orig_model_mb'],
        'storage_overhead_bytes': sizes['weight_overhead_b'] + sizes['act_overhead_b'],
    })
    wandb.finish()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--count',     type=int, default=24)
    p.add_argument('--data_dir',  type=str, default='./data')
    p.add_argument('--project',   type=str, default='mobilenet-cifar10-compression')
    p.add_argument('--checkpoint',type=str, default=None)
    args = p.parse_args()

    ckpt = args.checkpoint or find_checkpoint()
    if ckpt is None:
        print("No checkpoint found. Train first.")
        sys.exit(1)

    sys.argv = [sys.argv[0]]   # prevent sweep args leaking into sub-parsers

    sweep_id = wandb.sweep(SWEEP_CONFIG, project=args.project)
    print(f"Sweep ID: {sweep_id}")
    print(f"View at: https://wandb.ai/cs25m505-indian-institute-of-technology/{args.project}/sweeps/{sweep_id}")

    wandb.agent(sweep_id, function=lambda: run_one(args.data_dir, ckpt, args.project),
                count=args.count, project=args.project)


if __name__ == '__main__':
    main()
