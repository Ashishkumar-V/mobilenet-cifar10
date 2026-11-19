"""
test.py — Evaluate MobileNet-v2 with custom quantization compression.

Usage:
    python test.py --weight_quant_bits 8 --activation_quant_bits 8
    python test.py --weight_quant_bits 4 --activation_quant_bits 4 --checkpoint best.pth
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--weight_quant_bits',     type=int, default=8,
                   help='Bits for weight quantization (32 = no quantization)')
    p.add_argument('--activation_quant_bits', type=int, default=8,
                   help='Bits for activation quantization (32 = no quantization)')
    p.add_argument('--checkpoint',   type=str, default=None,
                   help='Path to .pth checkpoint (auto-detects if not set)')
    p.add_argument('--data_dir',     type=str, default='./data')
    p.add_argument('--batch_size',   type=int, default=128)
    p.add_argument('--num_workers',  type=int, default=2)
    p.add_argument('--project',      type=str, default='mobilenet-cifar10-compression')
    p.add_argument('--width_mult',   type=float, default=1.0)
    p.add_argument('--no_wandb',     action='store_true')
    return p.parse_args()


def find_checkpoint():
    """Auto-detect best checkpoint in current directory."""
    for f in sorted(os.listdir('.')):
        if f.endswith('.pth') and 'best' in f:
            return f
    for f in sorted(os.listdir('.')):
        if f.endswith('.pth'):
            return f
    return None


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Load checkpoint ──────────────────────────────────────────────────────
    ckpt_path = args.checkpoint or find_checkpoint()
    if ckpt_path is None or not os.path.exists(ckpt_path):
        print("ERROR: No checkpoint found. Train first with train.py")
        sys.exit(1)

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    width_mult = ckpt.get('args', {}).get('width_mult', args.width_mult)

    model = mobilenet_v2(width_mult=width_mult).to(device)
    model.load_state_dict(ckpt['model_state_dict'])

    # ── Data ─────────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = get_loaders(
        batch_size=args.batch_size, num_workers=args.num_workers, data_dir=args.data_dir
    )

    # ── Baseline accuracy (no compression) ───────────────────────────────────
    _, baseline_acc = evaluate(model, test_loader, device)
    print(f"\nBaseline accuracy (no compression): {baseline_acc:.2f}%")

    # ── Apply compression ─────────────────────────────────────────────────────
    compressed_model = copy.deepcopy(model)

    # 1. Weight quantization
    quant_params = quantize_weights(compressed_model, args.weight_quant_bits)
    print(f"Weight quantization: {args.weight_quant_bits}-bit  "
          f"({len(quant_params)} layers quantized)")

    # 2. Activation quantization — calibrate then apply hooks
    act_quantizer = ActivationQuantizer()
    act_stats = act_quantizer.calibrate(compressed_model, val_loader, device, num_batches=20)
    act_quantizer.apply(compressed_model, args.activation_quant_bits)
    print(f"Activation quantization: {args.activation_quant_bits}-bit  "
          f"({len(act_stats)} layers calibrated)")

    # 3. Evaluate compressed model
    _, quant_acc = evaluate(compressed_model, test_loader, device)
    act_quantizer.remove()
    print(f"Quantized accuracy: {quant_acc:.2f}%  "
          f"(drop: {baseline_acc - quant_acc:.2f}%)")

    # ── Size & compression ratios ─────────────────────────────────────────────
    sizes = compute_model_sizes(
        compressed_model, args.weight_quant_bits,
        args.activation_quant_bits, act_stats
    )

    print(f"\n{'='*55}")
    print(f"  Weight quantization: {args.weight_quant_bits}-bit")
    print(f"  Activation quantization: {args.activation_quant_bits}-bit")
    print(f"{'='*55}")
    print(f"  Original  model size : {sizes['orig_model_mb']:.2f} MB")
    print(f"  Quantized model size : {sizes['quant_model_mb']:.2f} MB")
    print(f"  Model compression    : {sizes['model_ratio']:.2f}x")
    print(f"  Weight compression   : {sizes['weight_ratio']:.2f}x")
    print(f"  Activation compression: {sizes['act_ratio']:.2f}x")
    print(f"  Storage overhead     : {sizes['weight_overhead_b'] + sizes['act_overhead_b']} bytes")
    print(f"  Accuracy (original)  : {baseline_acc:.2f}%")
    print(f"  Accuracy (quantized) : {quant_acc:.2f}%")
    print(f"{'='*55}\n")

    # ── W&B logging ──────────────────────────────────────────────────────────
    if not args.no_wandb:
        run_name = f"w{args.weight_quant_bits}b_a{args.activation_quant_bits}b"
        wandb.init(project=args.project, name=run_name, config={
            'weight_quant_bits':     args.weight_quant_bits,
            'activation_quant_bits': args.activation_quant_bits,
            'checkpoint':            ckpt_path,
        })
        wandb.log({
            'weight_quant_bits':      args.weight_quant_bits,
            'activation_quant_bits':  args.activation_quant_bits,
            'baseline_acc':           baseline_acc,
            'quantized_acc':          quant_acc,
            'accuracy_drop':          baseline_acc - quant_acc,
            'compression_ratio':      sizes['model_ratio'],
            'weight_compression':     sizes['weight_ratio'],
            'act_compression':        sizes['act_ratio'],
            'orig_model_mb':          sizes['orig_model_mb'],
            'quant_model_mb':         sizes['quant_model_mb'],
            'model_size_mb':          sizes['quant_model_mb'],
            'storage_overhead_bytes': sizes['weight_overhead_b'] + sizes['act_overhead_b'],
        })
        wandb.finish()

    return quant_acc, sizes


if __name__ == '__main__':
    main()
