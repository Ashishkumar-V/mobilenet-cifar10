"""
quantize.py — Custom Post-Training Quantization (PTQ) without any compression APIs.

Implements uniform asymmetric quantization for both weights and activations.

Quantization formula:
    scale     = (x_max - x_min) / (2^bits - 1)
    zero_point = clamp(round(-x_min / scale), 0, 2^bits-1)
    q          = clamp(round(x / scale) + zero_point, 0, 2^bits-1)   [stored as int]
    x_approx   = (q - zero_point) * scale                             [dequantized]

Storage overhead per quantized tensor:
    scale      : 4 bytes  (float32)
    zero_point : 4 bytes  (int32)
    Total overhead per layer: 8 bytes
"""

import torch
import torch.nn as nn
import copy


# ---------------------------------------------------------------------------
# Core quantization math
# ---------------------------------------------------------------------------

def quantize_tensor(x: torch.Tensor, bits: int):
    """
    Uniform asymmetric quantization.

    Returns:
        x_dequant : dequantized approximation (same shape as x)
        scale     : float
        zero_point: int
    """
    if bits >= 32:
        return x.clone(), 1.0, 0

    qmin, qmax = 0, (1 << bits) - 1
    x_min = x.min().item()
    x_max = x.max().item()

    if x_min == x_max:          # degenerate (all-zero layer, etc.)
        return torch.zeros_like(x), 1.0, 0

    scale      = (x_max - x_min) / (qmax - qmin)
    zero_point = int(round(-x_min / scale))
    zero_point = max(qmin, min(qmax, zero_point))

    q         = torch.clamp(torch.round(x / scale) + zero_point, qmin, qmax)
    x_dequant = (q - zero_point) * scale
    return x_dequant, scale, zero_point


# ---------------------------------------------------------------------------
# Weight quantization
# ---------------------------------------------------------------------------

def quantize_weights(model: nn.Module, bits: int) -> dict:
    """
    Quantize Conv2d and Linear weights in-place (simulated quantization).
    Original weights are replaced with dequantized approximations.

    Returns quant_params dict: {layer_name: {scale, zero_point, bits}}
    """
    if bits >= 32:
        return {}

    quant_params = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            w_dq, scale, zp = quantize_tensor(module.weight.data, bits)
            module.weight.data = w_dq
            quant_params[name] = {'scale': scale, 'zero_point': zp, 'bits': bits}

    return quant_params


# ---------------------------------------------------------------------------
# Activation quantization via forward hooks
# ---------------------------------------------------------------------------

class ActivationQuantizer:
    """
    Applies quantization to layer outputs during inference using pre-calibrated stats.
    """

    def __init__(self):
        self.stats   = {}   # name -> (x_min, x_max)
        self.handles = []

    def calibrate(self, model: nn.Module, loader, device, num_batches: int = 20):
        """Collect per-layer activation min/max over calibration data."""
        running = {}

        def make_hook(name):
            def hook(module, inp, out):
                v_min = out.detach().min().item()
                v_max = out.detach().max().item()
                if name not in running:
                    running[name] = [v_min, v_max]
                else:
                    running[name][0] = min(running[name][0], v_min)
                    running[name][1] = max(running[name][1], v_max)
            return hook

        handles = []
        for name, mod in model.named_modules():
            if isinstance(mod, (nn.Conv2d, nn.Linear, nn.ReLU6)):
                handles.append(mod.register_forward_hook(make_hook(name)))

        model.eval()
        with torch.no_grad():
            for i, (imgs, _) in enumerate(loader):
                if i >= num_batches:
                    break
                model(imgs.to(device))

        for h in handles:
            h.remove()

        self.stats = {k: tuple(v) for k, v in running.items()}
        return self.stats

    def apply(self, model: nn.Module, bits: int):
        """Register forward hooks that quantize activations during inference."""
        self.remove()
        if bits >= 32:
            return

        for name, mod in model.named_modules():
            if isinstance(mod, (nn.Conv2d, nn.Linear, nn.ReLU6)):
                stat = self.stats.get(name)
                if stat is None:
                    continue
                x_min, x_max = stat

                qmin, qmax = 0, (1 << bits) - 1
                if x_min == x_max:
                    continue
                scale = (x_max - x_min) / (qmax - qmin)
                zp    = max(qmin, min(qmax, int(round(-x_min / scale))))

                def make_hook(s, z, qmn, qmx):
                    def hook(module, inp, out):
                        q   = torch.clamp(torch.round(out / s) + z, qmn, qmx)
                        return (q - z) * s
                    return hook

                self.handles.append(
                    mod.register_forward_hook(make_hook(scale, zp, qmin, qmax))
                )

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


# ---------------------------------------------------------------------------
# Size & compression ratio calculation
# ---------------------------------------------------------------------------

def compute_model_sizes(model: nn.Module, weight_bits: int,
                        activation_bits: int, act_stats: dict) -> dict:
    """
    Compute memory sizes (in bytes) for original and quantized model.

    Weight storage:
        original   : num_params × 4 bytes (float32)
        quantized  : num_params × ceil(bits/8) + num_layers × 8 (overhead)

    Activation storage (peak, single sample):
        Measured as the sum of output tensor sizes across all Conv/Linear layers.
        original   : total_act_elements × 4 bytes
        quantized  : total_act_elements × ceil(bits/8) + num_act_layers × 8
    """
    # --- weights ---
    weight_params = 0
    num_weight_layers = 0
    for mod in model.modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            weight_params     += mod.weight.numel()
            num_weight_layers += 1

    orig_weight_bytes = weight_params * 4
    if weight_bits >= 32:
        quant_weight_bytes = orig_weight_bytes
    else:
        bytes_per_param    = max(1, weight_bits // 8)
        quant_weight_bytes = (weight_params * bytes_per_param
                              + num_weight_layers * 8)   # 8 = scale+zp overhead

    # --- activations (from calibration stats) ---
    # Approximate: use stat keys that correspond to Conv2d/Linear outputs
    # We estimate activation elements from the model by a single dummy forward pass
    act_elements   = 0
    num_act_layers = 0
    dummy = torch.zeros(1, 3, 32, 32)
    hooks, sizes   = [], []

    def size_hook(m, i, o):
        sizes.append(o.numel())

    for mod in model.modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            hooks.append(mod.register_forward_hook(size_hook))

    model.eval()
    with torch.no_grad():
        try:
            model(dummy)
        except Exception:
            pass

    for h in hooks:
        h.remove()

    act_elements   = sum(sizes)
    num_act_layers = len(sizes)

    orig_act_bytes = act_elements * 4
    if activation_bits >= 32:
        quant_act_bytes = orig_act_bytes
    else:
        bytes_per_act   = max(1, activation_bits // 8)
        quant_act_bytes = (act_elements * bytes_per_act
                           + num_act_layers * 8)

    total_orig  = orig_weight_bytes  + orig_act_bytes
    total_quant = quant_weight_bytes + quant_act_bytes

    return {
        # weights
        'orig_weight_mb':    orig_weight_bytes  / 1e6,
        'quant_weight_mb':   quant_weight_bytes / 1e6,
        'weight_ratio':      orig_weight_bytes  / max(quant_weight_bytes, 1),
        'weight_overhead_b': num_weight_layers  * 8,

        # activations
        'orig_act_mb':       orig_act_bytes  / 1e6,
        'quant_act_mb':      quant_act_bytes / 1e6,
        'act_ratio':         orig_act_bytes  / max(quant_act_bytes, 1),
        'act_overhead_b':    num_act_layers  * 8,
        'act_measurement':   'Sum of output tensor sizes across all Conv2d/Linear layers for a single sample',

        # total model
        'orig_model_mb':    total_orig  / 1e6,
        'quant_model_mb':   total_quant / 1e6,
        'model_ratio':      total_orig  / max(total_quant, 1),

        # misc
        'weight_params':    weight_params,
        'num_weight_layers':num_weight_layers,
        'act_elements':     act_elements,
        'num_act_layers':   num_act_layers,
    }
