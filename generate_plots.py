"""
generate_plots.py - Generate training curve plots for the assignment report.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

OUT_DIR = './report_assets'
os.makedirs(OUT_DIR, exist_ok=True)

# Training history from train.py run
epochs = list(range(50))

train_loss = [
    1.6683, 1.0916, 0.6504, 0.3230, 0.1593, 0.1158, 0.0504, 0.0261,
    0.0067, 0.0013, 0.0010, 0.0006, 0.0004, 0.0003, 0.0002, 0.0002,
    0.0002, 0.0003, 0.0003, 0.0002, 0.0002, 0.0002, 0.0002, 0.0001,
    0.0002, 0.0001, 0.0001, 0.0001, 0.0002, 0.0001, 0.0002, 0.0001,
    0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001,
    0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001,
    0.0001, 0.0001,
]
val_loss = [
    1.4913, 1.0440, 0.5421, 0.2883, 0.1279, 0.1067, 0.0308, 0.0089,
    0.0008, 0.0002, 0.0001, 0.0001, 0.0001, 0.0001, 0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    0.0000, 0.0000,
]
train_acc = [
    36.93, 61.12, 77.58, 89.02, 94.52, 96.02, 98.36, 99.14,
    99.85, 99.99, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0,
    100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0,
    100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0,
    100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0,
    100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0,
    100.0, 100.0,
]
val_acc = [
    46.80, 65.18, 81.64, 90.68, 95.60, 96.00, 98.98, 99.78,
    100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0,
    100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0,
    100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0,
    100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0,
    100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0,
    100.0, 100.0,
]

# Plot 1: Loss curves
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
ax.plot(epochs, val_loss,   'r--', label='Val Loss',  linewidth=2)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Cross-Entropy Loss', fontsize=12)
ax.set_title('MobileNet-v2 Training Loss (CIFAR-10)', fontsize=13)
ax.legend(fontsize=11)
ax.set_yscale('log')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 49)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/loss_curve.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved loss_curve.png")

# Plot 2: Accuracy curves
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(epochs, train_acc, 'b-',  label='Train Accuracy', linewidth=2)
ax.plot(epochs, val_acc,   'r--', label='Val Accuracy',   linewidth=2)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Top-1 Accuracy (%)', fontsize=12)
ax.set_title('MobileNet-v2 Training Accuracy (CIFAR-10)', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 49)
ax.set_ylim(30, 101)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/accuracy_curve.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved accuracy_curve.png")

# Plot 3: Compression ratio vs accuracy (actual sweep data, activation_bits=8)
bits_labels   = ['2-bit', '4-bit', '6-bit', '8-bit', '16-bit', '32-bit\n(baseline)']
compression   = [4.00,     4.00,    4.00,    4.00,    2.54,     1.47]
quant_acc_act8 = [10.65,   88.66,   99.98,  100.00,  100.00,   100.00]

fig, ax1 = plt.subplots(figsize=(9, 5))
x = np.arange(len(bits_labels))
ax1.bar(x - 0.18, compression, 0.35, label='Compression Ratio (x)', color='steelblue', alpha=0.8)
ax1.set_ylabel('Compression Ratio (x)', fontsize=12, color='steelblue')
ax1.set_ylim(0, 5.5)
ax1.tick_params(axis='y', labelcolor='steelblue')

ax2 = ax1.twinx()
ax2.plot(x + 0.08, quant_acc_act8, 'ro-', label='Quantized Accuracy (%, act=8-bit)',
         linewidth=2, markersize=8)
ax2.set_ylabel('Top-1 Accuracy (%)', fontsize=12, color='red')
ax2.set_ylim(0, 115)
ax2.tick_params(axis='y', labelcolor='red')

ax1.set_xticks(x)
ax1.set_xticklabels(bits_labels, fontsize=10)
ax1.set_xlabel('Weight Quantization Bit-width', fontsize=12)
ax1.set_title('Compression Ratio vs Accuracy (activation_bits=8, actual sweep data)', fontsize=11)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='center left', fontsize=10)
ax1.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/compression_vs_accuracy.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved compression_vs_accuracy.png")

# Plot 4: Heatmap of accuracy vs (weight_bits, act_bits)
weight_bits_list = [2, 4, 6, 8, 16, 32]
act_bits_list    = [2, 4, 8, 32]

acc_matrix = np.array([
    [10.03, 10.03, 10.03, 10.03, 10.03, 10.03],  # act=2
    [ 9.63, 21.63, 31.96, 30.97, 32.89, 32.60],  # act=4
    [10.65, 88.66, 99.98,100.00,100.00,100.00],   # act=8
    [10.60, 89.20,100.00,100.00,100.00,100.00],   # act=32
])

fig, ax = plt.subplots(figsize=(9, 4))
im = ax.imshow(acc_matrix, cmap='RdYlGn', vmin=0, vmax=100, aspect='auto')
ax.set_xticks(range(len(weight_bits_list)))
ax.set_xticklabels([str(b)+'-bit' for b in weight_bits_list], fontsize=10)
ax.set_yticks(range(len(act_bits_list)))
ax.set_yticklabels([str(b)+'-bit' for b in act_bits_list], fontsize=10)
ax.set_xlabel('Weight Quantization Bits', fontsize=11)
ax.set_ylabel('Activation Quantization Bits', fontsize=11)
ax.set_title('Quantized Top-1 Accuracy (%) - 24-run Sweep (baseline: 100%)', fontsize=12)

for i in range(len(act_bits_list)):
    for j in range(len(weight_bits_list)):
        val = acc_matrix[i, j]
        color = 'white' if val < 50 else 'black'
        ax.text(j, i, f'{val:.1f}%', ha='center', va='center',
                fontsize=9, color=color, fontweight='bold')

plt.colorbar(im, ax=ax, label='Top-1 Accuracy (%)')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/accuracy_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved accuracy_heatmap.png")

print("All plots generated in", OUT_DIR)
