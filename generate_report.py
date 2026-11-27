"""
generate_report.py - Generate Assignment 2 PDF report using fpdf2.
"""

import os
import sys
from fpdf import FPDF

ASSETS = './report_assets'
os.makedirs(ASSETS, exist_ok=True)

# ── Helpers ──────────────────────────────────────────────────────────────────

class PDF(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font('Helvetica', 'I', 8)
            self.set_text_color(120, 120, 120)
            self.cell(0, 8, 'CS6886W Assignment 2 - MobileNet-v2 Compression', align='C')
            self.ln(2)
            self.set_draw_color(180, 180, 180)
            self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
            self.ln(3)
            self.set_text_color(0, 0, 0)

    def footer(self):
        self.set_y(-13)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')
        self.set_text_color(0, 0, 0)

    def section(self, title, level=1):
        """Print a section or sub-section heading."""
        if level == 1:
            self.ln(4)
            self.set_font('Helvetica', 'B', 13)
            self.set_fill_color(220, 230, 245)
            self.cell(0, 8, title, new_x='LMARGIN', new_y='NEXT', fill=True)
            self.ln(2)
        else:
            self.ln(2)
            self.set_font('Helvetica', 'B', 11)
            self.cell(0, 7, title, new_x='LMARGIN', new_y='NEXT')
            self.ln(1)
        self.set_font('Helvetica', '', 10)

    def body(self, text, indent=0):
        """Print a paragraph of body text."""
        self.set_font('Helvetica', '', 10)
        self.set_x(self.l_margin + indent)
        self.multi_cell(self.epw - indent, 5.5, text)
        self.ln(1)

    def bullet(self, text, indent=5):
        self.set_font('Helvetica', '', 10)
        self.set_x(self.l_margin + indent)
        self.cell(5, 5.5, chr(149))
        self.multi_cell(self.epw - indent - 5, 5.5, text)

    def kv(self, key, value, indent=5):
        self.set_x(self.l_margin + indent)
        self.set_font('Helvetica', 'B', 10)
        self.cell(55, 5.5, key + ':')
        self.set_font('Helvetica', '', 10)
        self.multi_cell(self.epw - indent - 55, 5.5, value)

    def add_image_full(self, path, caption='', h=70):
        if not os.path.exists(path):
            self.body(f'[Figure not available: {path}]')
            return
        # Add page break if image won't fit on current page
        if self.will_page_break(h + 10):
            self.add_page()
        avail_w = self.epw
        self.image(path, x=self.l_margin, w=avail_w, h=h)
        if caption:
            self.ln(1)
            self.set_font('Helvetica', 'I', 9)
            self.set_text_color(80, 80, 80)
            self.cell(0, 5, caption, align='C')
            self.set_text_color(0, 0, 0)
            self.ln(3)
        else:
            self.ln(2)

    def table_row(self, cells, widths, bold=False, fill=False):
        if fill:
            self.set_fill_color(235, 240, 250)
        self.set_font('Helvetica', 'B' if bold else '', 9)
        for cell, w in zip(cells, widths):
            self.cell(w, 6, str(cell), border=1, align='C', fill=fill)
        self.ln()


# ── Build the PDF ─────────────────────────────────────────────────────────────

pdf = PDF(orientation='P', unit='mm', format='A4')
pdf.set_margins(20, 20, 20)
pdf.set_auto_page_break(auto=True, margin=18)
pdf.set_font('Helvetica', '', 10)

# ── Title Page ───────────────────────────────────────────────────────────────
pdf.add_page()
pdf.ln(20)
pdf.set_font('Helvetica', 'B', 20)
pdf.cell(0, 12, 'CS6886W - Systems Engineering for Deep Learning', align='C', new_x='LMARGIN', new_y='NEXT')
pdf.set_font('Helvetica', 'B', 15)
pdf.cell(0, 10, 'Assignment 2: MobileNet-v2 with Custom Quantization', align='C', new_x='LMARGIN', new_y='NEXT')
pdf.ln(10)
pdf.set_font('Helvetica', 'B', 11)
pdf.cell(0, 7, 'Roll No: CS25M505', align='C', new_x='LMARGIN', new_y='NEXT')
pdf.ln(15)
pdf.set_font('Helvetica', 'B', 11)
pdf.set_fill_color(240, 240, 255)
pdf.cell(0, 8, 'GitHub Repository: https://github.com/Ashishkumar-V/mobilenet-cifar10', align='C', fill=True, new_x='LMARGIN', new_y='NEXT')
pdf.ln(15)

# Summary box
pdf.set_fill_color(245, 248, 255)
pdf.set_draw_color(100, 130, 200)
pdf.rect(pdf.l_margin, pdf.get_y(), pdf.epw, 55, 'DF')
pdf.set_y(pdf.get_y() + 3)
pdf.set_font('Helvetica', 'B', 10)
pdf.cell(0, 6, 'Summary', align='C', new_x='LMARGIN', new_y='NEXT')
pdf.set_font('Helvetica', '', 9)
pdf.ln(1)
summary_lines = [
    ('Baseline test accuracy (no compression)',  '100% (note: overfit due to limited data - see Q1c)'),
    ('Original model size (float32)',            '15.37 MB  (8.95 MB weights + 6.42 MB activations)'),
    ('Best model compression ratio',            '4.00x  (8-bit weight + 8-bit activation)'),
    ('Best weight compression ratio',           '4.00x  (8-bit)'),
    ('Best activation compression ratio',       '4.00x  (8-bit)'),
    ('Quantized model size (8-bit)',             '3.84 MB'),
    ('Storage overhead',                        '848 bytes  (scale + zero_point per layer)'),
    ('Method',                                  'Post-Training Quantization (PTQ), uniform asymmetric'),
]
for k, v in summary_lines:
    pdf.set_x(pdf.l_margin + 5)
    pdf.set_font('Helvetica', 'B', 9)
    pdf.cell(85, 5.5, k + ':')
    pdf.set_font('Helvetica', '', 9)
    pdf.cell(0, 5.5, v)
    pdf.ln()

pdf.set_draw_color(0, 0, 0)

# ── Q1: Training Baseline ─────────────────────────────────────────────────────
pdf.add_page()
pdf.section('Question 1 - Training Baseline')

pdf.section('Q1(a) - CIFAR-10 Data Preparation and Augmentation', level=2)
pdf.body(
    'CIFAR-10 consists of 60,000 32x32 RGB images in 10 classes. '
    'The dataset was split into 45,000 training, 5,000 validation, and 10,000 test images using '
    'a random split with seed=42, ensuring reproducibility.'
)
pdf.body('Training transforms (applied in order):')
transforms = [
    'transforms.RandomCrop(32, padding=4)       - random crop with 4-pixel zero-padding',
    'transforms.RandomHorizontalFlip(p=0.5)      - horizontal flip with 50% probability',
    'transforms.ToTensor()                        - convert PIL image to [0,1] float tensor',
    'transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],\n'
    '                      std=[0.2023, 0.1994, 0.2010])  - per-channel normalisation (CIFAR-10 statistics)',
    'Cutout(n_holes=1, length=16)                - randomly masks a 16x16 patch to zero; forces robustness',
]
for t in transforms:
    pdf.bullet(t)

pdf.ln(2)
pdf.body('Validation/test transforms: ToTensor + same Normalize (no random augmentation).')

pdf.section('Q1(b) - MobileNet-v2 Configuration and Training Strategy', level=2)
pdf.body('Model configuration:')
model_cfg = [
    ('Architecture', 'MobileNet-v2 (inverted residuals + depthwise separable convolutions)'),
    ('CIFAR-10 adaptation', 'Initial conv stride=1 (not 2); second bottleneck block stride=1 - prevents '
     'aggressive spatial downsampling on 32x32 inputs'),
    ('Width multiplier', '1.0  (no channel scaling)'),
    ('Dropout', '0.2 (applied before the final classifier)'),
    ('Batch Normalisation', 'Standard BN after every Conv2d (momentum=0.1, eps=1e-5)'),
    ('Parameters', '2,236,682 total  (~8.95 MB in float32)'),
]
for k, v in model_cfg:
    pdf.kv(k, v)

pdf.ln(2)
pdf.body('Training strategy:')
train_cfg = [
    ('Optimizer',    'SGD (momentum=0.9, weight_decay=1e-4)'),
    ('Learning rate','0.01 (initial)'),
    ('LR schedule',  'CosineAnnealingLR(T_max=50)  - cosine decay from 0.01 to 0 over 50 epochs'),
    ('Batch size',   '128'),
    ('Epochs',       '50'),
    ('Loss',         'CrossEntropyLoss'),
    ('Regularisation','L2 weight decay (1e-4) + Cutout + Dropout (0.2)'),
    ('Seed',         '42  (Python, NumPy, PyTorch - deterministic training)'),
]
for k, v in train_cfg:
    pdf.kv(k, v)

pdf.section('Q1(c) - Test Accuracy, Training Curves, and Failure Modes', level=2)

pdf.body(
    'Final test top-1 accuracy: 100.00%  (see failure mode note below).\n'
    'The model converged to 100% validation accuracy by epoch 8 and remained there for all subsequent epochs.'
)

pdf.ln(1)
# Insert training curves
if os.path.exists(f'{ASSETS}/loss_curve.png'):
    pdf.add_image_full(f'{ASSETS}/loss_curve.png',
                       caption='Figure 1: Training and validation loss (log scale) over 50 epochs.',
                       h=62)
if os.path.exists(f'{ASSETS}/accuracy_curve.png'):
    pdf.add_image_full(f'{ASSETS}/accuracy_curve.png',
                       caption='Figure 2: Training and validation top-1 accuracy over 50 epochs.',
                       h=62)

pdf.set_font('Helvetica', 'B', 10)
pdf.cell(0, 6, 'Failure Mode - Data Leakage / Overfitting:', new_x='LMARGIN', new_y='NEXT')
pdf.set_font('Helvetica', '', 10)
pdf.body(
    'The 100% test accuracy is not representative of real generalisation. Due to storage '
    'constraints on the CPU-only machine used for this assignment, only data_batch_4 of '
    'CIFAR-10 was physically available. The other four training batch files were symlinked '
    'to data_batch_4, so the model effectively saw the same 10,000 images five times. '
    'The validation and test sets were also derived from this same batch, causing severe '
    'data leakage. '
    'With the full CIFAR-10 dataset (50,000 training images) and GPU training, '
    'MobileNet-v2 (width_mult=1.0) is expected to achieve 93-95% top-1 accuracy '
    'on CIFAR-10 as reported in the literature.'
)

# ── Q2: Compression Implementation ───────────────────────────────────────────
pdf.add_page()
pdf.section('Question 2 - Model Compression Implementation')

pdf.section('Q2(a) - Compression Method Design', level=2)
pdf.body(
    'I implemented Post-Training Quantization (PTQ) using uniform asymmetric quantization '
    'without using any compression library or API. PTQ was chosen since it does not require '
    'retraining - the trained float32 model is compressed in a single pass.'
)

pdf.body('Quantization formula:')
pdf.set_font('Courier', '', 9)
pdf.set_fill_color(248, 248, 248)
formula_lines = [
    '  scale      = (x_max - x_min) / (2^bits - 1)',
    '  zero_point = clamp(round(-x_min / scale), 0, 2^bits - 1)',
    '  q          = clamp(round(x / scale) + zero_point, 0, 2^bits - 1)   [integer]',
    '  x_approx   = (q - zero_point) * scale                               [dequantized]',
]
for line in formula_lines:
    pdf.set_x(pdf.l_margin + 5)
    pdf.cell(0, 5.5, line, new_x='LMARGIN', new_y='NEXT', fill=True)
pdf.ln(2)
pdf.set_font('Helvetica', '', 10)

pdf.body(
    'Asymmetric quantization uses a per-tensor scale and zero_point to map the full '
    'float range [x_min, x_max] onto the integer range [0, 2^bits-1]. '
    'This handles ReLU6 activations (always non-negative) and weights (symmetric or '
    'asymmetric distribution) uniformly without special casing.'
)

pdf.body('The compression has two steps:')
stages = [
    ('Step 1 - Weight quantization',
     'All Conv2d and Linear weight tensors are quantized in-place. The float32 weights '
     'are replaced with dequantized approximations, simulating low-bit weight storage.'),
    ('Step 2 - Activation quantization',
     'A calibration pass over 20 mini-batches collects per-layer activation min/max statistics. '
     'Forward hooks are then registered on Conv2d/Linear/ReLU6 modules. During inference, '
     'each hook quantizes and dequantizes the layer output to simulate low-bit activation storage.'),
]
for title, desc in stages:
    pdf.set_font('Helvetica', 'B', 10)
    pdf.set_x(pdf.l_margin + 3)
    pdf.cell(0, 6, title, new_x='LMARGIN', new_y='NEXT')
    pdf.set_font('Helvetica', '', 10)
    pdf.set_x(pdf.l_margin + 6)
    pdf.multi_cell(pdf.epw - 6, 5.5, desc)
    pdf.ln(2)

pdf.section('Q2(b) - Layers Compressed and Exceptions', level=2)
pdf.body(
    'All Conv2d and Linear layers in the network are quantized - both weights and activations. '
    'BatchNorm layers are not quantized: their scale (gamma) and shift (beta) parameters are '
    'small vectors (channel-wise scalars) and their contribution to total memory is negligible '
    '(< 0.3%). The ReLU6 activation function itself has no learnable parameters; however, '
    'the output tensor of ReLU6 is quantized via forward hooks during activation quantization.'
)

pdf.body('Layer breakdown for MobileNet-v2 (CIFAR-10):')
layer_rows = [
    ('Initial Conv2d 3->32',          'Yes', 'Yes'),
    ('Inverted Residual blocks (x17)','Yes (all Conv2d within each block)', 'Yes'),
    ('- expand 1x1 conv (t > 1)',     'Yes', 'Yes'),
    ('- depthwise 3x3 conv',          'Yes', 'Yes'),
    ('- project 1x1 conv',            'Yes', 'Yes'),
    ('Final 1x1 Conv2d 320->1280',     'Yes', 'Yes'),
    ('Classifier Linear 1280->10',     'Yes', 'Yes'),
    ('BatchNorm layers (all)',         'No  (excluded)', 'No  (excluded)'),
    ('Total quantized layers',         '53  Conv2d + Linear', '53  Conv2d + Linear'),
]
col_w = [80, 48, 48]
pdf.table_row(['Layer', 'Weight Quantized', 'Activation Quantized'], col_w, bold=True, fill=True)
for i, row in enumerate(layer_rows):
    pdf.table_row(row, col_w, fill=(i % 2 == 0))
pdf.ln(3)

pdf.section('Q2(c) - Storage Overhead', level=2)
pdf.body(
    'Each quantized tensor requires two additional values to enable dequantization: '
    'a float32 scale (4 bytes) and an int32 zero_point (4 bytes) = 8 bytes per tensor. '
    'These are stored alongside the quantized integers.'
)
pdf.body('Total overhead calculation:')
pdf.kv('Layers with quantized weights', '53  (Conv2d + Linear)')
pdf.kv('Layers with quantized activations', '53  (Conv2d + Linear; hooks track the same set)')
pdf.kv('Overhead per layer per tensor', '8 bytes  (4 B scale + 4 B zero_point)')
pdf.kv('Total weight overhead', '53 x 8  = 424 bytes')
pdf.kv('Total activation overhead', '53 x 8  = 424 bytes')
pdf.kv('Grand total storage overhead', '848 bytes  (~0.83 KB)')
pdf.ln(1)
pdf.body(
    'At 848 bytes, the metadata overhead is negligible relative to even the most compressed '
    'model (3.84 MB for 8-bit), constituting only 0.022% of the quantized model size.'
)

# ── Q3: Compression Results ───────────────────────────────────────────────────
pdf.add_page()
pdf.section('Question 3 - Compression Results')

pdf.section('Q3(a) - Compression Pipeline at Different Bit-widths', level=2)
pdf.body(
    'The W&B sweep explored a 6x4 grid: weight_quant_bits in {2, 4, 6, 8, 16, 32} '
    'and activation_quant_bits in {2, 4, 8, 32} - 24 total configurations. '
    'Each run loads the same float32 checkpoint, applies the two-stage PTQ pipeline, '
    'and evaluates on the full test set.'
)

pdf.section('Q3(b) - Compression Table and W&B Parallel Coordinates', level=2)

pdf.body(
    'Key results from the sweep (all configs share the same overfit baseline of 100%):',
)

# Table header
col_w2 = [22, 22, 24, 24, 26, 20, 28]
pdf.table_row(
    ['W bits', 'A bits', 'Model comp.', 'Weight comp.', 'Act comp.', 'Size MB', 'Quant acc.'],
    col_w2, bold=True, fill=True
)
table_data = [
    ('2',  '2',  '4.00x', '4.00x', '4.00x', '3.84', '10.03%'),
    ('2',  '8',  '4.00x', '4.00x', '4.00x', '3.84', '10.65%'),
    ('4',  '2',  '4.00x', '4.00x', '4.00x', '3.84', '10.03%'),
    ('4',  '4',  '4.00x', '4.00x', '4.00x', '3.84', '21.63%'),
    ('4',  '8',  '4.00x', '4.00x', '4.00x', '3.84', '88.66%'),
    ('4',  '32', '1.75x', '4.00x', '1.00x', '8.76', '89.20%'),
    ('6',  '8',  '4.00x', '4.00x', '4.00x', '3.84', '99.98%'),
    ('8',  '8',  '4.00x', '4.00x', '4.00x', '3.84', '100.00%'),
    ('8',  '32', '1.75x', '4.00x', '1.00x', '8.76', '100.00%'),
    ('16', '8',  '2.54x', '2.00x', '4.00x', '6.05', '100.00%'),
    ('16', '32', '1.40x', '2.00x', '1.00x', '10.97', '100.00%'),
    ('32', '8',  '1.47x', '1.00x', '4.00x', '10.45', '100.00%'),
    ('32', '32', '1.00x', '1.00x', '1.00x', '15.37', '100.00%'),
]
for i, row in enumerate(table_data):
    pdf.table_row(row, col_w2, fill=(i % 2 == 0))

pdf.ln(3)
pdf.body(
    'Key insight: Activation bit-width is the dominant factor. 2-bit activations destroy '
    'accuracy (10%) regardless of weight precision. 8-bit activations with 6+ bit weights '
    'preserve nearly full accuracy at 4.0x compression. '
    'Note: bytes_per_param = max(1, bits//8), so 2-8 bit all use 1 byte per value (4x). '
    'True sub-byte packing would require a dedicated integer storage engine.'
)

pdf.ln(2)
if os.path.exists(f'{ASSETS}/accuracy_heatmap.png'):
    pdf.add_image_full(f'{ASSETS}/accuracy_heatmap.png',
                       caption='Figure 3: Heatmap of quantized top-1 accuracy (%) across all 24 bit-width configurations.',
                       h=65)

pdf.ln(1)
if os.path.exists(f'{ASSETS}/compression_vs_accuracy.png'):
    pdf.add_image_full(f'{ASSETS}/compression_vs_accuracy.png',
                       caption='Figure 4: Compression ratio vs quantized accuracy across weight bit-widths (activation_bits=8).',
                       h=62)

pdf.ln(2)
pdf.set_fill_color(255, 250, 230)
pdf.set_draw_color(200, 150, 50)
pdf.set_font('Helvetica', 'B', 10)
pdf.cell(0, 6, 'W&B Parallel Coordinates Chart (forge.coreweave.com sweep view):', new_x='LMARGIN', new_y='NEXT')
pdf.set_font('Helvetica', '', 9)
pdf.set_draw_color(0, 0, 0)

parallel_note = (
    'The W&B sweep (ID: vhrnc3dk) with 24 configurations is logged at:\n'
    'https://forge.coreweave.com/wandb/cs25m505-indian-institute-of-technology/'
    'mobilenet-cifar10-compression/sweeps/vhrnc3dk\n\n'
    'The Parallel Coordinates chart shows the axes: activation_quant_bits | weight_quant_bits | '
    'compression_ratio | model_size_mb | quantized_acc.\n'
    'Please refer to the W&B sweep URL above, or see the screenshot below (captured after '
    'all 24 runs complete).'
)
pdf.set_x(pdf.l_margin + 3)
pdf.multi_cell(pdf.epw - 3, 5, parallel_note, border=1, fill=True)
pdf.set_fill_color(255, 255, 255)
pdf.set_draw_color(0, 0, 0)

# Check if parallel coordinates screenshot exists
pc_path = f'{ASSETS}/parallel_coordinates.png'
if os.path.exists(pc_path):
    pdf.ln(2)
    pdf.add_image_full(pc_path,
                       caption='Figure 4: W&B Parallel Coordinates chart - 24-run quantization sweep.',
                       h=75)
else:
    pdf.ln(2)
    pdf.set_font('Helvetica', 'I', 9)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 6, '[Insert W&B Parallel Coordinates screenshot from forge.coreweave.com]',
             align='C', new_x='LMARGIN', new_y='NEXT')
    pdf.set_text_color(0, 0, 0)

# ── Q4: Compression Analysis ──────────────────────────────────────────────────
pdf.add_page()
pdf.section('Question 4 - Compression Analysis')

pdf.section('Q4(a) - Compression Ratio of the Weights', level=2)
pdf.body(
    'Weight compression is computed as the ratio of original float32 weight memory to '
    'quantized weight memory, including the small scale/zero_point overhead:'
)
pdf.set_font('Courier', '', 9)
pdf.set_fill_color(248, 248, 248)
for line in [
    '  orig_weight_bytes  = num_params x 4  (float32)',
    '  quant_weight_bytes = num_params x ceil(bits/8)  +  num_layers x 8  (overhead)',
    '  weight_ratio       = orig_weight_bytes / quant_weight_bytes',
]:
    pdf.set_x(pdf.l_margin + 5)
    pdf.cell(0, 5.5, line, new_x='LMARGIN', new_y='NEXT', fill=True)
pdf.set_font('Helvetica', '', 10)
pdf.ln(2)

pdf.body('Concrete numbers for this model (2,236,682 weight params, 53 weight layers):')
w_rows = [
    ('32-bit (baseline)', '8,946,728', '8,946,728',   '1.00x'),
    ('16-bit',            '8,946,728', '4,473,788',   '2.00x'),
    ('8-bit',             '8,946,728', '2,237,106',   '3.999x'),
    ('4-bit',             '8,946,728', '2,237,106',   '3.999x  (packs to 1 B/param)'),
    ('2-bit',             '8,946,728', '2,237,106',   '3.999x  (packs to 1 B/param)'),
]
w_col = [35, 38, 38, 55]
pdf.table_row(['Bit-width', 'Orig (bytes)', 'Quant (bytes)', 'Compression'], w_col, bold=True, fill=True)
for i, row in enumerate(w_rows):
    pdf.table_row(row, w_col, fill=(i % 2 == 0))
pdf.ln(3)
pdf.body(
    'Best weight compression ratio: 4.00x (achieved at 8-bit and below). '
    'The original weight-only size is 8.95 MB; at 8-bit it becomes 2.24 MB.'
)

pdf.section('Q4(b) - Compression Ratio of the Activations', level=2)
pdf.body(
    'Activation size is estimated as the sum of all Conv2d/Linear layer output tensor sizes '
    'for a single forward pass (batch_size=1, input 3x32x32). This represents peak activation '
    'memory at inference time.'
)
pdf.body('Measurement method:')
pdf.bullet('A dummy forward pass (batch=1) is run through the model.')
pdf.bullet('A forward hook on each Conv2d/Linear layer records the number of elements in the output tensor.')
pdf.bullet('Total activation elements = sum of all hook counts = 1,605,000 elements.')
pdf.bullet('Float32 activation memory = 1,605,000 x 4 = 6,420,000 bytes ~ 6.42 MB.')

pdf.ln(2)
a_rows = [
    ('32-bit (baseline)', '6,420,000', '6,420,000',  '1.00x'),
    ('8-bit',             '6,420,000', '1,605,424',  '3.999x'),
    ('4-bit',             '6,420,000', '1,605,424',  '3.999x'),
    ('2-bit',             '6,420,000', '1,605,424',  '3.999x'),
]
a_col = [35, 38, 38, 55]
pdf.table_row(['Bit-width', 'Orig (bytes)', 'Quant (bytes)', 'Compression'], a_col, bold=True, fill=True)
for i, row in enumerate(a_rows):
    pdf.table_row(row, a_col, fill=(i % 2 == 0))
pdf.ln(3)
pdf.body(
    'Best activation compression ratio: 4.00x (8-bit). '
    'Activations go from 6.42 MB to 1.61 MB.'
)

pdf.section('Q4(v) - Final Approximated Model Size After Compression', level=2)
pdf.body('Total model size = weight size + activation size + storage overhead:')
pdf.kv('Original model size',   '8.95 MB (weights) + 6.42 MB (activations) = 15.37 MB')
pdf.kv('8-bit quantized size',  '2.24 MB (weights) + 1.61 MB (activations) + 0.001 MB (overhead) = 3.84 MB')
pdf.kv('16-bit quantized size', '4.47 MB (weights) + 1.61 MB (activations) + 0.001 MB = 6.08 MB')
pdf.kv('Overall reduction',     '15.37 MB -> 3.84 MB  (4.00x compression, saving 11.53 MB)')
pdf.ln(2)
pdf.body(
    'The final approximated in-memory size of the compressed MobileNet-v2 (8-bit PTQ) is '
    '3.84 MB - a 75% reduction from the original 15.37 MB, with negligible accuracy '
    'degradation at 8 bits (see Q3 table).'
)

# ── Q5: Reproducibility ───────────────────────────────────────────────────────
pdf.section('Question 5 - Reproducibility and Repository')

pdf.section('Q5(a) - Codebase Structure', level=2)
pdf.body('The code is split into separate files, each with a single responsibility:')
files = [
    ('model.py',    'MobileNet-v2 architecture adapted for CIFAR-10 (32x32 input)'),
    ('dataset.py',  'CIFAR-10 data loading, augmentation, and 45k/5k/10k splits'),
    ('engine.py',   'Training loop, evaluation loop, and optimizer factory'),
    ('train.py',    'Training script with W&B logging and checkpoint saving'),
    ('quantize.py', 'Custom PTQ: quantize_tensor, quantize_weights, ActivationQuantizer, compute_model_sizes'),
    ('test.py',     'Compression evaluation with detailed size/ratio reporting'),
    ('sweep.py',    'W&B sweep over 24 bit-width configurations'),
]
f_col = [35, 135]
pdf.table_row(['File', 'Responsibility'], f_col, bold=True, fill=True)
for i, (fname, desc) in enumerate(files):
    pdf.table_row([fname, desc], f_col, fill=(i % 2 == 0))
pdf.ln(3)

pdf.section('Q5(b) - Exact Reproduction Commands', level=2)
pdf.set_font('Courier', 'B', 9)
pdf.set_fill_color(240, 245, 255)
commands = [
    '# 1. Install dependencies',
    'pip install torch>=2.0.0 torchvision>=0.15.0 wandb>=0.15.0 numpy fpdf2 matplotlib',
    '',
    '# 2. Clone repository',
    'git clone https://github.com/Ashishkumar-V/mobilenet-cifar10.git',
    'cd mobilenet-cifar10',
    '',
    '# 3. Download CIFAR-10 (auto-downloads)',
    'python train.py --epochs 50 --lr 0.01 --optimizer sgd --batch_size 128',
    '',
    '# 4. Evaluate 8-bit quantization',
    'python test.py --weight_quant_bits 8 --activation_quant_bits 8',
    '',
    '# 5. Run full W&B sweep (24 configs)',
    'python sweep.py --count 24 --project mobilenet-cifar10-compression',
]
for cmd in commands:
    pdf.set_x(pdf.l_margin + 3)
    if cmd == '':
        pdf.ln(2)
    else:
        pdf.cell(0, 5, cmd, new_x='LMARGIN', new_y='NEXT', fill=True)
pdf.set_font('Helvetica', '', 10)
pdf.ln(3)

pdf.body('Seed configuration (applied in train.py before any data loading):')
pdf.set_font('Courier', '', 9)
seed_lines = [
    '  import random, numpy as np, torch',
    '  random.seed(42)',
    '  np.random.seed(42)',
    '  torch.manual_seed(42)',
    '  torch.cuda.manual_seed_all(42)',
    '  torch.backends.cudnn.deterministic = True',
    '  torch.backends.cudnn.benchmark     = False',
]
for line in seed_lines:
    pdf.set_x(pdf.l_margin + 3)
    pdf.cell(0, 5, line, new_x='LMARGIN', new_y='NEXT', fill=True)
pdf.set_font('Helvetica', '', 10)
pdf.ln(3)

pdf.body('Environment:')
env_items = [
    'Python 3.10+',
    'torch 2.0.0+   (CPU or CUDA)',
    'torchvision 0.15.0+',
    'wandb 0.15.0+',
    'numpy < 2.0',
    'Pillow 9.0.0+',
]
for item in env_items:
    pdf.bullet(item)

pdf.section('Q5(c) - GitHub Repository Link', level=2)
pdf.set_font('Helvetica', 'B', 11)
pdf.cell(0, 7, 'https://github.com/Ashishkumar-V/mobilenet-cifar10', align='C', new_x='LMARGIN', new_y='NEXT')
pdf.ln(2)
pdf.set_font('Helvetica', '', 10)
pdf.body(
    'The repository contains all Python source files, the README with exact commands and '
    'environment specifications, and the best model checkpoint '
    '(best_mobilenet_bs128_optsgd_lr0.01.pth). The W&B training run and compression sweep '
    'results are logged to the institutional forge.coreweave.com project '
    'mobilenet-cifar10 and mobilenet-cifar10-compression respectively.'
)

# ── References ───────────────────────────────────────────────────────────────
pdf.section('References')
refs = [
    '[1] PyTorch MobileNet-v2 reference implementation:\n'
    '    https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py',
    '[2] Template quantization repository for this assignment (VGG16 baseline, adapted to MobileNet-v2):\n'
    '    https://github.com/DextroLaev/CS6886-Sys_for_dl-Assignment3',
    '[3] Sandler et al., "MobileNetV2: Inverted Residuals and Linear Bottlenecks", CVPR 2018.',
]
for ref in refs:
    pdf.set_font('Helvetica', '', 10)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(pdf.epw, 5.5, ref)
    pdf.ln(2)

# ── Save ─────────────────────────────────────────────────────────────────────
out_path = './report_assets/CS25M505_A2_Report.pdf'
pdf.output(out_path)
print(f'PDF saved: {out_path}')
