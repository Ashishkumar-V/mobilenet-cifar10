"""
generate_docx.py - Generate Assignment 2 Word document report.
"""

import os
from docx import Document
from docx.shared import Pt, Inches, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

ASSETS = './report_assets'

doc = Document()

# ── Page margins ────────────────────────────────────────────────────────────
for section in doc.sections:
    section.top_margin    = Cm(2.0)
    section.bottom_margin = Cm(2.0)
    section.left_margin   = Cm(2.5)
    section.right_margin  = Cm(2.5)

# ── Style helpers ────────────────────────────────────────────────────────────

def set_cell_bg(cell, hex_color):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), hex_color)
    tcPr.append(shd)

def heading1(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    run = p.add_run(text)
    run.bold = True
    run.font.size = Pt(13)
    run.font.color.rgb = RGBColor(0x1A, 0x3A, 0x6B)
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after  = Pt(4)
    # bottom border to mimic section heading
    pPr = p._p.get_or_add_pPr()
    pBdr = OxmlElement('w:pBdr')
    bottom = OxmlElement('w:bottom')
    bottom.set(qn('w:val'),   'single')
    bottom.set(qn('w:sz'),    '4')
    bottom.set(qn('w:space'), '1')
    bottom.set(qn('w:color'), '7090C0')
    pBdr.append(bottom)
    pPr.append(pBdr)
    return p

def heading2(text):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = True
    run.font.size = Pt(11)
    run.font.color.rgb = RGBColor(0x1A, 0x3A, 0x6B)
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after  = Pt(3)
    return p

def body(text):
    p = doc.add_paragraph(text)
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after  = Pt(4)
    p.style.font.size = Pt(10)
    return p

def bullet(text):
    p = doc.add_paragraph(style='List Bullet')
    p.add_run(text).font.size = Pt(10)
    p.paragraph_format.space_before = Pt(1)
    p.paragraph_format.space_after  = Pt(1)
    return p

def kv_line(key, value):
    p = doc.add_paragraph()
    r1 = p.add_run(key + ':  ')
    r1.bold = True
    r1.font.size = Pt(10)
    r2 = p.add_run(value)
    r2.font.size = Pt(10)
    p.paragraph_format.space_before = Pt(1)
    p.paragraph_format.space_after  = Pt(1)
    return p

def code_block(lines):
    for line in lines:
        p = doc.add_paragraph()
        run = p.add_run(line if line else ' ')
        run.font.name = 'Courier New'
        run.font.size = Pt(9)
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after  = Pt(0)
        # light grey shading
        pPr = p._p.get_or_add_pPr()
        shd = OxmlElement('w:shd')
        shd.set(qn('w:val'), 'clear')
        shd.set(qn('w:color'), 'auto')
        shd.set(qn('w:fill'), 'F5F5F5')
        pPr.append(shd)

def caption(text):
    p = doc.add_paragraph(text)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.runs[0]
    run.italic = True
    run.font.size = Pt(9)
    run.font.color.rgb = RGBColor(80, 80, 80)
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after  = Pt(6)

def add_image(path, cap='', width_in=5.5):
    if os.path.exists(path):
        doc.add_picture(path, width=Inches(width_in))
        last = doc.paragraphs[-1]
        last.alignment = WD_ALIGN_PARAGRAPH.CENTER
        if cap:
            caption(cap)
    else:
        body(f'[Figure not available: {path}]')

def simple_table(headers, rows, col_widths_cm, header_bg='DCE6F5'):
    t = doc.add_table(rows=1 + len(rows), cols=len(headers))
    t.style = 'Table Grid'
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    # header row
    hrow = t.rows[0]
    for i, (h, w) in enumerate(zip(headers, col_widths_cm)):
        cell = hrow.cells[i]
        cell.width = Cm(w)
        cell.text = h
        cell.paragraphs[0].runs[0].bold = True
        cell.paragraphs[0].runs[0].font.size = Pt(9)
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_cell_bg(cell, header_bg)
    # data rows
    for ri, row in enumerate(rows):
        trow = t.rows[ri + 1]
        bg = 'EBF0FA' if ri % 2 == 0 else 'FFFFFF'
        for ci, (val, w) in enumerate(zip(row, col_widths_cm)):
            cell = trow.cells[ci]
            cell.width = Cm(w)
            cell.text = str(val)
            cell.paragraphs[0].runs[0].font.size = Pt(9)
            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            set_cell_bg(cell, bg)
    doc.add_paragraph()

# ════════════════════════════════════════════════════════════════════════════
# Title Page
# ════════════════════════════════════════════════════════════════════════════
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run('CS6886W - Systems Engineering for Deep Learning')
r.bold = True
r.font.size = Pt(20)
p.paragraph_format.space_before = Pt(36)
p.paragraph_format.space_after  = Pt(6)

p2 = doc.add_paragraph()
p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
r2 = p2.add_run('Assignment 2: MobileNet-v2 with Custom Quantization')
r2.bold = True
r2.font.size = Pt(15)
p2.paragraph_format.space_after = Pt(16)

p3 = doc.add_paragraph()
p3.alignment = WD_ALIGN_PARAGRAPH.CENTER
r3 = p3.add_run('Roll No: CS25M505')
r3.bold = True
r3.font.size = Pt(11)
p3.paragraph_format.space_after = Pt(20)

p4 = doc.add_paragraph()
p4.alignment = WD_ALIGN_PARAGRAPH.CENTER
r4 = p4.add_run('GitHub Repository: https://github.com/Ashishkumar-V/mobilenet-cifar10')
r4.bold = True
r4.font.size = Pt(11)
p4.paragraph_format.space_after = Pt(24)

# Summary box as a 2-col table
summary_data = [
    ('Baseline test accuracy (no compression)',  '100% (note: overfit due to limited data - see Q1c)'),
    ('Original model size (float32)',            '15.37 MB  (8.95 MB weights + 6.42 MB activations)'),
    ('Best model compression ratio',             '4.00x  (8-bit weight + 8-bit activation)'),
    ('Best weight compression ratio',            '4.00x  (8-bit)'),
    ('Best activation compression ratio',        '4.00x  (8-bit)'),
    ('Quantized model size (8-bit)',              '3.84 MB'),
    ('Storage overhead',                         '848 bytes  (scale + zero_point per layer)'),
    ('Method',                                   'Post-Training Quantization (PTQ), uniform asymmetric'),
]
ps = doc.add_paragraph()
ps.alignment = WD_ALIGN_PARAGRAPH.CENTER
rs = ps.add_run('Summary')
rs.bold = True
rs.font.size = Pt(11)
ps.paragraph_format.space_after = Pt(4)

st = doc.add_table(rows=len(summary_data), cols=2)
st.style = 'Table Grid'
st.alignment = WD_TABLE_ALIGNMENT.CENTER
for ri, (k, v) in enumerate(summary_data):
    bg = 'EEF2FC' if ri % 2 == 0 else 'F8FAFF'
    kc = st.rows[ri].cells[0]
    vc = st.rows[ri].cells[1]
    kc.width = Cm(7.5)
    vc.width = Cm(9.0)
    kc.text = k
    vc.text = v
    kc.paragraphs[0].runs[0].bold = True
    kc.paragraphs[0].runs[0].font.size = Pt(9)
    vc.paragraphs[0].runs[0].font.size = Pt(9)
    set_cell_bg(kc, bg)
    set_cell_bg(vc, bg)
doc.add_paragraph()

doc.add_page_break()

# ════════════════════════════════════════════════════════════════════════════
# Q1: Training Baseline
# ════════════════════════════════════════════════════════════════════════════
heading1('Question 1 - Training Baseline')

heading2('Q1(a) - CIFAR-10 Data Preparation and Augmentation')
body(
    'CIFAR-10 consists of 60,000 32x32 RGB images in 10 classes. '
    'The dataset was split into 45,000 training, 5,000 validation, and 10,000 test images using '
    'a random split with seed=42, ensuring reproducibility.'
)
body('Training transforms (applied in order):')
bullet('transforms.RandomCrop(32, padding=4)       - random crop with 4-pixel zero-padding')
bullet('transforms.RandomHorizontalFlip(p=0.5)      - horizontal flip with 50% probability')
bullet('transforms.ToTensor()                        - convert PIL image to [0,1] float tensor')
bullet('transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])  - per-channel normalisation')
bullet('Cutout(n_holes=1, length=16)                - randomly masks a 16x16 patch to zero; forces robustness')
body('Validation/test transforms: ToTensor + same Normalize (no random augmentation).')

heading2('Q1(b) - MobileNet-v2 Configuration and Training Strategy')
body('Model configuration:')
model_cfg = [
    ('Architecture',        'MobileNet-v2 (inverted residuals + depthwise separable convolutions)'),
    ('CIFAR-10 adaptation', 'Initial conv stride=1 (not 2); second bottleneck block stride=1 - prevents aggressive spatial downsampling on 32x32 inputs'),
    ('Width multiplier',    '1.0  (no channel scaling)'),
    ('Dropout',             '0.2 (applied before the final classifier)'),
    ('Batch Normalisation', 'Standard BN after every Conv2d (momentum=0.1, eps=1e-5)'),
    ('Parameters',          '2,236,682 total  (~8.95 MB in float32)'),
]
for k, v in model_cfg:
    kv_line(k, v)

body('Training strategy:')
train_cfg = [
    ('Optimizer',      'SGD (momentum=0.9, weight_decay=1e-4)'),
    ('Learning rate',  '0.01 (initial)'),
    ('LR schedule',    'CosineAnnealingLR(T_max=50)  - cosine decay from 0.01 to 0 over 50 epochs'),
    ('Batch size',     '128'),
    ('Epochs',         '50'),
    ('Loss',           'CrossEntropyLoss'),
    ('Regularisation', 'L2 weight decay (1e-4) + Cutout + Dropout (0.2)'),
    ('Seed',           '42  (Python, NumPy, PyTorch - deterministic training)'),
]
for k, v in train_cfg:
    kv_line(k, v)

heading2('Q1(c) - Test Accuracy, Training Curves, and Failure Modes')
body(
    'Final test top-1 accuracy: 100.00%  (see failure mode note below).\n'
    'The model converged to 100% validation accuracy by epoch 8 and remained there for all subsequent epochs.'
)
add_image(f'{ASSETS}/loss_curve.png',
          'Figure 1: Training and validation loss (log scale) over 50 epochs.')
add_image(f'{ASSETS}/accuracy_curve.png',
          'Figure 2: Training and validation top-1 accuracy over 50 epochs.')

p_fm = doc.add_paragraph()
r_fm = p_fm.add_run('Failure Mode - Data Leakage / Overfitting:')
r_fm.bold = True
r_fm.font.size = Pt(10)
body(
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

doc.add_page_break()

# ════════════════════════════════════════════════════════════════════════════
# Q2: Compression Implementation
# ════════════════════════════════════════════════════════════════════════════
heading1('Question 2 - Model Compression Implementation')

heading2('Q2(a) - Compression Method Design')
body(
    'I implemented Post-Training Quantization (PTQ) using uniform asymmetric quantization '
    'without using any compression library or API. PTQ was chosen since it does not require '
    'retraining - the trained float32 model is compressed in a single pass.'
)
body('Quantization formula:')
code_block([
    '  scale      = (x_max - x_min) / (2^bits - 1)',
    '  zero_point = clamp(round(-x_min / scale), 0, 2^bits - 1)',
    '  q          = clamp(round(x / scale) + zero_point, 0, 2^bits - 1)   [integer]',
    '  x_approx   = (q - zero_point) * scale                               [dequantized]',
])
doc.add_paragraph()
body(
    'Asymmetric quantization uses a per-tensor scale and zero_point to map the full '
    'float range [x_min, x_max] onto the integer range [0, 2^bits-1]. '
    'This handles ReLU6 activations (always non-negative) and weights (symmetric or '
    'asymmetric distribution) uniformly without special casing.'
)
body('The compression has two steps:')

p_s1 = doc.add_paragraph()
r_s1 = p_s1.add_run('Step 1 - Weight quantization:  ')
r_s1.bold = True
r_s1.font.size = Pt(10)
p_s1.add_run(
    'All Conv2d and Linear weight tensors are quantized in-place. The float32 weights '
    'are replaced with dequantized approximations, simulating low-bit weight storage.'
).font.size = Pt(10)

p_s2 = doc.add_paragraph()
r_s2 = p_s2.add_run('Step 2 - Activation quantization:  ')
r_s2.bold = True
r_s2.font.size = Pt(10)
p_s2.add_run(
    'A calibration pass over 20 mini-batches collects per-layer activation min/max statistics. '
    'Forward hooks are then registered on Conv2d/Linear/ReLU6 modules. During inference, '
    'each hook quantizes and dequantizes the layer output to simulate low-bit activation storage.'
).font.size = Pt(10)

heading2('Q2(b) - Layers Compressed and Exceptions')
body(
    'All Conv2d and Linear layers in the network are quantized - both weights and activations. '
    'BatchNorm layers are not quantized: their scale (gamma) and shift (beta) parameters are '
    'small vectors (channel-wise scalars) and their contribution to total memory is negligible '
    '(< 0.3%). The ReLU6 activation function itself has no learnable parameters; however, '
    'the output tensor of ReLU6 is quantized via forward hooks during activation quantization.'
)
body('Layer breakdown for MobileNet-v2 (CIFAR-10):')

layer_rows = [
    ('Initial Conv2d 3->32',           'Yes',           'Yes'),
    ('Inverted Residual blocks (x17)',  'Yes (sub-rows)', 'Yes'),
    ('- expand 1x1 conv (t > 1)',       'Yes',           'Yes'),
    ('- depthwise 3x3 conv',            'Yes',           'Yes'),
    ('- project 1x1 conv',              'Yes',           'Yes'),
    ('Final 1x1 Conv2d 320->1280',      'Yes',           'Yes'),
    ('Classifier Linear 1280->10',      'Yes',           'Yes'),
    ('BatchNorm layers (all)',           'No (excluded)', 'No (excluded)'),
    ('Total quantized layers',          '53 Conv2d+Lin', '53 Conv2d+Lin'),
]
simple_table(
    ['Layer', 'Weight Quantized', 'Activation Quantized'],
    layer_rows,
    [8.0, 5.0, 5.0]
)

heading2('Q2(c) - Storage Overhead')
body(
    'Each quantized tensor requires two additional values to enable dequantization: '
    'a float32 scale (4 bytes) and an int32 zero_point (4 bytes) = 8 bytes per tensor. '
    'These are stored alongside the quantized integers.'
)
body('Total overhead calculation:')
kv_line('Layers with quantized weights',     '53  (Conv2d + Linear)')
kv_line('Layers with quantized activations', '53  (Conv2d + Linear; hooks track the same set)')
kv_line('Overhead per layer per tensor',     '8 bytes  (4 B scale + 4 B zero_point)')
kv_line('Total weight overhead',             '53 x 8  = 424 bytes')
kv_line('Total activation overhead',         '53 x 8  = 424 bytes')
kv_line('Grand total storage overhead',      '848 bytes  (~0.83 KB)')
body(
    'At 848 bytes, the metadata overhead is negligible relative to even the most compressed '
    'model (3.84 MB for 8-bit), constituting only 0.022% of the quantized model size.'
)

doc.add_page_break()

# ════════════════════════════════════════════════════════════════════════════
# Q3: Compression Results
# ════════════════════════════════════════════════════════════════════════════
heading1('Question 3 - Compression Results')

heading2('Q3(a) - Compression Pipeline at Different Bit-widths')
body(
    'The W&B sweep explored a 6x4 grid: weight_quant_bits in {2, 4, 6, 8, 16, 32} '
    'and activation_quant_bits in {2, 4, 8, 32} - 24 total configurations. '
    'Each run loads the same float32 checkpoint, applies the two-stage PTQ pipeline, '
    'and evaluates on the full test set.'
)

heading2('Q3(b) - Compression Table and W&B Parallel Coordinates')
body('Key results from the sweep (all configs share the same overfit baseline of 100%):')

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
    ('16', '32', '1.40x', '2.00x', '1.00x', '10.97','100.00%'),
    ('32', '8',  '1.47x', '1.00x', '4.00x', '10.45','100.00%'),
    ('32', '32', '1.00x', '1.00x', '1.00x', '15.37','100.00%'),
]
simple_table(
    ['W bits','A bits','Model comp.','Weight comp.','Act comp.','Size MB','Quant acc.'],
    table_data,
    [1.6, 1.6, 2.2, 2.4, 2.1, 1.9, 2.5]
)
body(
    'Key insight: Activation bit-width is the dominant factor. 2-bit activations destroy '
    'accuracy (10%) regardless of weight precision. 8-bit activations with 6+ bit weights '
    'preserve nearly full accuracy at 4.0x compression. '
    'Note: bytes_per_param = max(1, bits//8), so 2-8 bit all use 1 byte per value (4x). '
    'True sub-byte packing would require a dedicated integer storage engine.'
)

add_image(f'{ASSETS}/accuracy_heatmap.png',
          'Figure 3: Heatmap of quantized top-1 accuracy (%) across all 24 bit-width configurations.')
add_image(f'{ASSETS}/compression_vs_accuracy.png',
          'Figure 4: Compression ratio vs quantized accuracy across weight bit-widths (activation_bits=8).')

p_wb = doc.add_paragraph()
r_wb = p_wb.add_run('W&B Parallel Coordinates Chart (forge.coreweave.com sweep view):')
r_wb.bold = True
r_wb.font.size = Pt(10)
body(
    'The W&B sweep (ID: vhrnc3dk) with 24 configurations is logged at:\n'
    'https://forge.coreweave.com/wandb/cs25m505-indian-institute-of-technology/'
    'mobilenet-cifar10-compression/sweeps/vhrnc3dk\n\n'
    'The Parallel Coordinates chart shows the axes: activation_quant_bits | weight_quant_bits | '
    'compression_ratio | model_size_mb | quantized_acc.\n'
    'Please refer to the W&B sweep URL above, or see the screenshot below (captured after '
    'all 24 runs complete).'
)
add_image(f'{ASSETS}/parallel_coordinates.png',
          'Figure 5: W&B Parallel Coordinates chart - 24-run quantization sweep.')

doc.add_page_break()

# ════════════════════════════════════════════════════════════════════════════
# Q4: Compression Analysis
# ════════════════════════════════════════════════════════════════════════════
heading1('Question 4 - Compression Analysis')

heading2('Q4(a) - Compression Ratio of the Weights')
body(
    'Weight compression is computed as the ratio of original float32 weight memory to '
    'quantized weight memory, including the small scale/zero_point overhead:'
)
code_block([
    '  orig_weight_bytes  = num_params x 4  (float32)',
    '  quant_weight_bytes = num_params x ceil(bits/8)  +  num_layers x 8  (overhead)',
    '  weight_ratio       = orig_weight_bytes / quant_weight_bytes',
])
doc.add_paragraph()
body('Concrete numbers for this model (2,236,682 weight params, 53 weight layers):')
w_rows = [
    ('32-bit (baseline)', '8,946,728', '8,946,728', '1.00x'),
    ('16-bit',            '8,946,728', '4,473,788', '2.00x'),
    ('8-bit',             '8,946,728', '2,237,106', '3.999x'),
    ('4-bit',             '8,946,728', '2,237,106', '3.999x (1 B/param)'),
    ('2-bit',             '8,946,728', '2,237,106', '3.999x (1 B/param)'),
]
simple_table(
    ['Bit-width', 'Orig (bytes)', 'Quant (bytes)', 'Compression'],
    w_rows,
    [3.5, 3.8, 3.8, 5.2]
)
body(
    'Best weight compression ratio: 4.00x (achieved at 8-bit and below). '
    'The original weight-only size is 8.95 MB; at 8-bit it becomes 2.24 MB.'
)

heading2('Q4(b) - Compression Ratio of the Activations')
body(
    'Activation size is estimated as the sum of all Conv2d/Linear layer output tensor sizes '
    'for a single forward pass (batch_size=1, input 3x32x32). This represents peak activation '
    'memory at inference time.'
)
body('Measurement method:')
bullet('A dummy forward pass (batch=1) is run through the model.')
bullet('A forward hook on each Conv2d/Linear layer records the number of elements in the output tensor.')
bullet('Total activation elements = sum of all hook counts = 1,605,000 elements.')
bullet('Float32 activation memory = 1,605,000 x 4 = 6,420,000 bytes ~ 6.42 MB.')

a_rows = [
    ('32-bit (baseline)', '6,420,000', '6,420,000', '1.00x'),
    ('8-bit',             '6,420,000', '1,605,424', '3.999x'),
    ('4-bit',             '6,420,000', '1,605,424', '3.999x'),
    ('2-bit',             '6,420,000', '1,605,424', '3.999x'),
]
simple_table(
    ['Bit-width', 'Orig (bytes)', 'Quant (bytes)', 'Compression'],
    a_rows,
    [3.5, 3.8, 3.8, 5.2]
)
body('Best activation compression ratio: 4.00x (8-bit). Activations go from 6.42 MB to 1.61 MB.')

heading2('Q4(c) - Final Approximated Model Size After Compression')
body('Total model size = weight size + activation size + storage overhead:')
kv_line('Original model size',   '8.95 MB (weights) + 6.42 MB (activations) = 15.37 MB')
kv_line('8-bit quantized size',  '2.24 MB (weights) + 1.61 MB (activations) + 0.001 MB (overhead) = 3.84 MB')
kv_line('16-bit quantized size', '4.47 MB (weights) + 1.61 MB (activations) + 0.001 MB = 6.08 MB')
kv_line('Overall reduction',     '15.37 MB -> 3.84 MB  (4.00x compression, saving 11.53 MB)')
body(
    'The final approximated in-memory size of the compressed MobileNet-v2 (8-bit PTQ) is '
    '3.84 MB - a 75% reduction from the original 15.37 MB, with negligible accuracy '
    'degradation at 8 bits (see Q3 table).'
)

doc.add_page_break()

# ════════════════════════════════════════════════════════════════════════════
# Q5: Reproducibility
# ════════════════════════════════════════════════════════════════════════════
heading1('Question 5 - Reproducibility and Repository')

heading2('Q5(a) - Codebase Structure')
body('The code is split into separate files, each with a single responsibility:')
files = [
    ('model.py',    'MobileNet-v2 architecture adapted for CIFAR-10 (32x32 input)'),
    ('dataset.py',  'CIFAR-10 data loading, augmentation, and 45k/5k/10k splits'),
    ('engine.py',   'Training loop, evaluation loop, and optimizer factory'),
    ('train.py',    'Training script with W&B logging and checkpoint saving'),
    ('quantize.py', 'Custom PTQ: quantize_tensor, quantize_weights, ActivationQuantizer, compute_model_sizes'),
    ('test.py',     'Compression evaluation with detailed size/ratio reporting'),
    ('sweep.py',    'W&B sweep over 24 bit-width configurations'),
]
simple_table(
    ['File', 'Responsibility'],
    files,
    [3.5, 13.0]
)

heading2('Q5(b) - Exact Reproduction Commands')
code_block([
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
])
doc.add_paragraph()
body('Seed configuration (applied in train.py before any data loading):')
code_block([
    '  import random, numpy as np, torch',
    '  random.seed(42)',
    '  np.random.seed(42)',
    '  torch.manual_seed(42)',
    '  torch.cuda.manual_seed_all(42)',
    '  torch.backends.cudnn.deterministic = True',
    '  torch.backends.cudnn.benchmark     = False',
])
doc.add_paragraph()
body('Environment:')
for item in ['Python 3.10+', 'torch 2.0.0+   (CPU or CUDA)', 'torchvision 0.15.0+',
             'wandb 0.15.0+', 'numpy < 2.0', 'Pillow 9.0.0+']:
    bullet(item)

heading2('Q5(c) - GitHub Repository Link')
p_link = doc.add_paragraph()
p_link.alignment = WD_ALIGN_PARAGRAPH.CENTER
r_link = p_link.add_run('https://github.com/Ashishkumar-V/mobilenet-cifar10')
r_link.bold = True
r_link.font.size = Pt(12)
body(
    'The repository contains all Python source files, the README with exact commands and '
    'environment specifications, and the best model checkpoint '
    '(best_mobilenet_bs128_optsgd_lr0.01.pth). The W&B training run and compression sweep '
    'results are logged to the institutional forge.coreweave.com project '
    'mobilenet-cifar10 and mobilenet-cifar10-compression respectively.'
)

# ════════════════════════════════════════════════════════════════════════════
# References
# ════════════════════════════════════════════════════════════════════════════
heading1('References')
refs = [
    '[1] PyTorch MobileNet-v2 reference implementation:\n'
    '    https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py',
    '[2] Template quantization repository for this assignment (VGG16 baseline, adapted to MobileNet-v2):\n'
    '    https://github.com/DextroLaev/CS6886-Sys_for_dl-Assignment3',
    '[3] Sandler et al., "MobileNetV2: Inverted Residuals and Linear Bottlenecks", CVPR 2018.',
]
for ref in refs:
    body(ref)

# ════════════════════════════════════════════════════════════════════════════
# Save
# ════════════════════════════════════════════════════════════════════════════
out_path = './report_assets/CS25M505_A2_Report.docx'
doc.save(out_path)
print(f'Word document saved: {out_path}')
