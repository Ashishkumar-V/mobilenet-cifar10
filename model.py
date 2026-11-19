"""
model.py — MobileNet-v2 adapted for CIFAR-10 (32×32 input).

Key modification vs. ImageNet version:
  - Initial conv uses stride=1 (not stride=2) to preserve spatial resolution
  - Second block stride changed from 2→1 (32×32 is too small for aggressive downsampling)
"""

import torch
import torch.nn as nn


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU6(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride,
                      padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True),
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        hidden = int(round(inp * expand_ratio))
        self.use_skip = (stride == 1 and inp == oup)

        layers = []
        if expand_ratio != 1:                                    # pointwise expand
            layers.append(ConvBNReLU6(inp, hidden, kernel_size=1))
        layers += [
            ConvBNReLU6(hidden, hidden, stride=stride, groups=hidden),  # depthwise
            nn.Conv2d(hidden, oup, 1, 1, 0, bias=False),                # pointwise project
            nn.BatchNorm2d(oup),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.conv(x) if self.use_skip else self.conv(x)


class MobileNetV2(nn.Module):
    """
    MobileNet-v2 for CIFAR-10.

    Args:
        num_classes: number of output classes (default 10)
        width_mult:  channel width multiplier (default 1.0)
        dropout:     dropout probability before classifier (default 0.2)
    """

    # (expand_ratio, out_channels, num_blocks, stride)
    # strides adjusted for 32×32 input: first two stages use stride=1
    _SETTINGS = [
        (1,  16, 1, 1),
        (6,  24, 2, 1),   # stride 1 (original=2) — keep 32×32 spatial
        (6,  32, 3, 2),   # 32→16
        (6,  64, 4, 2),   # 16→8
        (6,  96, 3, 1),
        (6, 160, 3, 2),   # 8→4
        (6, 320, 1, 1),
    ]

    def __init__(self, num_classes=10, width_mult=1.0, dropout=0.2):
        super().__init__()
        in_ch = _make_divisible(32 * width_mult, 8)
        last_ch = _make_divisible(1280 * max(1.0, width_mult), 8)

        # Initial conv — stride=1 for CIFAR-10
        features = [ConvBNReLU6(3, in_ch, stride=1)]

        for t, c, n, s in self._SETTINGS:
            out_ch = _make_divisible(c * width_mult, 8)
            for i in range(n):
                features.append(InvertedResidual(in_ch, out_ch, s if i == 0 else 1, t))
                in_ch = out_ch

        features.append(ConvBNReLU6(in_ch, last_ch, kernel_size=1))
        self.features = nn.Sequential(*features)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(last_ch, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return self.classifier(x)


def mobilenet_v2(num_classes=10, width_mult=1.0, dropout=0.2):
    return MobileNetV2(num_classes=num_classes, width_mult=width_mult, dropout=dropout)


if __name__ == "__main__":
    model = mobilenet_v2()
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    total = sum(p.numel() for p in model.parameters())
    print(f"Output: {out.shape}  |  Params: {total:,}  |  Size: {total*4/1e6:.2f} MB")
