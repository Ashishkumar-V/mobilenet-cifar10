"""
engine.py — Training and evaluation utilities.
"""

import torch
import torch.nn as nn


def train_one_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = total_correct = total = 0
    criterion = nn.CrossEntropyLoss()

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss    += loss.item() * images.size(0)
        total_correct += (outputs.argmax(1) == labels).sum().item()
        total         += images.size(0)

    if scheduler is not None:
        scheduler.step()

    return total_loss / total, 100.0 * total_correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = total_correct = total = 0
    criterion = nn.CrossEntropyLoss()

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss    += loss.item() * images.size(0)
        total_correct += (outputs.argmax(1) == labels).sum().item()
        total         += images.size(0)

    return total_loss / total, 100.0 * total_correct / total


def get_optimizer(model, optimizer_name, lr, momentum=0.9, weight_decay=1e-4):
    name = optimizer_name.lower()
    if name == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=lr,
                               momentum=momentum, weight_decay=weight_decay)
    if name == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {optimizer_name}")
