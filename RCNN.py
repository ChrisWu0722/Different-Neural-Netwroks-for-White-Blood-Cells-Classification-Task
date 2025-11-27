#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RCNN-style classifier (ImageFolder) but trained/logged in the same style as mask_rcnn_resnet101_wbc.py
so that we can compare curves (loss, box-head-top1, fg-top1) across RCNN and Mask R-CNN.

Key ideas:
- keep classification-only backbone (ResNet-101)
- keep ImageFolder dataset structure (train/val)
- but use the SAME optimizer/scheduler/logging/curves names as mask_rcnn_resnet101_wbc.py
- "box_head_top1" == normal classification top-1 here
- "fg_top1"       == normal classification top-1 here
"""

import os
import sys
import time
import json
import csv
import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

cudnn.benchmark = True
plt.ion()

# =========================================================
# basic utils (matching mask_rcnn_resnet101_wbc.py style)
# =========================================================
def set_torch_threads(n=4):
    os.environ.setdefault("OMP_NUM_THREADS", str(n))
    os.environ.setdefault("MKL_NUM_THREADS", str(n))
    try:
        torch.set_float32_matmul_precision("medium")
    except Exception:
        pass


def get_transforms(train: bool):
    # 和你原来RCNN的transform保持大体一致，但去掉太激进的增强
    if train:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])


def build_dataloaders(data_dir: str, batch_size: int = 16, num_workers: int = 2):
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), get_transforms(train=(x == "train")))
        for x in ["train", "val"]
    }
    dataloaders = {
        x: DataLoader(
            image_datasets[x],
            batch_size=batch_size,
            shuffle=True if x == "train" else False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        for x in ["train", "val"]
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}
    class_names = image_datasets["train"].classes
    return dataloaders, dataset_sizes, class_names


def get_model(num_classes: int):
    # 和你原来的一样，用 resnet101 + 替换 fc
    model = models.resnet101(weights="IMAGENET1K_V1")
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, num_classes)
    return model


def get_optimizer_and_scheduler(model, lr=0.002):
    # 跟 Mask R-CNN 保持同一风格：SGD + MultiStep
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[15, 30, 45],
        gamma=0.5
    )
    return optimizer, scheduler


@torch.inference_mode()
def evaluate_top1(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        outputs = model(imgs)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / max(1, total)


def train_one_epoch(model, criterion, optimizer, train_loader, device, scaler=None, accum_steps=1):
    model.train()
    running_loss = 0.0
    seen = 0

    pbar = tqdm(train_loader, desc="Train", file=sys.stdout, dynamic_ncols=True, ascii=True)
    optimizer.zero_grad(set_to_none=True)

    for step, (imgs, labels) in enumerate(pbar, 1):
        imgs = imgs.to(device)
        labels = labels.to(device)

        if scaler is not None and device.type == "cuda":
            with torch.cuda.amp.autocast():
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            loss_scaled = loss / accum_steps
            scaler.scale(loss_scaled).backward()
            if step % accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss = loss / accum_steps
            loss.backward()
            if step % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item() * imgs.size(0) * accum_steps  # 把除掉的accum再乘回来
        seen += imgs.size(0)
        pbar.set_postfix(loss=f"{running_loss / max(1, seen):.4f}")

    epoch_loss = running_loss / max(1, seen)
    return epoch_loss


def plot_and_save_curves(hist, logs_dir: Path):
    # 1) loss
    plt.figure()
    plt.plot(hist["train_loss"], label="train")
    plt.plot(hist["val_loss"], label="val")
    plt.title("Total Loss")
    plt.legend()
    plt.savefig(logs_dir / "loss_curve.png", dpi=200, bbox_inches="tight")
    plt.close()

    # 2) box-head top1 (对RCNN来说就是普通top1)
    plt.figure()
    plt.plot(hist["train_top1"], label="train top-1")
    plt.plot(hist["val_top1"], label="val top-1")
    plt.title("Box-Head Top-1 Accuracy")
    plt.legend()
    plt.savefig(logs_dir / "box_head_top1_curve.png", dpi=200, bbox_inches="tight")
    plt.close()

    # 3) fg top1 (对RCNN来说也是普通top1)
    plt.figure()
    plt.plot(hist["train_fg_top1"], label="train FG top-1")
    plt.plot(hist["val_fg_top1"], label="val FG top-1")
    plt.title("Foreground-Detected Top-1 Accuracy")
    plt.legend()
    plt.savefig(logs_dir / "fg_top1_curve.png", dpi=200, bbox_inches="tight")
    plt.close()


def main():
    set_torch_threads(4)

    # ===== paths =====
    data_dir = r"D:/桌面文件/Thesis/Thesis B/R-CNN/NewWBCDataset"

    # ===== datasets & loaders =====
    dataloaders, dataset_sizes, class_names = build_dataloaders(
        data_dir, batch_size=16, num_workers=2
    )

    num_classes = len(class_names)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Classes:", class_names, "->", num_classes)

    # ===== model / opt / sch =====
    model = get_model(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer, scheduler = get_optimizer_and_scheduler(model, lr=0.002)

    # amp scaler
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # ===== logging dir =====
    timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    logs_dir = Path("logs") / f"{timestamp}_rcnn_maskstyle"
    logs_dir.mkdir(parents=True, exist_ok=True)
    save_path = logs_dir / "rcnn_best.pth"
    print(f"[Logger] Current training log dir: {logs_dir}")

    # save run config
    run_cfg = {
        "timestamp": timestamp,
        "data_dir": data_dir,
        "num_classes": num_classes,
        "optimizer": "SGD",
        "lr": 0.002,
        "scheduler": "MultiStepLR(15,30,45)",
        "device": str(device),
        "class_names": class_names,
    }
    with open(logs_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_cfg, f, indent=2, ensure_ascii=False)

    # ===== history =====
    hist = {
        "train_loss": [],
        "val_loss": [],
        "train_top1": [],
        "val_top1": [],
        "train_fg_top1": [],
        "val_fg_top1": [],
    }

    # CSV header
    csv_path = logs_dir / "history.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss",
            "val_loss",
            "train_top1",
            "val_top1",
            "train_fg_top1",
            "val_fg_top1",
        ])

    best_val = float("inf")
    epochs = 20
    accum_steps = 1  # 分类任务一般不用accum，你想对齐也可以设成4

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        print("-" * 12)

        # ===== train =====
        train_loss = train_one_epoch(
            model, criterion, optimizer,
            dataloaders["train"], device,
            scaler=scaler, accum_steps=accum_steps
        )

        # ===== val loss =====
        model.eval()
        running_val_loss = 0.0
        with torch.inference_mode():
            for imgs, labels in dataloaders["val"]:
                imgs = imgs.to(device)
                labels = labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * imgs.size(0)
        val_loss = running_val_loss / max(1, dataset_sizes["val"])

        # ===== acc / fg_top1 =====
        train_top1 = evaluate_top1(model, dataloaders["train"], device)
        val_top1 = evaluate_top1(model, dataloaders["val"], device)

        # 在RCNN里，fg_top1 = 普通top1
        train_fg_top1 = train_top1
        val_fg_top1 = val_top1

        # ===== scheduler step =====
        scheduler.step()

        # ===== record =====
        hist["train_loss"].append(train_loss)
        hist["val_loss"].append(val_loss)
        hist["train_top1"].append(train_top1)
        hist["val_top1"].append(val_top1)
        hist["train_fg_top1"].append(train_fg_top1)
        hist["val_fg_top1"].append(val_fg_top1)

        print(f"train_loss:    {train_loss:.4f}")
        print(f"val_loss:      {val_loss:.4f}")
        print(f"train_top1:    {train_top1:.4f}")
        print(f"val_top1:      {val_top1:.4f}")
        print(f"train_fg_top1: {train_fg_top1:.4f}")
        print(f"val_fg_top1:   {val_fg_top1:.4f}")

        # write csv
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                train_loss,
                val_loss,
                train_top1,
                val_top1,
                train_fg_top1,
                val_fg_top1,
            ])

        # save best
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"✓ Saved best model to {save_path}")

        torch.cuda.empty_cache()
        print()

    # ===== plot curves =====
    plot_and_save_curves(hist, logs_dir)
    plt.ioff()
    print("Training finished.")


if __name__ == "__main__":
    main()
