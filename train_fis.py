"""
Training script for FIS-enhanced DeepJSCC
Compatible with Deep-JSCC-PyTorch (2019)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
from datetime import datetime

from model import DeepJSCC_FIS
from utils import get_psnr

# =========================
# Device
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================
# Train one epoch
# =========================
def train_one_epoch(model, loader, optimizer, criterion, epoch, args):
    model.train()
    total_loss = 0.0
    total_psnr = 0.0

    for batch_idx, (images, _) in enumerate(loader):
        images = images.to(device)

        optimizer.zero_grad()

        x_hat, info = model(
            images,
            fis_snr=args.snr,
            target_rate=args.target_rate,
            return_info=True
        )

        loss = criterion(x_hat, images)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        psnr = get_psnr(
            image=x_hat * 255.0,
            gt=images * 255.0
        ).item()
        total_psnr += psnr

        if batch_idx % args.log_interval == 0:
            print(
                f"Epoch [{epoch}] "
                f"Batch [{batch_idx}/{len(loader)}] "
                f"Loss: {loss.item():.4f} "
                f"PSNR: {psnr:.2f} dB "
                f"AvgBits: {info['avg_bits']:.2f}"
            )

    return total_loss / len(loader), total_psnr / len(loader)

# =========================
# Validation
# =========================
def validate(model, loader, criterion, epoch, args):
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)

            x_hat = model(
                images,
                fis_snr=args.snr,
                target_rate=args.target_rate
            )

            loss = criterion(x_hat, images)
            psnr = get_psnr(
                image=x_hat * 255.0,
                gt=images * 255.0
            ).item()

            total_loss += loss.item()
            total_psnr += psnr

    avg_loss = total_loss / len(loader)
    avg_psnr = total_psnr / len(loader)

    print(
        f"[Validation] Epoch [{epoch}] "
        f"Loss: {avg_loss:.4f} "
        f"PSNR: {avg_psnr:.2f} dB"
    )

    return avg_loss, avg_psnr

# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--snr', type=float, default=19.0)
    parser.add_argument('--target_rate', type=float, default=0.5)
    parser.add_argument('--log_interval', type=int, default=10)
    args = parser.parse_args()

    # =========================
    # Directories
    # =========================
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = f'FIS_SNR{args.snr}_{timestamp}'
    save_dir = f'./out/checkpoint/{exp_name}'
    log_dir = f'./out/logs/{exp_name}'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    writer = SummaryWriter(log_dir)

    # =========================
    # Model
    # =========================
    model = DeepJSCC_FIS(
        c=16,
        channel_type='AWGN',
        snr=args.snr
    ).to(device)

    # =========================
    # Dataset (CIFAR-10 – SAME AS BASELINE)
    # =========================
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_set = CIFAR10(
        root='./dataset',
        train=True,
        download=True,
        transform=transform
    )

    val_set = CIFAR10(
        root='./dataset',
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # =========================
    # Loss / Optimizer / Scheduler
    # =========================
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Scheduler giống baseline nhưng scale cho 50 epochs
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=640, gamma=0.1
    )

    # =========================
    # Training loop
    # =========================
    best_psnr = 0.0

    for epoch in range(1, args.epochs + 1):
        print(f"\n===== Epoch {epoch}/{args.epochs} =====")

        train_loss, train_psnr = train_one_epoch(
            model, train_loader, optimizer, criterion, epoch, args
        )

        val_loss, val_psnr = validate(
            model, val_loader, criterion, epoch, args
        )

        scheduler.step()

        writer.add_scalar('Train/Loss', train_loss, epoch)
        writer.add_scalar('Train/PSNR', train_psnr, epoch)
        writer.add_scalar('Val/Loss', val_loss, epoch)
        writer.add_scalar('Val/PSNR', val_psnr, epoch)

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(model.state_dict(), f"{save_dir}/best.pth")
            print(f"✓ Saved best model (PSNR={best_psnr:.2f})")

    writer.close()
    print("Training finished.")

if __name__ == '__main__':
    main()
