"""
Training script for FIS-enhanced model
File: train_fis.py
Compatible with Deep-JSCC-PyTorch (2019)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
from datetime import datetime

from model import JSCC_FIS
from channel import Channel
from dataset import Vanilla
from torch.utils.data import DataLoader
from torchvision import transforms
from utils import get_psnr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================
# Train one epoch
# =========================
def train_one_epoch(model, train_loader, channel, optimizer, criterion, epoch, args):
    model.train()

    for batch_idx, (images, _) in enumerate(train_loader):
        images = images.to(device)

        optimizer.zero_grad()

        # Forward (encoder + FIS)
        encoded, _, info = model(
            images,
            snr=args.snr,
            target_rate=args.target_rate,
            return_info=True
        )

        # Channel
        encoded_noisy = channel(encoded)

        # Decode
        decoded_noisy = model.decoder(encoded_noisy)

        # Loss
        loss = criterion(decoded_noisy, images)
        loss.backward()
        optimizer.step()

        # PSNR (utils.py gốc dùng max_val = 255)
        psnr = get_psnr(
            image=decoded_noisy * 255.0,
            gt=images * 255.0
        ).item()

        if batch_idx % args.log_interval == 0:
            print(
                f"Epoch [{epoch}] "
                f"Batch [{batch_idx}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f} "
                f"PSNR: {psnr:.2f} dB "
                f"AvgBits: {info['avg_bits']:.2f}"
            )


# =========================
# Validation
# =========================
def validate(model, val_loader, channel, criterion, epoch, args):
    model.eval()

    total_psnr = 0.0
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for images, _ in val_loader:
            images = images.to(device)


            encoded, _, info = model(
                images,
                snr=args.snr,
                target_rate=args.target_rate,
                return_info=True
            )

            encoded_noisy = channel(encoded)
            decoded_noisy = model.decoder(encoded_noisy)

            loss = criterion(decoded_noisy, images)

            psnr = get_psnr(
                image=decoded_noisy * 255.0,
                gt=images * 255.0
            ).item()

            total_loss += loss.item()
            total_psnr += psnr
            count += 1

    avg_loss = total_loss / count
    avg_psnr = total_psnr / count

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
    parser = argparse.ArgumentParser(description='Train FIS-Enhanced Deep JSCC')

    # Model
    parser.add_argument('--C', type=int, default=16)
    parser.add_argument('--channel_num', type=int, default=16)

    # Training
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    # Channel
    parser.add_argument('--snr', type=float, default=10.0)
    parser.add_argument('--channel', type=str, default='AWGN', choices=['AWGN', 'Rayleigh'])

    # FIS
    parser.add_argument('--target_rate', type=float, default=0.5)

    # Logging
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--save_dir', type=str, default='./out/checkpoint')
    parser.add_argument('--log_dir', type=str, default='./out/logs')

    args = parser.parse_args()

    # Output dirs
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = f'FIS_SNR{args.snr}_{timestamp}'
    save_path = os.path.join(args.save_dir, exp_name)
    log_path = os.path.join(args.log_dir, exp_name)

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    writer = SummaryWriter(log_path)

    # Model
    model = JSCC_FIS(C=args.C, channel_num=args.channel_num).to(device)
    print(f"Model created: {exp_name}")

    # Dataset (Vanilla ImageNet)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    train_dataset = Vanilla(
        root='./images',
        transform=transform
    )

    val_dataset = Vanilla(
        root='./images',
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Loss & Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Channel
    channel = Channel(
        channel_type=args.channel,
        snr=args.snr
    ).to(device)

    # Training loop
    best_psnr = 0.0

    for epoch in range(1, args.epochs + 1):
        print(f"\n===== Epoch {epoch}/{args.epochs} =====")

        train_one_epoch(
            model,
            train_loader,
            channel,
            optimizer,
            criterion,
            epoch,
            args
        )

        val_loss, val_psnr = validate(
            model,
            val_loader,
            channel,
            criterion,
            epoch,
            args
        )

        writer.add_scalar('Val/Loss', val_loss, epoch)
        writer.add_scalar('Val/PSNR', val_psnr, epoch)

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(
                model.state_dict(),
                os.path.join(save_path, 'best.pth')
            )
            print(f"Best model saved (PSNR = {best_psnr:.2f} dB)")

    writer.close()
    print(f"\nTraining finished. Best PSNR: {best_psnr:.2f} dB")


if __name__ == '__main__':
    main()
