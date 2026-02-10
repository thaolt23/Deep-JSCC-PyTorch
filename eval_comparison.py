"""
Evaluation script to compare Baseline DeepJSCC vs FIS-DeepJSCC
WITH explicit channel simulation (AWGN / Rayleigh)

This version is adapted for Deep-JSCC-PyTorch-main
and follows the evaluation logic required by the instructor.
"""

import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import DeepJSCC, DeepJSCC_FIS



# -------------------------------------------------
# PSNR
# -------------------------------------------------
def calculate_psnr(img1, img2, max_val=1.0):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr.item()


# -------------------------------------------------
# Evaluation with explicit channel
# -------------------------------------------------
@torch.no_grad()
def evaluate(model, dataloader, snr_list, device,
             is_fis=False, target_rate=0.3):

    model.eval()
    psnr_all = []

    for snr in snr_list:
        print(f"\nEvaluating at SNR = {snr} dB")

        psnr_list = []

        for images, _ in dataloader:
            images = images.to(device)

            if is_fis:
                # FIS: set SNR giống train_fis.py
                model.snr = snr
                outputs = model(images)[1]
            else:
                # Baseline: channel nằm trong forward
                outputs = model(images)[1]

            psnr = calculate_psnr(images, outputs)
            psnr_list.append(psnr)

        avg_psnr = float(np.mean(psnr_list))
        psnr_all.append(avg_psnr)
        print(f"PSNR = {avg_psnr:.2f} dB")

    return psnr_all


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--baseline_checkpoint', type=str, required=True)
    parser.add_argument('--fis_checkpoint', type=str, required=True)
    parser.add_argument('--snr_list', nargs='+', type=float,
                        default=[1, 4, 7, 10, 13])
    parser.add_argument('--channel', type=str, default='AWGN')
    parser.add_argument('--target_rate', type=float, default=0.3)
    parser.add_argument('--save_plot', type=str, default='comparison.png')

    args = parser.parse_args()

    # -------------------------------------------------
    # Device
    # -------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -------------------------------------------------
    # Dataset (CIFAR-10 test)
    # -------------------------------------------------
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    test_dataset = datasets.CIFAR10(
        root='./dataset',
        train=False,
        download=True,
        transform=transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2
    )

    # -------------------------------------------------
    # IMPORTANT: match training architecture
    # -------------------------------------------------
    C_BASELINE = 8
    C_FIS = 16

    # -------------------------------------------------
    # Load models
    # -------------------------------------------------
    print("\nLoading baseline model...")
    baseline = DeepJSCC(C_BASELINE).to(device)
    baseline.load_state_dict(
        torch.load(args.baseline_checkpoint, map_location=device)
    )
    baseline.eval()

    print("Loading FIS model...")
    fis = DeepJSCC_FIS(C_FIS).to(device)
    fis.load_state_dict(
        torch.load(args.fis_checkpoint, map_location=device)
    )
    fis.eval()

    # -------------------------------------------------
    # Evaluation
    # -------------------------------------------------
    print("\n=== Evaluating Baseline ===")
    baseline_psnr = evaluate(
        baseline,
        test_loader,
        args.snr_list,
        device,
        is_fis=False
    )

    print("\n=== Evaluating FIS ===")
    fis_psnr = evaluate(
        fis,
        test_loader,
        args.snr_list,
        device,
        is_fis=True,
        target_rate=args.target_rate
    )

    # -------------------------------------------------
    # Plot
    # -------------------------------------------------
    plt.figure(figsize=(7, 5))
    plt.plot(args.snr_list, baseline_psnr, 'o-', linewidth=2,
             label='Baseline')
    plt.plot(args.snr_list, fis_psnr, 's-', linewidth=2,
             label='FIS')

    plt.xlabel('SNR (dB)')
    plt.ylabel('PSNR (dB)')
    plt.title('PSNR vs SNR')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.save_plot, dpi=300)

    print(f"\nSaved comparison plot to {args.save_plot}")


if __name__ == '__main__':
    main()
