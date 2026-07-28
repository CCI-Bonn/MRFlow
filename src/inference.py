"""
Inference Script for Multi-Modal MRI Synthesis

This script runs inference and evaluation on test datasets using trained DiT models.
It supports comprehensive metrics including PSNR, SSIM, MS-SSIM, LPIPS, HFEN, and FID.

Usage:
    python inference.py --config config/config.yaml --dataset brats --checkpoint path/to/checkpoint.pth

The script can:
- Test on multiple datasets (BraTS, IXI, UCSF, UPENN)
- Evaluate different modality combinations (1-to-1, 2-to-1, 3-to-1)
- Save generated images and detailed metrics
- Generate visualization plots
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import os
import argparse
import numpy as np
from math import log10
from torchmetrics.image.ssim import (
    StructuralSimilarityIndexMeasure,
    MultiScaleStructuralSimilarityIndexMeasure
)
import torch.nn.functional as F
import matplotlib.pyplot as plt
import lpips
from torchmetrics.image.fid import FrechetInceptionDistance

from dataloader import MyDataset
from medvae import MVAE
from utils import load_config, get_dataset_paths, create_model


# Initialize MedVAE for latent encoding/decoding
MEDVAE = MVAE(model_name="medvae_4_3_2d", modality="mri")
MEDVAE.eval().cuda()
MEDVAE.requires_grad_(False)


# Modality mappings for label display
MODALITY_MAPPING_BRATS = {"t1": 0, "t1ce": 1, "t2": 2, "flair": 3}
MODALITY_MAPPING_IXI = {'pd': 0, 't1': 1, 't2': 2}


# =============================================================================
# Visualization
# =============================================================================

def plot_reconstructions(
    target_img: torch.Tensor,
    pred_img: torch.Tensor,
    all_imgs: torch.Tensor,
    one_hot_input_mod: torch.Tensor,
    one_hot_target_mod: torch.Tensor,
    epoch: int,
    modality: str,
    save_dir: str = "./val_outputs",
    dataset_name: str = 'brats'
):
    """
    Plot and save reconstruction comparison.
    
    Shows input modalities, ground truth target, and model prediction
    for visual quality assessment.
    
    Args:
        target_img: Ground truth target images [B, C, H, W]
        pred_img: Predicted images [B, C, H, W]
        all_imgs: Input modality images [B, num_inputs, H, W]
        one_hot_input_mod: One-hot encoding of inputs [B, num_modalities]
        one_hot_target_mod: One-hot encoding of target [B, num_modalities]
        epoch: Batch index for filename
        modality: Target modality name
        save_dir: Directory to save plots
        dataset_name: Dataset name for labeling
    """
    os.makedirs(save_dir, exist_ok=True)

    # Define modality labels
    if dataset_name in ['brats', 'UCSF', 'UPENN']:
        labels = ['t1', 't1ce', 't2', 'flair']
    elif dataset_name == 'ixi':
        labels = ['pd', 't1', 't2']
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")

    # Move to CPU
    target_img = target_img.detach().cpu()
    pred_img = pred_img.detach().cpu()
    all_imgs = all_imgs.detach().cpu()
    one_hot_input_mod = one_hot_input_mod.detach().cpu()
    one_hot_target_mod = one_hot_target_mod.detach().cpu()

    # Select middle sample from batch
    bs = target_img.shape[0]
    index = bs // 2

    target_img = target_img[index]
    pred_img = pred_img[index]
    all_imgs = all_imgs[index]
    one_hot_input_mod = one_hot_input_mod[index]
    one_hot_target_mod = one_hot_target_mod[index]

    num_inputs = all_imgs.shape[0] if all_imgs.ndim == 3 else 1
    num_cols = num_inputs + 2

    # Create figure
    fig, axes = plt.subplots(nrows=1, ncols=num_cols, figsize=(3 * num_cols, 3))
    if num_cols == 1:
        axes = [axes]

    # Get input labels
    input_labels = [label for label, oh in zip(labels, one_hot_input_mod) if oh == 1]
    
    # Plot input modalities
    for j in range(num_inputs):
        img_label = input_labels[j] if j < len(input_labels) else f"Input {j}"
        axes[j].imshow(all_imgs[j], cmap='gray', vmin=-1, vmax=1)
        axes[j].axis('off')
        axes[j].set_title(f"Input {img_label}")

    # Plot target
    axes[num_inputs].imshow(target_img[0], cmap='gray')
    axes[num_inputs].axis('off')
    axes[num_inputs].set_title(f"Target {labels[one_hot_target_mod.argmax()]}")

    # Plot prediction
    axes[num_inputs + 1].imshow(pred_img[0], cmap='gray')
    axes[num_inputs + 1].axis('off')
    axes[num_inputs + 1].set_title(f"Prediction {labels[one_hot_target_mod.argmax()]}")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'inference_reconstructions_{epoch}_pred_{modality}.png'))
    plt.close(fig)


# =============================================================================
# ODE Solvers
# =============================================================================

@torch.no_grad()
def euler_ode(
    init_input: torch.Tensor,
    y: torch.Tensor,
    one_hot_target_modality: torch.Tensor,
    input_imgs_mods: torch.Tensor,
    model: nn.Module,
    N: int = 100
) -> torch.Tensor:
    """
    Euler method ODE solver for inference.
    
    Args:
        init_input: Initial noise or starting point [B, C, H, W]
        y: Conditioning (input modalities) [B, num_slots*C, H, W]
        one_hot_target_modality: Target modality encoding [B, num_modalities]
        input_imgs_mods: Input modality indices [B, num_slots]
        model: DiT model
        N: Number of integration steps
        
    Returns:
        Generated latent [B, C, H, W]
    """
    eps, dt, T = 1e-3, 1./N, 1.0
    x = init_input.detach().clone()
    device = x.device

    for i in range(N):
        t = torch.ones(x.shape[0], device=device) * (i / N * (T - eps) + eps)
        with torch.amp.autocast(device_type='cuda'):
            pred = model(x, t*999, y, one_hot_target_modality, input_imgs_mods=input_imgs_mods)
        x = x + pred * dt
    return x


@torch.no_grad()
def heun_ode(
    init_input: torch.Tensor,
    y: torch.Tensor,
    one_hot_target_modality: torch.Tensor,
    input_imgs_mods: torch.Tensor,
    model: nn.Module,
    N: int = 100
) -> torch.Tensor:
    """
    Heun's method (2nd order Runge-Kutta) ODE solver.
    More accurate than Euler for the same number of steps.
    
    Args:
        init_input: Initial noise or starting point [B, C, H, W]
        y: Conditioning (input modalities) [B, num_slots*C, H, W]
        one_hot_target_modality: Target modality encoding [B, num_modalities]
        input_imgs_mods: Input modality indices [B, num_slots]
        model: DiT model
        N: Number of integration steps
        
    Returns:
        Generated latent [B, C, H, W]
    """
    eps, dt, T = 1e-3, 1./N, 1.0
    x = init_input.detach().clone()
    device = x.device

    for i in range(N):
        t = torch.ones(x.shape[0], device=device) * (i / N * (T - eps) + eps)
        t_next = torch.ones(x.shape[0], device=device) * ((i+1) / N * (T - eps) + eps)
        
        with torch.amp.autocast(device_type='cuda'):
            k1 = model(x, t * 999, y, one_hot_target_modality, input_imgs_mods=input_imgs_mods)
            k2 = model(x + dt * k1, t_next * 999, y, one_hot_target_modality, input_imgs_mods=input_imgs_mods)
        
        x = x + dt * 0.5 * (k1 + k2)
    return x


@torch.no_grad()
def rk4_ode(
    init_input: torch.Tensor,
    y: torch.Tensor,
    one_hot_target_modality: torch.Tensor,
    input_imgs_mods: torch.Tensor,
    model: nn.Module,
    N: int = 100
) -> torch.Tensor:
    """
    4th order Runge-Kutta ODE solver.
    Highest accuracy but requires 4 function evaluations per step.
    
    Args:
        init_input: Initial noise or starting point [B, C, H, W]
        y: Conditioning (input modalities) [B, num_slots*C, H, W]
        one_hot_target_modality: Target modality encoding [B, num_modalities]
        input_imgs_mods: Input modality indices [B, num_slots]
        model: DiT model
        N: Number of integration steps
        
    Returns:
        Generated latent [B, C, H, W]
    """
    eps, dt, T = 1e-3, 1./N, 1.0
    x = init_input.detach().clone()
    device = x.device

    for i in range(N):
        t = torch.ones(x.shape[0], device=device) * (i / N * (T - eps) + eps)
        t_next = torch.ones(x.shape[0], device=device) * ((i+1) / N * (T - eps) + eps)
        
        with torch.amp.autocast(device_type='cuda'):
            k1 = model(x, t * 999, y, one_hot_target_modality, input_imgs_mods=input_imgs_mods)
            k2 = model(x + 0.5*dt*k1, (t + 0.5*dt) * 999, y, one_hot_target_modality, input_imgs_mods=input_imgs_mods)
            k3 = model(x + 0.5*dt*k2, (t + 0.5*dt) * 999, y, one_hot_target_modality, input_imgs_mods=input_imgs_mods)
            k4 = model(x + dt*k3, t_next * 999, y, one_hot_target_modality, input_imgs_mods=input_imgs_mods)
        
        x = x + dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0
    return x


# =============================================================================
# Metrics
# =============================================================================

def calculate_psnr(target: torch.Tensor, pred: torch.Tensor) -> tuple:
    """
    Calculate Peak Signal-to-Noise Ratio.
    
    Args:
        target: Target images
        pred: Predicted images
        
    Returns:
        Tuple of (psnr_value, mse_value)
    """
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu()
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu()
    
    mse = F.mse_loss(pred, target)
    psnr = 20 * log10(1.0 / torch.sqrt(mse))
    return psnr, mse


def hfen(pred: torch.Tensor, gt: torch.Tensor, sigma: float = 1.5, kernel_size: int = 15) -> torch.Tensor:
    """
    Calculate High-Frequency Error Norm (HFEN).
    
    Measures errors in high-frequency components using Laplacian of Gaussian.
    
    Args:
        pred: Predicted images
        gt: Ground truth images
        sigma: Standard deviation for Gaussian
        kernel_size: Size of the LoG kernel
        
    Returns:
        HFEN value
    """
    # Ensure 4D input [B, C, H, W]
    if pred.ndim == 3:
        pred = pred.unsqueeze(0)
    elif pred.ndim == 2:
        pred = pred.unsqueeze(0).unsqueeze(0)
    if gt.ndim == 3:
        gt = gt.unsqueeze(0)
    elif gt.ndim == 2:
        gt = gt.unsqueeze(0).unsqueeze(0)

    assert pred.shape == gt.shape
    device = pred.device

    # Create Laplacian of Gaussian kernel
    ax = torch.arange(kernel_size, device=device) - kernel_size // 2
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    norm = (xx**2 + yy**2) / (2 * sigma**2)
    log_kernel = (norm - 1) * torch.exp(-norm)
    log_kernel = log_kernel - log_kernel.mean()
    log_kernel = log_kernel.view(1, 1, kernel_size, kernel_size)

    # Apply LoG filter
    pred_log = F.conv2d(pred, log_kernel, padding=kernel_size // 2)
    gt_log = F.conv2d(gt, log_kernel, padding=kernel_size // 2)
    
    return torch.norm(pred_log - gt_log)


# =============================================================================
# Latent Encoding Helper
# =============================================================================

@torch.no_grad()
def encode_batch(medvae_input_imgs: torch.Tensor, medvae_target_img: torch.Tensor,
                dataset_name: str) -> tuple:
    """
    Encode input and target images to latent space using MedVAE.
    
    Args:
        medvae_input_imgs: Input images [B, num_slots*3, H, W]
        medvae_target_img: Target images [B, 3, H, W]
        dataset_name: Dataset name to determine number of slots
        
    Returns:
        Tuple of (input_latents, target_latent)
    """
    num_slots = 3 if dataset_name in ['brats', 'UCSF', 'UPENN'] else 2
    slot_latents = []
    
    for i in range(num_slots):
        yi = medvae_input_imgs[:, i*3:(i+1)*3, :, :]
        _, post, _ = MEDVAE.model(yi, decode=False)
        slot_latents.append(post.mean)
    
    _, target_post, _ = MEDVAE.model(medvae_target_img, decode=False)
    
    return torch.cat(slot_latents, dim=1), target_post.mean


# =============================================================================
# Evaluation Loop
# =============================================================================

@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    modality: str,
    sigma: float,
    ode_solver,
    save_dir: str = "./inference_outputs",
    plot_after: int = 1000,
    target_modality: str = None,
    save_generated_images: bool = False,
    n_steps: int = 20,
    dataset_name: str = 'brats'
):
    """
    Run comprehensive evaluation on test dataset.
    
    Generates images using the trained model and computes multiple quality metrics.
    Can optionally save generated images and create visualization plots.
    
    Args:
        model: Trained DiT model
        dataloader: Test data loader
        device: Device to run on
        modality: Target modality name
        sigma: Noise scaling factor (0=deterministic, 1=standard Gaussian)
        ode_solver: ODE solver function to use
        save_dir: Directory for outputs
        plot_after: Plot visualizations every N batches
        target_modality: Target modality for organizing saved images
        save_generated_images: Whether to save generated .npy files
        n_steps: Number of ODE integration steps
        dataset_name: Dataset name
    """
    model.eval()

    # Initialize metric accumulators
    avg_psnr, avg_ssim, avg_ms_ssim = [], [], []
    avg_lpips, avg_mse, avg_l1, avg_hfen = [], [], [], []
    patient_names = []
    fid_real_images, fid_generated_images = [], []

    # Initialize metric calculators
    lpips_loss = lpips.LPIPS(net='vgg').to(device)
    ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
    fid_metric = FrechetInceptionDistance(feature=2048).to(device)

    for i, batch in enumerate(tqdm(dataloader, desc="Evaluating")):

        # Unpack batch
        target_img_original, target_img, all_imgs, one_hot_input_mod, one_hot_target_mod, \
            input_imgs_mods, medvae_input_imgs, medvae_target_img, img_paths = [
            x.to(device) if torch.is_tensor(x) else x for x in batch
        ]

        patient_names += img_paths

        # Encode to latent space
        cond, target_latent_z = encode_batch(medvae_input_imgs, medvae_target_img, dataset_name)

        n, c, h, w = target_latent_z.shape
        original_shape = (target_img_original.shape[1], target_img_original.shape[2])

        # Initialize starting point based on sigma
        # sigma=0: deterministic (start from zeros)
        # sigma=1: stochastic (start from Gaussian noise)
        if sigma == 0:
            z0 = torch.zeros(n, c, h, w, device=device)
        else:
            z0 = sigma * torch.randn(n, c, h, w, device=device)

        # Generate latent using ODE solver
        pred_latent = ode_solver(z0, cond, one_hot_target_mod, input_imgs_mods, model, N=n_steps)

        # Decode from latent space
        pred = MEDVAE.model.decode(pred_latent)
        pred = pred[:, 0:1, :, :]  # Take first channel only

        target_img_original = target_img_original.unsqueeze(1)  # [B, 1, H, W]

        pred_batch = pred.clone().detach()
        target_batch = target_img_original.clone().detach()

        # Compute metrics for each sample in batch
        for b in range(n):
            img_path = img_paths[b]

            single_pred = pred_batch[b:b+1]
            single_target = target_batch[b:b+1]

            # Normalize prediction to [0, 1] and resize to original dimensions
            single_pred_01 = torch.clamp(single_pred, -1, 1)
            single_pred_01 = (single_pred_01 + 1) / 2.0
            single_pred_01 = F.interpolate(
                single_pred_01,
                size=original_shape,
                mode='bilinear',
                align_corners=False
            )

            # Compute LPIPS (requires GPU)
            lpips_score = lpips_loss(single_pred_01.to(device), single_target.to(device)).mean()
            avg_lpips.append(lpips_score.detach().cpu().item())

            # Move to CPU for other metrics
            single_pred_01 = single_pred_01.detach().cpu()
            single_target = single_target.detach().cpu()

            # Compute metrics
            psnr, mse = calculate_psnr(single_target, single_pred_01)
            avg_psnr.append(psnr)
            avg_mse.append(mse.detach().cpu().item())
            avg_l1.append(F.l1_loss(single_pred_01, single_target).item())

            avg_ssim.append(ssim(single_pred_01, single_target))
            avg_ms_ssim.append(ms_ssim(single_pred_01, single_target))
            avg_hfen.append(hfen(single_pred_01, single_target).item())

            # Prepare for FID calculation (requires 3-channel RGB and uint8)
            fid_real_images.append((single_target * 255).repeat(1, 3, 1, 1).round().to(torch.uint8))
            fid_generated_images.append((single_pred_01 * 255).repeat(1, 3, 1, 1).round().to(torch.uint8))

            # Save generated image if requested
            if save_generated_images:
                _save_generated(single_pred, img_path, target_modality, save_dir, original_shape)

        # Create visualization plots periodically
        if i % plot_after == 0:
            pred_vis = (pred.clamp(-1, 1) + 1) / 2
            plot_reconstructions(
                target_img_original, pred_vis, all_imgs,
                one_hot_input_mod, one_hot_target_mod,
                i, modality, save_dir, dataset_name
            )

        # Update FID metric
        fid_metric.update(torch.cat(fid_real_images, dim=0).to(device), real=True)
        fid_metric.update(torch.cat(fid_generated_images, dim=0).to(device), real=False)
        fid_real_images, fid_generated_images = [], []

    # Compute final FID score
    fid_value = fid_metric.compute().item()

    # Save per-sample results
    result_filename = os.path.join(save_dir, 'result.txt')
    with open(result_filename, "w") as f:
        for name_, psnr_, ssim_, ms_ssim_, lpips_ in zip(
            patient_names, avg_psnr, avg_ssim, avg_ms_ssim, avg_lpips
        ):
            f.write(f"{name_},{psnr_},{ms_ssim_},{lpips_}\n")

    # Print summary statistics
    print(f'\n{"="*80}')
    print(f'Final Evaluation Results:')
    print(f'{"="*80}')
    print(f'  MSE:     {np.mean(avg_mse):.4f} ± {np.std(avg_mse):.4f}')
    print(f'  L1:      {np.mean(avg_l1):.4f} ± {np.std(avg_l1):.4f}')
    print(f'  PSNR:    {np.mean(avg_psnr):.4f} ± {np.std(avg_psnr):.4f}')
    print(f'  SSIM:    {np.mean(avg_ssim):.4f} ± {np.std(avg_ssim):.4f}')
    print(f'  MS-SSIM: {np.mean(avg_ms_ssim):.4f} ± {np.std(avg_ms_ssim):.4f}')
    print(f'  LPIPS:   {np.mean(avg_lpips):.4f} ± {np.std(avg_lpips):.4f}')
    print(f'  HFEN:    {np.mean(avg_hfen):.4f} ± {np.std(avg_hfen):.4f}')
    print(f'  FID:     {fid_value:.4f}')
    print(f'{"="*80}\n')


def _save_generated(
    single_pred: torch.Tensor,
    img_path: str,
    target_modality: str,
    save_dir: str,
    original_shape: tuple
):
    """
    Save generated image as numpy array.
    
    Organizes saved images by modality in separate directories.
    
    Args:
        single_pred: Generated image tensor [1, 1, H, W]
        img_path: Original image path for naming
        target_modality: Target modality name
        save_dir: Base save directory
        original_shape: Original image dimensions for resizing
    """
    # Mapping from modality name to directory and suffix
    modality_map = {
        't1': ('T1', '0000'),
        't1ce': ('T1CE', '0001'),
        't2': ('T2', '0002'),
        'flair': ('FLAIR', '0003'),
    }
    
    if target_modality not in modality_map:
        return

    final_dir, modality_id = modality_map[target_modality]
    final_dir = os.path.join(save_dir, "generated_images", final_dir)
    os.makedirs(final_dir, exist_ok=True)

    # Construct output filename
    filename_wo_ext = os.path.splitext(os.path.basename(img_path))[0]
    patient_id = os.path.basename(os.path.dirname(img_path))
    out_path = os.path.join(final_dir, f"{patient_id}_{filename_wo_ext}_{modality_id}.npy")

    # Resize to original dimensions and save
    pred_np = F.interpolate(
        single_pred,
        size=original_shape,
        mode='bilinear',
        align_corners=False
    )
    pred_np = pred_np.detach().cpu().squeeze(0).numpy()
    np.save(out_path, pred_np)


# =============================================================================
# Main Inference Function
# =============================================================================

def infer(
    modalities: tuple,
    sigma: float = 1.0,
    save_dir: str = "./inference_outputs",
    plot_after: int = 1000,
    target_modality: str = None,
    save_generated_images: bool = False,
    n_steps: int = 20,
    dataset_name: str = 'brats',
    checkpoint_path: str = None,
    config: dict = None
):
    """
    Run inference on test dataset.
    
    Args:
        modalities: Tuple of (input_modalities_list, output_modality_str)
        sigma: Noise scaling (0=deterministic, 1=stochastic)
        save_dir: Output directory
        plot_after: Visualization interval
        target_modality: Target modality name
        save_generated_images: Whether to save .npy files
        n_steps: Number of ODE steps
        dataset_name: Dataset name
        checkpoint_path: Path to model checkpoint
        config: Configuration dictionary
    """
    assert checkpoint_path is not None, "Please provide a checkpoint path."
    assert dataset_name in ['brats', 'ixi', 'UCSF', 'UPENN'], \
        f"Invalid dataset name: {dataset_name}"
    assert config is not None, "Please provide a config dict (needed to build the model)."

    # Initialize model (architecture, input_size, in_channels come from config['model'])
    model = create_model(config, dataset_name=dataset_name)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    model.load_state_dict(checkpoint["model"])
    model.eval().cuda()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    total_num_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_num_params:,}\n')

    input_modalities, output_modality = modalities
    print(f'Input: {input_modalities} → Output: {output_modality}')

    # Get test dataset path
    test_root = get_dataset_paths(config, dataset_name)['test']

    # Create test dataset
    test_dataset = MyDataset(
        root=test_root,
        img_size=tuple(config['dataset']['img_size']),
        testing=True,
        input_modalities=input_modalities,
        output_modality=output_modality,
        dataset_name=dataset_name
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config['inference']['batch_size'],
        shuffle=False,
        num_workers=config['inference']['num_workers'],
        pin_memory=True
    )

    print(f'Test samples: {len(test_dataset)}\n')

    # Select ODE solver
    solver_name = config['inference']['solver']
    ode_solver = {'euler': euler_ode, 'heun': heun_ode, 'rk4': rk4_ode}[solver_name]

    # Run evaluation
    evaluate(
        model, test_dataloader, device, output_modality,
        sigma=sigma,
        ode_solver=ode_solver,
        save_dir=save_dir,
        plot_after=plot_after,
        target_modality=target_modality,
        save_generated_images=save_generated_images,
        n_steps=n_steps,
        dataset_name=dataset_name
    )


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point for inference script."""
    parser = argparse.ArgumentParser(description='Run inference on MRI Synthesis Model')
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['brats', 'ixi', 'UCSF', 'UPENN'],
        help='Dataset to evaluate on'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--sigma',
        type=float,
        default=1.0,
        help='Noise scaling factor (0=deterministic, 1=stochastic)'
    )
    parser.add_argument(
        '--n_steps',
        type=int,
        default=10,
        help='Number of ODE integration steps'
    )
    parser.add_argument(
        '--save_images',
        action='store_true',
        help='Save generated images as .npy files'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set random seed for reproducibility
    seed = config['inference']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Get modality combinations to test
    modality_tests = config['inference']['modality_tests'][args.dataset]
    modalities_list = []
    for category in modality_tests.values():
        modalities_list.extend(category)
    
    print(f'\n{"="*120}')
    print(f'Starting Inference')
    print(f'Dataset: {args.dataset} | Seed: {seed} | Steps: {args.n_steps} | Sigma: {args.sigma}')
    print(f'Checkpoint: {args.checkpoint}')
    print(f'{"="*120}\n')
    
    # Run inference for each modality combination
    for modalities in modalities_list:
        input_mods, output_mod = modalities[0], modalities[1]
        input_str = '_'.join(input_mods)
        
        save_dir = os.path.join(
            config['paths']['inference_outputs'],
            f'{args.dataset}_{seed}',
            f'LatentSpace_{input_str}_to_{output_mod}_SIGMA_{args.sigma}'
        )
        os.makedirs(save_dir, exist_ok=True)
        
        print(f'\n{"-"*80}')
        print(f'Testing: {input_mods} → {output_mod}')
        print(f'{"-"*80}')
        
        infer(
            modalities=modalities,
            sigma=args.sigma,
            save_dir=save_dir,
            plot_after=40,
            target_modality=output_mod,
            save_generated_images=args.save_images,
            n_steps=args.n_steps,
            dataset_name=args.dataset,
            checkpoint_path=args.checkpoint,
            config=config
        )
    
    print("\n✅ Inference complete!")


if __name__ == "__main__":
    main()
