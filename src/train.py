"""
Training Script for MRI Synthesis with Rectified Flow

This script trains a Diffusion Transformer (DiT) model for multi-modal MRI
synthesis using Rectified Flow in latent space. The model learns to generate
missing MRI modalities from available ones.

Usage:
    python train.py --config config/config.yaml --dataset brats
"""

import os
import argparse
import random
from datetime import datetime

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
import lpips

from dataloader import MyDataset
from utils import create_model
from medvae import MVAE


# Initialize MedVAE model (frozen, used for latent encoding)
MEDVAE = MVAE(model_name="medvae_4_3_2d", modality="mri")
MEDVAE.eval().cuda()
MEDVAE.requires_grad_(False)


def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


#################################################################################
#                              Metrics                                          #
#################################################################################

# LPIPS loads a VGG backbone; instantiate once and reuse across validation batches
_LPIPS_MODEL = None


def _get_lpips_model(device):
    global _LPIPS_MODEL
    if _LPIPS_MODEL is None:
        _LPIPS_MODEL = lpips.LPIPS(net='vgg').to(device)
        _LPIPS_MODEL.eval()
    return _LPIPS_MODEL

def calculate_batch_psnr_torch(original, generated, max_val=1.0):
    """Calculate Peak Signal-to-Noise Ratio (PSNR) for a batch."""
    mse = F.mse_loss(original, generated, reduction='none')
    mse = mse.view(mse.shape[0], -1).mean(dim=1)
    psnr = 20 * torch.log10(torch.tensor(max_val)) - 10 * torch.log10(mse)
    return psnr.mean().item()


def calculate_batch_ssim_torch(original, generated, max_val=1.0):
    """Calculate Structural Similarity Index (SSIM) for a batch."""
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=max_val).to(original.device)
    return ssim_metric(original, generated).item()


def calculate_batch_l1_torch(original, generated):
    """Calculate L1 (Mean Absolute Error) for a batch."""
    return F.l1_loss(original, generated).item()


def calculate_batch_lpips(original, generated, device):
    """Calculate Learned Perceptual Image Patch Similarity (LPIPS) for a batch."""
    lpips_model = _get_lpips_model(device)
    with torch.no_grad():
        original = original.clamp(0, 1).float()
        generated = generated.clamp(0, 1).float()
        return lpips_model(original, generated).mean().item()


#################################################################################
#                              ODE Solvers                                      #
#################################################################################

@torch.no_grad()
def heun_ode(init_input, y, one_hot_target_modality, input_imgs_mods, model, N=100):
    """
    Heun's method for ODE integration (second-order, predictor-corrector).
    
    More accurate than Euler, commonly used for diffusion models.
    """
    eps, dt, T = 1e-3, 1./N, 1.0
    x = init_input.detach().clone()
    device = init_input.device

    for i in tqdm(range(N), desc="Heun ODE"):
        t      = torch.ones(x.shape[0], device=device) * (i / N * (T - eps) + eps)
        t_next = torch.ones(x.shape[0], device=device) * ((i+1) / N * (T - eps) + eps)
        k1 = model(x,          t      * 999, y, one_hot_target_modality, input_imgs_mods=input_imgs_mods)
        k2 = model(x + dt * k1, t_next * 999, y, one_hot_target_modality, input_imgs_mods=input_imgs_mods)
        x = x + dt * 0.5 * (k1 + k2)
    return x


#################################################################################
#                              Visualization                                    #
#################################################################################

def plot_reconstructions(target_img, pred_img, all_imgs, one_hot_input_mod, 
                        one_hot_target_mod, epoch, save_dir, dataset_name='brats'):
    """
    Plot and save reconstruction visualizations.
    
    Shows input modalities, ground truth target, and predicted target side-by-side.
    """
    labels = ['pd', 't1', 't2'] if dataset_name == 'ixi' else ['t1', 't1ce', 't2', 'flair']

    # Move to CPU and select up to 3 samples (fewer if the batch itself is smaller)
    n_plot = min(3, target_img.shape[0])
    target_img         = target_img.detach().cpu()[:n_plot]
    pred_img           = pred_img.detach().cpu()[:n_plot]
    all_imgs           = all_imgs.detach().cpu()[:n_plot]
    one_hot_input_mod  = one_hot_input_mod.detach().cpu()[:n_plot]
    one_hot_target_mod = one_hot_target_mod.detach().cpu()[:n_plot]

    num_inputs = all_imgs.shape[1] if all_imgs.ndim == 4 else 1
    num_cols   = num_inputs + 2  # inputs + target + prediction

    fig, axes = plt.subplots(nrows=n_plot, ncols=num_cols, figsize=(3 * num_cols, 3 * n_plot), squeeze=False)

    for i in range(n_plot):
        target_label = labels[one_hot_target_mod[i].argmax()]
        input_labels = [l for j, l in enumerate(labels) if j != one_hot_target_mod[i].argmax()]

        # Plot input modalities
        for j in range(num_inputs):
            axes[i, j].imshow(all_imgs[i, j], cmap='gray')
            axes[i, j].axis('off')
            axes[i, j].set_title(f"Input {input_labels[j]}")

        # Plot ground truth target
        axes[i, num_inputs].imshow(target_img[i, 0], cmap='gray')
        axes[i, num_inputs].axis('off')
        axes[i, num_inputs].set_title(f"Target {target_label}")

        # Plot prediction
        axes[i, num_inputs+1].imshow(pred_img[i, 0], cmap='gray')
        axes[i, num_inputs+1].axis('off')
        axes[i, num_inputs+1].set_title(f"Prediction {target_label}")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'reconstructions_{epoch}.png'))
    plt.close(fig)


#################################################################################
#                          Latent Encoding Helper                               #
#################################################################################

@torch.no_grad()
def encode_batch(medvae_input_imgs, medvae_target_img, dataset_name):
    """
    Encode images through frozen MedVAE to obtain latent representations.
    
    Args:
        medvae_input_imgs: Input modality images [B, num_slots*3, H, W]
        medvae_target_img: Target modality image [B, 3, H, W]
        dataset_name: Dataset name to determine number of slots
    
    Returns:
        tuple: (concatenated_input_latents, target_latent)
    """
    num_slots = 3 if dataset_name in ['brats', 'UCSF', 'UPENN'] else 2
    slot_latents = []
    
    # Encode each input modality slot
    for i in range(num_slots):
        yi = medvae_input_imgs[:, i*3:(i+1)*3, :, :]
        _, post, _ = MEDVAE.model(yi, decode=False)
        slot_latents.append(post.mean)
    
    # Encode target modality
    _, target_post, _ = MEDVAE.model(medvae_target_img, decode=False)
    
    # Concatenate all input latents
    all_input_latent_z = torch.cat(slot_latents, dim=1)
    return all_input_latent_z, target_post.mean


#################################################################################
#                              Training Loop                                    #
#################################################################################

def train_one_epoch(model, dataloader, optimizer, device, epoch, config, dataset_name='brats'):
    """
    Train for one epoch.
    
    Implements Rectified Flow training:
    1. Sample time t uniformly from [eps, T]
    2. Create linear interpolation: x_t = t*x_1 + (1-t)*x_0
    3. Predict velocity: v = x_1 - x_0
    4. Minimize MSE between predicted and true velocity
    """
    model.train()
    total_loss = 0.0
    mse_loss = nn.MSELoss()
    
    # Rectified Flow parameters
    eps = config['training']['eps']
    T = config['training']['T']
    step = 0

    pbar = tqdm(dataloader, desc=f"Train Epoch {epoch+1}")

    for batch in pbar:
        step += 1

        # Unpack batch
        _, target_img, all_imgs, one_hot_input_mod, one_hot_target_mod, \
            input_imgs_mods, medvae_input_imgs, medvae_target_img, _ = [
            x.to(device) if torch.is_tensor(x) else x for x in batch
        ]

        # Encode through MedVAE
        all_input_latent_z, target_latent_z = encode_batch(
            medvae_input_imgs, medvae_target_img, dataset_name
        )

        n, c, h, w = target_latent_z.shape
        
        # Sample noise (x_0 in Rectified Flow)
        z0 = torch.randn_like(target_latent_z)

        # Sample time uniformly
        t = torch.rand(n, device=device) * (T - eps) + eps

        # Create linear interpolation: x_t = t*x_1 + (1-t)*x_0
        t_expand = t.view(-1, 1, 1, 1).expand(n, c, h, w)
        perturbed_data = t_expand * target_latent_z + (1. - t_expand) * z0
        
        # Target velocity: v = x_1 - x_0
        target = target_latent_z - z0

        # Predict velocity
        pred_velocity = model(
            x=perturbed_data,
            t=t * config['training']['t_scale'],
            y=all_input_latent_z,
            one_hot_target_modality=one_hot_target_mod,
            input_imgs_mods=input_imgs_mods
        )

        # Compute loss
        loss = mse_loss(pred_velocity, target)

        # Optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

        # Periodic logging
        if step % config['training']['print_interval'] == 0 or step == 1:
            print(f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, '
                  f'Avg Loss: {total_loss / step:.4f}, Step: {step}')

    pbar.close()
    return total_loss / step


#################################################################################
#                             Validation Loop                                   #
#################################################################################

@torch.no_grad()
def validate(model, dataloader, device, epoch, config, save_dir, dataset_name='brats'):
    """
    Validate the model.
    
    Computes both:
    1. Training-style loss (velocity prediction MSE)
    2. Generation quality metrics (PSNR, SSIM, L1, LPIPS) on one batch
    """
    model.eval()
    total_loss = 0.0
    mse_loss = nn.MSELoss()

    eps = config['training']['eps']
    T = config['training']['T']
    step = 0

    # Select ODE solver
    ode_solver = heun_ode

    for i, batch in enumerate(tqdm(dataloader, desc="Validating")):
        step += 1

        # Unpack batch
        _, target_img, all_imgs, one_hot_input_mod, one_hot_target_mod, \
            input_imgs_mods, medvae_input_imgs, medvae_target_img, _ = [
            x.to(device) if torch.is_tensor(x) else x for x in batch
        ]

        # Encode through MedVAE
        all_input_latent_z, target_latent_z = encode_batch(
            medvae_input_imgs, medvae_target_img, dataset_name
        )

        n, c, h, w = target_latent_z.shape
        z0 = torch.randn_like(target_latent_z)

        # Sample time uniformly
        t = torch.rand(n, device=device) * (T - eps) + eps

        # Create interpolation and compute loss
        t_expand = t.view(-1, 1, 1, 1).expand(n, c, h, w)
        perturbed_data = t_expand * target_latent_z + (1. - t_expand) * z0
        target = target_latent_z - z0

        with torch.amp.autocast(device_type='cuda'):
            pred_velocity = model(
                x=perturbed_data,
                t=t * config['training']['t_scale'],
                y=all_input_latent_z,
                one_hot_target_modality=one_hot_target_mod,
                input_imgs_mods=input_imgs_mods
            )
            loss = mse_loss(pred_velocity, target)

        total_loss += loss.item()

        # Generate samples and compute metrics on one batch
        if i == 10:
            # Generate from noise using ODE solver
            pred_latent_z = ode_solver(
                init_input=z0, 
                y=all_input_latent_z,
                one_hot_target_modality=one_hot_target_mod,
                input_imgs_mods=input_imgs_mods,
                model=model, 
                N=20
            )

            # Decode to image space
            pred_image = MEDVAE.model.decode(pred_latent_z).clamp(-1, 1)

            # Normalize to [0, 1] for metrics
            target_img_01 = (target_img + 1) / 2
            pred_image_01 = (pred_image + 1) / 2
            pred_image_01 = pred_image_01[:, 0:1, :, :]  # First channel only

            # Plot reconstructions
            plot_reconstructions(
                target_img_01, pred_image_01, all_imgs, 
                one_hot_input_mod, one_hot_target_mod, 
                epoch, save_dir, dataset_name
            )

            # Calculate metrics
            batch_psnr  = calculate_batch_psnr_torch(target_img_01, pred_image_01)
            batch_ssim  = calculate_batch_ssim_torch(target_img_01, pred_image_01)
            batch_l1    = calculate_batch_l1_torch(target_img_01.to(device), pred_image_01.to(device))
            batch_lpips = calculate_batch_lpips(target_img_01.to(device), pred_image_01.to(device), device)

            print(f'\n{"*"*120}')
            print(f"Validation Metrics - PSNR: {batch_psnr:.4f}, SSIM: {batch_ssim:.4f}, "
                  f"L1: {batch_l1:.4f}, LPIPS: {batch_lpips:.4f}")
            print(f'{"*"*120}\n')

    return total_loss / step


#################################################################################
#                               Checkpoint Saving                               #
#################################################################################

def save_model(model, curr_loss, save_path, epoch, optimizer, best_loss, save_every=5):
    """
    Save model checkpoints.
    
    Saves:
    1. Regular checkpoints every 'save_every' epochs
    2. Best model when validation loss improves
    """
    # Regular checkpoint saving
    if epoch % save_every == 0:
        checkpoint = {
            "model": model.state_dict(),
            "epoch": epoch,
            "optimizer": optimizer.state_dict(),
            "best_loss": best_loss,
        }
        torch.save(checkpoint, os.path.join(save_path, f'epoch_{epoch}_rectifiedFlow.pth'))

    # Best model saving
    if curr_loss < best_loss:
        print(f"New best loss: {best_loss:.4f} -> {curr_loss:.4f}")
        best_loss = curr_loss
        checkpoint = {
            "model": model.state_dict(),
            "epoch": epoch,
            "optimizer": optimizer.state_dict(),
            "best_loss": best_loss,
        }
        torch.save(checkpoint, os.path.join(save_path, 'best.pth'))

    return best_loss


#################################################################################
#                               Main Training                                   #
#################################################################################

def train(config):
    """Main training function."""
    
    # Extract configuration
    dataset_name = config['dataset']['name']
    epochs = config['training']['epochs']
    lr = config['training']['learning_rate']
    batch_size = config['training']['batch_size']
    num_workers = config['training']['num_workers']
    data_parallelism = config['training']['data_parallel']
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model
    model = create_model(config)
    if data_parallelism:
        model = nn.DataParallel(model)
    model.to(device)

    total_num_params = sum(p.numel() for p in model.parameters())

    # Setup optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, 
        milestones=config['training']['scheduler']['milestones'], 
        gamma=config['training']['scheduler']['gamma']
    )

    start_epoch = 0
    best_loss = float('inf')

    # Resume from checkpoint if specified
    if config['training']['continue_training']:
        checkpoint = torch.load(config['training']['checkpoint_path'], map_location='cuda')
        model.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        print(f'\n\nCONTINUING TRAINING from epoch {start_epoch}\n\n')

        if config['training']['reset_epoch']:
            start_epoch = 0
            best_loss = float('inf')

    # Setup datasets
    train_root = config['dataset']['paths'][dataset_name]['train']
    val_root = config['dataset']['paths'][dataset_name]['val']
    
    train_dataset = MyDataset(
        root=train_root, 
        img_size=tuple(config['dataset']['img_size']), 
        dataset_name=dataset_name
    )
    val_dataset = MyDataset(
        root=val_root, 
        img_size=tuple(config['dataset']['img_size']), 
        dataset_name=dataset_name
    )

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    )

    # Create output directories (one subdirectory per dataset, so the pretrained
    # checkpoints/brats.pth and checkpoints/ixi.pth are never overwritten)
    save_dir = os.path.join(config['paths']['validation_outputs'], dataset_name)
    model_save_path = os.path.join(config['paths']['checkpoints'], dataset_name)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(model_save_path, exist_ok=True)

    # Print training info
    print(f'\nDataset: {dataset_name}')
    print(f'Epochs: {epochs} | Batch size: {batch_size} | LR: {lr} | Device: {device}')
    print(f'Train size: {len(train_dataset)} | Val size: {len(val_dataset)}')
    print(f'Train batches: {len(train_dataloader)} | Val batches: {len(val_dataloader)}')
    print(f'Total parameters: {total_num_params:,}\n')
    print('=' * 120)

    # Avoid crash if validation not run on first epoch
    val_loss = float('inf')

    # Training loop
    for epoch in range(start_epoch, epochs):
        print(f'\nTime: {datetime.now().strftime("%Y%m%d-%H%M")} => Epoch {epoch+1}/{epochs}')

        # Train
        train_loss = train_one_epoch(
            model, train_dataloader, optimizer, device, epoch, config, dataset_name
        )
        train_dataset.set_epoch(epoch + 1)
        print(f'Train loss: {train_loss:.4f}')

        # Validate
        if epoch % config['training']['validate_interval'] == 0:
            val_loss = validate(
                model, val_dataloader, device, epoch, config, save_dir, dataset_name
            )
            val_dataset.set_epoch(epoch + 1)

        print(f'Val loss: {val_loss:.4f}')

        # Save checkpoints
        best_loss = save_model(
            model, val_loss, model_save_path, epoch, optimizer, best_loss,
            save_every=config['training']['save_interval']
        )

        scheduler.step()
        print('=' * 120)

    print("Training complete!")


#################################################################################
#                                   Main                                        #
#################################################################################

def main():
    parser = argparse.ArgumentParser(description='Train MRI Synthesis Model')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--dataset', type=str, default=None,
                       help='Dataset name (overrides config)')
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Override dataset if specified
    if args.dataset:
        config['dataset']['name'] = args.dataset

    # Set random seed
    set_seed(config['training']['seed'])

    # Start training
    train(config)


if __name__ == "__main__":
    main()
