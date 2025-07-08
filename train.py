import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
import random
import os
from diffusers import AutoencoderKL
from dataloader import BraTS_Dataset
from model import LatentFusionModel, LatentFusionTransformerModel, LatentFusionTransformerModel_Cross
from datetime import datetime
import numpy as np
from math import log10
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
import torchvision.transforms as T
from PIL import Image
from torch.optim.lr_scheduler import MultiStepLR
#import lpips

VAE = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2", subfolder="vae")
VAE.eval().cuda()  # Use CUDA if available


MODALITY_SUFFIXES = {
    "t1": "-t1n.nii.gz",
    "t1ce": "-t1c.nii.gz",
    "t2": "-t2w.nii.gz",
    "flair": "-t2f.nii.gz"
}

MODALITY_MAPPING = {
    "t1": 0,
    "t1ce": 1,
    "t2": 2,
    "flair": 3
}

MODALITIES = list(MODALITY_SUFFIXES.keys())

# Configuration
LATENT_DIM = 256
SAVE_DIR = "./val_outputs"
os.makedirs(SAVE_DIR, exist_ok=True)


def calculate_batch_psnr(original, generated, max_val=255.0):
    """
    original: numpy array of shape [B, H, W] or [B, C, H, W]
    generated: same shape as original
    Returns: list of PSNRs for each image in the batch
    """
    original = np.array(original).astype(np.float64)
    generated = np.array(generated).astype(np.float64)

    B = original.shape[0]
    psnrs = []

    for i in range(B):
        ori = original[i]
        gen = generated[i]

        mse = np.mean((ori - gen) ** 2)
        if mse == 0:
            psnrs.append(float('inf'))
        else:
            psnrs.append(10 * log10((max_val ** 2) / mse))

    return np.mean(psnrs)


def kl_divergence_between_gaussians(mu_pred, std_pred, mu_gt, std_gt, eps=1e-8, scale=100):
    """
    Compute KL divergence between two Gaussians: 
    N(mu_pred, std_pred^2) and N(mu_gt, std_gt^2)

    Args:
        mu_pred, std_pred: predicted mean and std, shape [B, C, H, W]
        mu_gt, std_gt: ground truth mean and std, shape [B, C, H, W]
    
    Returns:
        Scalar tensor: average KL divergence over batch
    """
    mu_pred = mu_pred / scale
    mu_gt = mu_gt / scale
    
    # Scale stds and clamp to avoid division by zero or instability
    std_pred = torch.clamp(std_pred / scale, min=1e-2)
    std_gt = torch.clamp(std_gt / scale, min=1e-2)
            
    var_pred = std_pred ** 2
    var_gt = std_gt ** 2

    term1 = torch.log((var_gt + eps) / (var_pred + eps))
    term2 = (var_pred + (mu_pred - mu_gt) ** 2) / (var_gt + eps)
    kl = 0.5 * (term1 + term2 - 1)

    return kl.mean()  # Or .sum() if you want total KL


# ----- TRAINING LOOP -----
def train_one_epoch(model, dataloader, optimizer, device, epoch, print_after=50):
    model.train()
    total_loss = 0.0
    pbar = tqdm(dataloader, desc=f"Train Epoch {epoch+1}")
    
    mse_loss = nn.MSELoss()
    #perceptual_loss = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization
    kl_loss = kl_divergence_between_gaussians
    
    step = 0
    
    for batch in pbar:
        
        step += 1
        
        target_img, target_mean, target_std, one_hot_input_mod, one_hot_target_mod, latent_mean, latent_std = [x.to(device) for x in batch]

        target_img = target_img.to(device)
        
        pred_latent_mean, pred_latent_std = model(latent_mean, latent_std, one_hot_input_mod, one_hot_target_mod)
        
        # Calculate loss
        #loss = 0.7 * criterion(pred_latent_mean, target_mean) + 0.3 * criterion(pred_latent_std, target_std)
        #loss = kl_divergence_between_gaussians(mu_pred=pred_latent_mean, std_pred=pred_latent_std, mu_gt=target_mean, std_gt=target_std, eps=1e-8)
        
        pred_latent_sample = torch.randn(pred_latent_mean.shape).to(device)
        pred_latent_sample = pred_latent_mean + pred_latent_std * pred_latent_sample
        with torch.no_grad():
                pred_image = VAE.decode(pred_latent_sample).sample
        
        #print(target_img.device)
        #print(pred_image.device)
        
        loss = mse_loss(pred_image, target_img) + kl_loss(mu_pred=pred_latent_mean, std_pred=pred_latent_std, mu_gt=target_mean, std_gt=target_std, eps=1e-8)
        #perceptual_loss(pred_image, target_img) + 
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item())
        
        if step % print_after == 0 or step == 1:
            print(f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, Current Loss: {total_loss / step}, iteration_num: {step}')
        
        
    return total_loss / step

# ----- VALIDATION LOOP -----
@torch.no_grad()
def validate(model, dataloader, device, print_after=50):
    model.eval()
    total_loss = 0.0

    step = 0
    
    mse_loss = nn.MSELoss()
    #perceptual_loss = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization
    kl_loss = kl_divergence_between_gaussians
    
    for i, batch in enumerate(tqdm(dataloader, desc="Validating")):
        
        step += 1
        
        target_img, target_mean, target_std, one_hot_input_mod, one_hot_target_mod, latent_mean, latent_std = [x.to(device) for x in batch]

        target_img = target_img.to(device)
        
        pred_latent_mean, pred_latent_std = model(latent_mean, latent_std, one_hot_input_mod, one_hot_target_mod)
        
        pred_latent_sample = torch.randn(pred_latent_mean.shape).to(device)
        pred_latent_sample = pred_latent_mean + pred_latent_std * pred_latent_sample
        with torch.no_grad():
                pred_image = VAE.decode(pred_latent_sample).sample
        
        loss = mse_loss(pred_image, target_img) + kl_loss(mu_pred=pred_latent_mean, std_pred=pred_latent_std, mu_gt=target_mean, std_gt=target_std, eps=1e-8)
        #+ perceptual_loss(pred_image, target_img) + 
        
        
        # Calculate loss
        #loss = 0.7 * criterion(pred_latent_mean, target_mean) + 0.3 * criterion(pred_latent_std, target_std)
        #loss = kl_divergence_between_gaussians(mu_pred=pred_latent_mean, std_pred=pred_latent_std, mu_gt=target_mean, std_gt=target_std, eps=1e-8)
        loss = mse_loss(pred_image, target_img) + kl_loss(mu_pred=pred_latent_mean, std_pred=pred_latent_std, mu_gt=target_mean, std_gt=target_std, eps=1e-8)
        
        total_loss += loss.item()

        if step < 5:
            pred_latent_sample = torch.randn(pred_latent_mean.shape).to(device)
            pred_latent_sample = pred_latent_mean + pred_latent_std * pred_latent_sample
            
            with torch.no_grad():
                pred_image = VAE.decode(pred_latent_sample).sample
            
            psnr = calculate_batch_psnr(original=target_img.cpu().numpy(), generated=pred_image.cpu().numpy(), max_val=2.0)
            print(f'PSNR: {psnr:.2f} dB')
            
            ssim = StructuralSimilarityIndexMeasure(data_range=2.0)
            ssim_score = ssim(pred_image.cpu(), target_img.cpu())  # supports batches!
            print(f'SSIM: {ssim_score:.4f}\n')
        
        '''
        if save_images and i == 0:  # Save only one batch of predictions
            recon = decoder(pred_latent[:4])
            target = decoder(target_latent[:4])
            combined = torch.cat([recon, target], dim=0)  # 4 predicted + 4 GT
            save_image(combined, f"{SAVE_DIR}/epoch_{epoch+1}_pred_vs_gt.png", nrow=4)
        '''
        
    avg_loss = total_loss / step
    return avg_loss


def save_model(model, curr_loss, path, epoch, optimizer, best_loss, save_after=2):
    
    # Save the model every `save_after` epochs
    if epoch % save_after == 0:
    
        # save the latest model
        model_weights = model.state_dict()
        checkpoint = {
            "model": model_weights,
            "epoch": epoch,
            "optimizer": optimizer.state_dict(),
            "best_loss": best_loss,
        }
        torch.save(checkpoint, os.path.join(path, f'epoch_{epoch}_medsam_lite_.pth'))
    
    ## save the best model
    model_weights = model.state_dict()
    if curr_loss < best_loss:
        print(f"New best loss: {best_loss:.4f} -> {curr_loss:.4f}")
        best_loss = curr_loss
        checkpoint = {
            "model": model_weights,
            "epoch": epoch,
            "optimizer": optimizer.state_dict(),
            "best_loss": best_loss,
        }
        torch.save(checkpoint, os.path.join(path, 'best.pth'))
    
    return best_loss


def train():
    
    epochs = 50
    lr = 1e-3
    batch_size = 64
    num_workers = 4
    data_parallelism=False  # Set to True if using multiple GPUs
    
    continue_training = False  # Allows training to continue
    continue_training_checkpoint = '/home/sysgen/Desktop/Priyank/MRI_Gen/checkpoints/best.pth'
    continue_training_from0epoch = False  # Allows model to load from pre-trained but training starts from 0 epoch and best_loss=inf
    
    model_save_path = "./checkpoints"
    os.makedirs(model_save_path, exist_ok=True)
    
    #model = LatentFusionModel()
    #model = LatentFusionTransformerModel(depth=5, embedding_dim=32, num_heads=4)
    model = LatentFusionTransformerModel_Cross(depth=3, embedding_dim=32, num_heads=4)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = nn.DataParallel(model) if data_parallelism else model
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Drop LR by 10x at epoch 20 and again at epoch 40
    scheduler = MultiStepLR(optimizer, milestones=[250], gamma=0.1)      # <-------------------------------------------------------------------------
    
    start_epoch = 0
    
    best_loss = float('inf')
    
    if continue_training:
        assert continue_training_checkpoint is not None
        checkpoint = torch.load(continue_training_checkpoint, map_location='cuda')
        model.load_state_dict(checkpoint['model'])
        #optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        
        print('\n\n****************************************************************************')
        print('CONTINUING TRAINNG')
        print(f'LOADED MODEL from {continue_training_checkpoint} ==> EPOCH:{start_epoch}')
        print('****************************************************************************\n\n')
        
        if continue_training_from0epoch:
            start_epoch = 0
            best_loss = float('inf')
        

    train_dataset = BraTS_Dataset(root="/home/sysgen/Desktop/Priyank/Datasets/brats24/BraTS-GLI/numpy_files_training/", img_size=(256, 256))
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True)
    
    val_dataset = BraTS_Dataset(root="/home/sysgen/Desktop/Priyank/Datasets/brats24/BraTS-GLI/numpy_files_val/", img_size=(256, 256))
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True)
    
    
    print(f'Epochs: {epochs}, Batch size: {batch_size}, Learning rate: {lr}, Device: {device}\n')
    print(f'Training dataset size: {len(train_dataset)}')
    print(f'Validation dataset size: {len(val_dataset)}\n')
    print(f'Training dataset: {train_dataset.root}')
    print(f'Validation dataset: {val_dataset.root}\n')
    print(f'Number of training batches: {len(train_dataloader)}')
    print(f'Number of validation batches: {len(val_dataloader)}\n\n')
    
    print(f'Starting training for {epochs} epochs...\n\n')
    print('===================================================================================================================================\n')
    
    
    for epoch in range(start_epoch, epochs):
        print(f'Time: {datetime.now().strftime("%Y%m%d-%H%M")} ==> Epoch {epoch+1}/{epochs}\n')
        
        # Train for one epoch
        train_loss = train_one_epoch(model, train_dataloader, optimizer, device, epoch)
        
        scheduler.step()
        
        # Validate the model
        val_loss = validate(model, val_dataloader, device)
        
        # Print losses
        print(f'\n\nTime: {datetime.now().strftime("%Y%m%d-%H%M")} ==> Train loss: {train_loss:.4f}')
        print(f'Time: {datetime.now().strftime("%Y%m%d-%H%M")} ==> Validation loss: {val_loss:.4f}\n')
        
        # Save the model
        best_loss = save_model(model, val_loss, model_save_path, epoch, optimizer, best_loss, save_after=5)
        
        print('===================================================================================================================================\n')
    
    print("Training complete!")
    

if __name__ == "__main__":
    train()
