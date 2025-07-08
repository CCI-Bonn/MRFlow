import os
import numpy as np
import nibabel as nib
from tqdm import tqdm
import random
import shutil
from pathlib import Path
from diffusers import AutoencoderKL
import torchvision.transforms as T
from PIL import Image
import torch

VAE = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2", subfolder="vae")
VAE.eval().cuda()  # Use CUDA if available

# Map modality suffix to identifier
MODALITY_SUFFIXES = {
    "t1": "-t1n.nii.gz",
    "t1ce": "-t1c.nii.gz",
    "t2": "-t2w.nii.gz",
    "flair": "-t2f.nii.gz"
}


def preprocess_image_for_vae(image, target_size=(512, 512)):
    # Convert to PIL and then RGB (3 channels)
    image = Image.fromarray(image).convert("RGB")
    
    # --- Step 3: Preprocess image for VAE ---
    transform = T.Compose([
        T.Resize(target_size),
        T.ToTensor(),
        T.Normalize([0.5], [0.5]),  # Normalize to [-1, 1]
    ])
    input_tensor = transform(image).unsqueeze(0).cuda()  # [1, 3, H, W]
    
    return input_tensor


def preprocess_mri_slice(img_3d):
    """Clip MRI intensities and normalize to [0, 255]"""
    img = np.copy(img_3d)
    img = np.where(np.isnan(img), 0, img)

    mask = img > 0
    if np.sum(mask) == 0:
        return np.zeros_like(img, dtype=np.uint8)

    lower = np.percentile(img[mask], 0.5)
    upper = np.percentile(img[mask], 99.5)
    img = np.clip(img, lower, upper)

    try:
        img = ((img - img.min()) / (img.max() - img.min())) * 255.0
    except:
        img = np.zeros_like(img)

    return img.astype(np.uint8)


def convert_brats_to_slices(original_root, new_root, min_pixels=2000):
    """
    Converts BraTS volumes to 2D axial slices, filters based on brain content,
    and saves valid 4-channel slices as .npy files.
    """
    
    check = 0
    
    with tqdm(desc="Processing", unit="files") as pbar:
        for subject_dir, _, files in os.walk(original_root):
            files = [f for f in files if f.endswith(".nii.gz")]
            if not files:
                continue

            subject_id = os.path.basename(subject_dir)

            try:
                # Load and normalize each modality
                vol_channels = []
                for modality, suffix in MODALITY_SUFFIXES.items():
                    matched = [f for f in files if f.endswith(suffix)]
                    if not matched:
                        raise FileNotFoundError(f"Missing modality {modality} for {subject_id}")
                    path = os.path.join(subject_dir, matched[0])
                    img = nib.load(path).get_fdata()
                    img = preprocess_mri_slice(img)
                    vol_channels.append(img)
            except Exception as e:
                print(f"[ERROR] Skipping {subject_id}: {e}")
                continue

            vol = np.stack(vol_channels, axis=0)  # shape: (4, 240, 240, 155)
            
            for z in range(vol.shape[-1]):
                
                # save a dict
                d = {}
                
                # Ensure each modality has at least min_pixels
                if not all(np.count_nonzero(vol[m, :, :, z]) >= min_pixels for m in range(4)):
                    continue

                slice_4ch = vol[:, :, :, z]  # shape: (4, 240, 240)
                
                d['t1'] = slice_4ch[0],
                d['t1ce'] = slice_4ch[1],
                d['t2'] = slice_4ch[2],
                d['flair'] = slice_4ch[3]
                
                input_t1_tensor = preprocess_image_for_vae(image=slice_4ch[0], target_size=(256, 256))
                input_t1ce_tensor = preprocess_image_for_vae(image=slice_4ch[1], target_size=(256, 256))
                input_t2_tensor = preprocess_image_for_vae(image=slice_4ch[2], target_size=(256, 256))
                input_flair_tensor = preprocess_image_for_vae(image=slice_4ch[3], target_size=(256, 256))
                
                input_concat = torch.cat([input_t1_tensor, input_t1ce_tensor, input_t2_tensor, input_flair_tensor], dim=0)
                
                with torch.no_grad():
                    #latents_t1 = VAE.encode(input_t1_tensor).latent_dist
                    #latents_t1ce = VAE.encode(input_t1ce_tensor).latent_dist
                    #latents_t2 = VAE.encode(input_t2_tensor).latent_dist
                    #latents_flair = VAE.encode(input_flair_tensor).latent_dist
                    latents = VAE.encode(input_concat).latent_dist 
                
                '''
                latents_t1_mean = latents_t1.mean.squeeze().cpu().numpy()
                latents_t1_std = latents_t1.std.squeeze().cpu().numpy()
                
                latents_t1ce_mean = latents_t1ce.mean.squeeze().cpu().numpy()
                latents_t1ce_std = latents_t1ce.std.squeeze().cpu().numpy()
                
                latents_t2_mean = latents_t2.mean.squeeze().cpu().numpy()
                latents_t2_std = latents_t2.std.squeeze().cpu().numpy()
                
                latents_flair_mean = latents_flair.mean.squeeze().cpu().numpy()
                latents_flair_std = latents_flair.std.squeeze().cpu().numpy()
                '''
                
                latents_t1_mean = latents.mean[0,:,:,:].cpu().numpy()
                latents_t1_std = latents.std[0,:,:,:].cpu().numpy()
                
                latents_t1ce_mean = latents.mean[1,:,:,:].cpu().numpy()
                latents_t1ce_std = latents.std[1,:,:,:].cpu().numpy()
                
                latents_t2_mean = latents.mean[2,:,:,:].cpu().numpy()
                latents_t2_std = latents.std[2,:,:,:].cpu().numpy()
                
                latents_flair_mean = latents.mean[3,:,:,:].cpu().numpy()
                latents_flair_std = latents.std[3,:,:,:].cpu().numpy()
                
                if check == 0:
                    print(f'Shapes:\n MEAN:: latents_t1_mean={latents_t1_mean.shape} latents_t1ce_mean={latents_t1ce_mean.shape}\n STD:: latents_t2_std={latents_t2_std.shape} latents_flair_std={latents_flair_std.shape}')
                    check = 1
                
                d['t1_latent_mean'] = latents_t1_mean
                d['t1_latent_std'] = latents_t1_std
                
                d['t1ce_latent_mean'] = latents_t1ce_mean
                d['t1ce_latent_std'] = latents_t1ce_std
                
                d['t2_latent_mean'] = latents_t2_mean
                d['t2_latent_std'] = latents_t2_std
                
                d['flair_latent_mean'] = latents_flair_mean
                d['flair_latent_std'] = latents_flair_std
                
                # Save slice
                out_dir = os.path.join(new_root, subject_id)
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, f"{z:03d}")
                np.savez(out_path + '.npz', **d)

                print(f"Saved: {out_path}")

            pbar.update(1)  # Update progress bar for each subject processed


def transfer_to_val(training_root, val_root, split=0.7):
    # Create val folder if it doesn't exist
    os.makedirs(training_root, exist_ok=True)

    # Get list of subdirectories (folders only)
    all_folders = [f for f in os.listdir(training_root) if os.path.isdir(os.path.join(training_root, f))]

    # Shuffle for randomness
    random.shuffle(all_folders)

    # 30% to validation
    num_val = int((1 - split) * len(all_folders))
    val_folders = all_folders[:num_val]

    # Move selected folders
    for folder in val_folders:
        src = os.path.join(training_root, folder)
        dst = os.path.join(val_root, folder)
        shutil.move(src, dst)

    print(f"Moved {len(val_folders)} folders to {val_root}")
            

# Example usage
if __name__ == "__main__":
    
    # Uncomment the following lines to run the conversion
    # Makes numpy files from BraTS volumes
    '''
    for modality, suffix in MODALITY_SUFFIXES.items():
        print(modality)
    original_root = "/home/sysgen/Desktop/Priyank/Datasets/brats24/BraTS-GLI/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/"
    new_root = "/home/sysgen/Desktop/Priyank/Datasets/brats24/BraTS-GLI/numpy_files_training/"
    convert_brats_to_slices(original_root, new_root, min_pixels=2000)
    '''
    
    # Uncomment the following lines to transfer 30% of training data to validation
    # This will move 30% of the folders from training to validation
    #'''
    training_root = "/home/sysgen/Desktop/Priyank/Datasets/brats24/BraTS-GLI/numpy_files_training/"
    val_root = "/home/sysgen/Desktop/Priyank/Datasets/brats24/BraTS-GLI/numpy_files_val/"
    transfer_to_val(training_root, val_root, split=0.7)
    #'''
    
    