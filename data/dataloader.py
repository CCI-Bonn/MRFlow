from torch.utils.data import Dataset
import numpy as np
import random
import torch
import os
from PIL import Image
import torchvision.transforms as T


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


class BraTS_Dataset(Dataset):
    def __init__(self, root, img_size=(512, 512)):
        self.root = root
        self.img_paths = []
        for dir_path, dirnames, filenames in os.walk(root):
            for file in filenames:
                self.img_paths.append(os.path.join(dir_path, file))
        
        self.img_size = img_size

    
    def get_random_modalities(self):
        # Randomly shuffle
        shuffled = random.sample(MODALITIES, k=len(MODALITIES))

        # Randomly pick output size (1 modality)
        output_modality = shuffled.pop()

        # Remaining becomes input (random size from 1 to len-1)
        input_size = random.randint(1, len(shuffled))
        input_modalities = random.sample(shuffled, k=input_size)

        #print(f"Input Modalities: {input_modalities}")
        #print(f"Output Modality: {output}")

        return input_modalities, output_modality
    
    
    def preprocess_image(self, image):
        # Convert to PIL and then RGB (3 channels)
        image = Image.fromarray(image).convert("RGB")
        
        # --- Step 3: Preprocess image for VAE ---
        transform = T.Compose([
            T.Resize(self.img_size),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),  # Normalize to [-1, 1]
        ])
        input_tensor = transform(image)  # [3, H, W]
        
        return input_tensor
    
    
    def __len__(self): 
        return len(self.img_paths) 
    
    
    def __getitem__(self, idx):
        
        input_modalities, output_modality = self.get_random_modalities()
        
        #print(f"Input Modalities: {input_modalities}")
        #print(f"Output Modality: {output_modality}")
        
        data = np.load(self.img_paths[idx])#.astype(np.uint8)
        
        imgs = [data['t1'].astype(np.uint8), data['t1ce'].astype(np.uint8), data['t2'].astype(np.uint8), data['flair'].astype(np.uint8)]
        means = [data['t1_latent_mean'], data['t1ce_latent_mean'], data['t2_latent_mean'], data['flair_latent_mean']]
        stds = [data['t1_latent_std'], data['t1ce_latent_std'], data['t2_latent_std'], data['flair_latent_std']]
        
        one_hot_input_mod = torch.tensor([0, 0, 0, 0], dtype=torch.float32)
        one_hot_target_mod = torch.tensor([0, 0, 0, 0], dtype=torch.float32)
        
        all_zeros = np.zeros_like(means[0])
        
        all_means = []
        all_stds = []
        
        target_mean, target_std, target_img = None, None, None
        
        for mod in MODALITIES:
            index = MODALITY_MAPPING[mod]            
            if mod in input_modalities:
                all_means.append(means[index])
                all_stds.append(stds[index])
                one_hot_input_mod[index] = 1
            else:
                all_means.append(all_zeros)
                all_stds.append(all_zeros)
            
            if mod == output_modality:
                target_mean, target_std, target_img = means[index], stds[index], imgs[index]
                one_hot_target_mod[index] = 1
        
        target_img = self.preprocess_image(target_img.squeeze())
        
        # Concatenate along the first dimension
        all_means = np.concatenate(all_means, axis=0)
        all_stds = np.concatenate(all_stds, axis=0)
        
        return target_img, target_mean, target_std, one_hot_input_mod, one_hot_target_mod, torch.tensor(all_means).float(), torch.tensor(all_stds).float()

'''
root = "/home/sysgen/Desktop/Priyank/Datasets/brats24/BraTS-GLI/numpy_files_training/"
dataset = BraTS_Dataset(root)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=3, shuffle=True)
it = iter(dataloader)
batch = next(it)
target_img, target_mean, target_std, one_hot_input_mod, one_hot_target_mod, latent_mean, latent_std = batch

'''