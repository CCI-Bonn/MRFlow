"""
Dataset Module for Multi-Modal MRI Synthesis

This module provides a PyTorch Dataset class for loading and preprocessing
multi-modal MRI data. It supports flexible input/output modality combinations
and handles missing modalities by zero-padding.

Supported datasets:
- BraTS: 4 modalities (T1, T1CE, T2, FLAIR)
- IXI: 3 modalities (PD, T1, T2)
- UCSF: 4 modalities (T1, T1CE, T2, FLAIR)
- UPENN: 4 modalities (T1, T1CE, T2, FLAIR)
"""

from torch.utils.data import Dataset
import numpy as np
import random
import torch
import os
import torch.nn.functional as F


# Modality name to index mappings
MODALITY_MAPPING_BRATS = {
    "t1": 0,
    "t1ce": 1,
    "t2": 2,
    "flair": 3
}

MODALITY_MAPPING_IXI = {
    'pd': 0,
    't1': 1,
    't2': 2
}


class MyDataset(Dataset):
    """
    Multi-modal MRI Dataset.
    
    This dataset handles multi-modal MRI data stored as numpy arrays.
    During training, it randomly samples input/output modality combinations.
    During testing, it uses fixed modality combinations.
    
    Data Format:
        - Files should be .npy arrays with shape [num_modalities, H, W]
        - For BraTS/UCSF/UPENN: [4, H, W] with order [T1, T1CE, T2, FLAIR]
        - For IXI: [3, H, W] with order [PD, T1, T2]
    
    Attributes:
        root: Root directory containing the dataset
        img_size: Target size for resizing images (H, W)
        testing: If True, use fixed modality combinations
        input_modalities: List of input modality names (for testing)
        output_modality: Output modality name (for testing)
        dataset_name: Name of the dataset ('brats', 'ixi', 'UCSF', 'UPENN')
    """
    
    def __init__(
        self,
        root: str,
        img_size: tuple = (512, 512),
        testing: bool = False,
        input_modalities: list = None,
        output_modality: str = None,
        use_nifti: bool = False,
        dataset_name: str = "brats"
    ):
        """
        Initialize dataset.
        
        Args:
            root: Root directory containing .npy files
            img_size: Target size for image resizing (H, W)
            testing: Whether in testing mode (uses fixed modality combinations)
            input_modalities: List of input modality names (required for testing)
            output_modality: Target output modality name (required for testing)
            use_nifti: Whether to use NIfTI format (currently unused)
            dataset_name: Dataset name ('brats', 'ixi', 'UCSF', 'UPENN')
        """
        self.root = root
        self.use_nifti = use_nifti
        self.dataset_name = dataset_name
        
        # Collect all file paths
        self.img_paths = []
        for dir_path, dirnames, filenames in os.walk(root):
            for file in filenames:
                if file.endswith('.npy'):
                    self.img_paths.append(os.path.join(dir_path, file))
        
        self.testing = testing
        self.current_epoch = 0
        
        # Set modality mappings based on dataset
        if self.dataset_name in ["brats", "UCSF", "UPENN"]:
            self.MODALITY_MAPPING = MODALITY_MAPPING_BRATS
            self.MODALITIES = list(MODALITY_MAPPING_BRATS.keys())
        elif self.dataset_name == "ixi":
            self.MODALITY_MAPPING = MODALITY_MAPPING_IXI
            self.MODALITIES = list(MODALITY_MAPPING_IXI.keys())
        else:
            raise ValueError(
                f"Unsupported dataset name: {dataset_name}. "
                "Choose from 'brats', 'UCSF', 'UPENN', or 'ixi'."
            )
        
        self.input_modalities = input_modalities
        self.output_modality = output_modality
        
        # Validation for testing mode
        if self.testing:
            assert self.input_modalities is not None, \
                "Input modalities must be provided in testing mode."
            assert self.output_modality is not None, \
                "Output modality must be provided in testing mode."

        self.img_size = img_size

    def set_epoch(self, epoch: int):
        """
        Set current epoch (useful for epoch-dependent data augmentation).
        
        Args:
            epoch: Current epoch number
        """
        self.current_epoch = epoch

    def get_random_modalities(self) -> tuple:
        """
        Sample random input/output modality combination.
        
        In training mode:
            - Randomly chooses number of input modalities (1, 2, or 3)
            - Randomly selects which modalities to use as input
            - Randomly selects target modality (different from inputs)
        
        In testing mode:
            - Returns fixed modalities provided at initialization
        
        Returns:
            Tuple of (input_modalities: list, output_modality: str)
        """
        if self.testing:
            return self.input_modalities, self.output_modality

        # Randomly choose how many input modalities to use
        # Weights: [1-to-1, 2-to-1, 3-to-1] with equal probability
        weights = [1/3, 1/3, 1/3]
        input_size = random.choices([1, 2, 3], weights=weights)[0]
        
        # Randomly select output modality
        output_modality = random.choice(self.MODALITIES)
        
        # Select input modalities from remaining modalities
        remaining = [m for m in self.MODALITIES if m != output_modality]
        input_size = min(input_size, len(remaining))  # Safety check
        
        random.shuffle(remaining)
        input_modalities = remaining[:input_size]
        
        return input_modalities, output_modality

    def preprocess_image_vae(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for VAE encoding (MedVAE requires 3-channel input).
        
        Args:
            image: Input image array [H, W]
            
        Returns:
            Preprocessed tensor [3, H, W] normalized to [-1, 1]
        """
        image = torch.from_numpy(image).float()  # [H, W]
        image = image.unsqueeze(0).repeat(3, 1, 1)  # [3, H, W] - repeat to 3 channels
        image = F.interpolate(
            image.unsqueeze(0),
            size=self.img_size,
            mode="bilinear",
            align_corners=False
        ).squeeze(0)
        image = image * 2 - 1  # Normalize to [-1, 1]
        return image

    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for model input (single channel).
        
        Args:
            image: Input image array [H, W]
            
        Returns:
            Preprocessed tensor [1, H, W] normalized to [-1, 1]
        """
        image = torch.from_numpy(image).float()  # [H, W]
        image = image.unsqueeze(0)  # [1, H, W]
        image = F.interpolate(
            image.unsqueeze(0),
            size=self.img_size,
            mode="bilinear",
            align_corners=False
        ).squeeze(0)
        image = image * 2 - 1  # Normalize to [-1, 1]
        return image

    def __len__(self) -> int:
        """Return total number of samples in dataset."""
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> tuple:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple containing:
                - target_img_original: Original target image [H, W]
                - target_img: Preprocessed target image [1, H, W]
                - all_imgs: Concatenated input images [num_inputs, H, W]
                - one_hot_input_mod: One-hot encoding of available input modalities
                - one_hot_target_mod: One-hot encoding of target modality
                - input_imgs_mods: Modality indices for each slot (4 = NA/missing)
                - medvae_input_imgs: Input images for VAE encoding [num_slots*3, H, W]
                - medvae_target_img: Target image for VAE encoding [3, H, W]
                - img_path: Path to the image file
        """
        # Sample modality combination for this sample
        input_modalities, output_modality = self.get_random_modalities()

        # Load data
        data = np.load(self.img_paths[idx])

        # Extract individual modality images and initialize one-hot encodings
        if self.dataset_name in ["brats", "UCSF", "UPENN"]:
            imgs = [data[0, :, :], data[1, :, :], data[2, :, :], data[3, :, :]]
            one_hot_input_mod = torch.tensor([0, 0, 0, 0], dtype=torch.float32)
            one_hot_target_mod = torch.tensor([0, 0, 0, 0], dtype=torch.float32)
        elif self.dataset_name == "ixi":
            imgs = [data[0, :, :], data[1, :, :], data[2, :, :]]
            one_hot_input_mod = torch.tensor([0, 0, 0], dtype=torch.float32)
            one_hot_target_mod = torch.tensor([0, 0, 0], dtype=torch.float32)

        # Prepare containers for input modalities
        input_imgs = []
        input_imgs_mods = []
        medvae_input_imgs = []

        # Create zero placeholder for missing modalities
        all_zeros = np.zeros_like(imgs[0], dtype=imgs[0].dtype)

        # Build input modality slots
        # Non-target modalities are shuffled to randomize slot ordering
        non_target_modalities = [m for m in self.MODALITIES if m != output_modality]
        random.shuffle(non_target_modalities)

        for mod in non_target_modalities:
            index = self.MODALITY_MAPPING[mod]

            if mod in input_modalities:
                # Available modality: use real image
                img = self.preprocess_image(imgs[index].squeeze())
                input_imgs.append(img)
                one_hot_input_mod[index] = 1
                input_imgs_mods.append(index)

                medvae_img = self.preprocess_image_vae(imgs[index].squeeze())
                medvae_input_imgs.append(medvae_img)
            else:
                # Missing modality: fill with zeros
                img = self.preprocess_image(all_zeros.squeeze())
                input_imgs.append(img)
                input_imgs_mods.append(4)  # 4 = NA/missing modality indicator

                # For VAE, also use zeros for missing modalities
                medvae_img = self.preprocess_image_vae(all_zeros.squeeze())
                medvae_input_imgs.append(medvae_img)

        # Process target modality
        target_index = self.MODALITY_MAPPING[output_modality]
        target_img = self.preprocess_image(imgs[target_index].squeeze())
        
        # Keep original size target for evaluation metrics
        target_img_original = torch.from_numpy(imgs[target_index].squeeze().copy()).float()
        
        one_hot_target_mod[target_index] = 1
        medvae_target_img = self.preprocess_image_vae(imgs[target_index].squeeze())

        # Concatenate all inputs along channel dimension
        all_imgs = torch.cat(input_imgs, dim=0)
        medvae_input_imgs = torch.cat(medvae_input_imgs, dim=0)

        # Validate expected number of input channels
        if self.dataset_name in ["brats", "UCSF", "UPENN"]:
            assert all_imgs.shape[0] == 3, \
                f"Expected 3 input channels for {self.dataset_name}, got {all_imgs.shape[0]}"
        elif self.dataset_name == "ixi":
            assert all_imgs.shape[0] == 2, \
                f"Expected 2 input channels for IXI, got {all_imgs.shape[0]}"

        input_imgs_mods = torch.tensor(input_imgs_mods, dtype=torch.int64)

        return (
            target_img_original,
            target_img,
            all_imgs,
            one_hot_input_mod,
            one_hot_target_mod,
            input_imgs_mods,
            medvae_input_imgs,
            medvae_target_img,
            self.img_paths[idx]
        )
