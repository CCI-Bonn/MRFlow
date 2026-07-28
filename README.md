# Multi-Modal MRI Synthesis using Diffusion Transformers

A PyTorch implementation of MRFlow for synthesizing MRI modalities using rectified flow in the latent space. The model can generate missing MRI modalities from available ones, supporting flexible 1-to-1, 2-to-1, and 3-to-1 synthesis.

## 🎯 Overview

This project implements a conditional diffusion model that operates in the latent space of MedVAE to synthesize MRI modalities. The model uses:
- **Rectified Flow**: Straight-path optimal transport for efficient sampling
- **Latent Diffusion**: Operates in compressed latent space via MedVAE
- **Cross-Attention Fusion**: Integrates multiple input modalities
- **Flexible Input/Output**: Supports any combination of input/output modalities

See [`demo_generation_trajectory.ipynb`](demo_generation_trajectory.ipynb) for a walkthrough that generates a missing modality directly from raw `.nii.gz` files and visualizes the rectified-flow trajectory from `t=0` (noise) to `t=1` (final image).

### Supported Datasets

| Dataset | Modalities | Resolution | Use Case | Status |
|---------|-----------|------------|----------|--------|
| **BraTS** | T1, T1CE, T2, FLAIR (4) | Variable | Brain tumor segmentation | Trained checkpoint provided (`checkpoints/brats.pth`) |
| **IXI** | PD, T1, T2 (3) | Variable | Normal brain imaging | Trained checkpoint provided (`checkpoints/ixi.pth`), not independently verified |
| **UCSF** | T1, T1CE, T2, FLAIR (4) | Variable | Clinical brain imaging | Supported by the dataloader/model, but no checkpoint has been trained |
| **UPENN** | T1, T1CE, T2, FLAIR (4) | Variable | Clinical brain imaging | Supported by the dataloader/model, but no checkpoint has been trained |

## 📋 Requirements

### System Requirements
- Python 3.8+
- CUDA-capable GPU (recommended: 16GB+ VRAM)
- 32GB+ RAM recommended for training

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/mri-synthesis.git
cd mri-synthesis

# Create a conda environment (or use `python -m venv venv` instead)
conda create -n MRI_Gen python=3.10
conda activate MRI_Gen

# Install dependencies
pip install -r requirements.txt
```

## 📁 Project Structure

```
mri_synthesis_project/
├── config/
│   └── config.yaml              # Configuration file
├── src/
│   ├── model.py                 # DiT model architecture
│   ├── dataloader.py            # Dataset loading and preprocessing
│   ├── train.py                 # Training script
│   ├── inference.py             # Inference and evaluation script
│   └── utils.py                 # Config loading + model factory shared by train.py/inference.py
├── checkpoints/                 # Saved model checkpoints (brats.pth, ixi.pth)
├── outputs/
│   ├── validation/              # Validation outputs during training
│   └── inference/               # Test results
├── demo_generation_trajectory.ipynb  # Notebook: generate a modality from raw NIfTI, visualize t=0->t=1
├── requirements.txt
└── README.md
```

## 🗂️ Data Preparation

### Dataset Format

The model expects data in the following format:

```
dataset/
├── train/
│   ├── patient001/
│   │   ├── slice_001.npy       # Shape: [num_modalities, H, W]
│   │   ├── slice_002.npy
│   │   └── ...
│   ├── patient002/
│   └── ...
├── val/
│   └── ...
└── test/
    └── ...
```

### Data Structure

Each `.npy` file should contain a numpy array with shape `[num_modalities, H, W]`:

- **For BraTS/UCSF/UPENN**: `[4, H, W]` with modalities in order `[T1, T1CE, T2, FLAIR]`
- **For IXI**: `[3, H, W]` with modalities in order `[PD, T1, T2]`

### Preprocessing Example

```python
import numpy as np
import nibabel as nib

# Load NIfTI files
t1 = nib.load('t1.nii.gz').get_fdata()
t1ce = nib.load('t1ce.nii.gz').get_fdata()
t2 = nib.load('t2.nii.gz').get_fdata()
flair = nib.load('flair.nii.gz').get_fdata()

# For 2D: extract a slice
slice_idx = 100
t1_slice = t1[:, :, slice_idx]
t1ce_slice = t1ce[:, :, slice_idx]
t2_slice = t2[:, :, slice_idx]
flair_slice = flair[:, :, slice_idx]

# Stack modalities
combined = np.stack([t1_slice, t1ce_slice, t2_slice, flair_slice], axis=0)

# Normalize to [0, 1]
combined = (combined - combined.min()) / (combined.max() - combined.min())

# Save as numpy array
np.save('slice_100.npy', combined)
```

### Important Notes

1. **Normalization**: Images should be normalized to [0, 1] range
2. **Resolution**: Original resolution is preserved; the model resizes during training
3. **Missing Modalities**: Not applicable during preprocessing; handled by the model during training/inference

## ⚙️ Configuration

Edit `config/config.yaml` to customize:

### Key Configuration Sections

```yaml
dataset:
  name: 'brats'  # Choose: brats, ixi, UCSF, UPENN
  paths:
    brats:
      train: "/path/to/train"
      val: "/path/to/val"
      test: "/path/to/test"
  img_size: [256, 256]

model:
  architecture: 'S_2'  # Only S_2 has a trained checkpoint; see "Model Architectures" below

training:
  epochs: 200
  batch_size: 8
  learning_rate: 1.0e-4
  eps: 1.0e-3     # rectified-flow minimum timestep
  T: 1.0          # rectified-flow maximum timestep
  t_scale: 999.0  # timestep scaling factor fed into the model
  seed: 42
```

### Model Architectures

`model.py` defines the full DiT size family below (`{SIZE}_{PATCH_SIZE}`, used as `config.model.architecture`), but **only `S_2` has actually been trained** — that's the architecture behind both provided checkpoints (`checkpoints/brats.pth`, `checkpoints/ixi.pth`). The others are implemented and can be trained with `train.py`, but come with no checkpoint and haven't been validated by us.

| Architecture | Depth | Hidden Size | Parameters (BraTS, 4 modalities) | Status |
|--------------|-------|-------------|-----------------------------------|--------|
| S_2 | 12 | 384 | 38.2M | **Trained** — checkpoints provided |
| B_2 | 12 | 768 | 151.6M | Untrained — implemented, feel free to try |
| L_2 | 24 | 1024 | 495.7M | Untrained — implemented, feel free to try |
| XL_2 | 28 | 1152 | 722.7M | Untrained — implemented, feel free to try |

Patch sizes 4 and 8 (e.g. `S_4`, `B_8`) are also implemented for each size and follow the same "untrained, feel free to try" status.

## 🚀 Training

### Basic Training

```bash
# Train on BraTS dataset
python src/train.py --config config/config.yaml --dataset brats

# Train on IXI dataset
python src/train.py --config config/config.yaml --dataset ixi
```

### Resume Training

1. Set in `config.yaml`:
```yaml
training:
  continue_training: true
  checkpoint_path: 'checkpoints/brats/epoch_50_rectifiedFlow.pth'
  reset_epoch: false  # Set true to restart epoch counter
```

2. Run training command as usual

### Monitor Training

- **Checkpoints**: Saved every N epochs (configurable) in `checkpoints/<dataset_name>/` (e.g. `checkpoints/brats/`) — kept separate from the pretrained `checkpoints/brats.pth` / `checkpoints/ixi.pth` so a new training run never overwrites them
- **Best Model**: Automatically saved as `best.pth` based on validation loss
- **Visualizations**: Generated during validation in `outputs/validation/`
- **Logs**: Printed to console with loss, metrics, and progress

### Training Tips

1. **Batch Size**: Reduce if running out of memory
2. **Learning Rate**: 1e-4 is a good starting point; reduce if training is unstable
3. **Validation Interval**: Set to 5-10 epochs for faster training
4. **Data Augmentation**: Currently not implemented; can be added to dataloader.py

## 🔮 Inference

### Run Evaluation

```bash
python src/inference.py \
    --config config/config.yaml \
    --dataset brats \
    --checkpoint checkpoints/brats.pth \
    --n_steps 10 \
    --sigma 1.0
```

### Arguments

- `--config`: Path to configuration file
- `--dataset`: Dataset to evaluate (brats, ixi, UCSF, UPENN)
- `--checkpoint`: Path to trained model checkpoint
- `--n_steps`: Number of ODE solver steps (10-100, higher = better quality but slower)
- `--sigma`: Noise scaling (0.0 = deterministic, 1.0 = stochastic)
- `--save_images`: Add this flag to save generated images as .npy files

### Output

The inference script generates:

1. **Metrics** (`result.txt`): Per-sample PSNR, SSIM, MS-SSIM, LPIPS
2. **Summary Statistics**: Mean and std for all metrics + FID score
3. **Visualizations**: Comparison plots (input, target, prediction)
4. **Generated Images** (optional): Saved as .npy files organized by modality

### Metrics Explained

| Metric | Range | Better | Description |
|--------|-------|--------|-------------|
| **PSNR** | [0, ∞) | Higher | Peak Signal-to-Noise Ratio |
| **SSIM** | [0, 1] | Higher | Structural Similarity Index |
| **MS-SSIM** | [0, 1] | Higher | Multi-Scale SSIM |
| **LPIPS** | [0, ∞) | Lower | Perceptual similarity (VGG) |
| **HFEN** | [0, ∞) | Lower | High-Frequency Error Norm |
| **FID** | [0, ∞) | Lower | Fréchet Inception Distance |
| **L1** | [0, ∞) | Lower | Mean Absolute Error |
| **MSE** | [0, ∞) | Lower | Mean Squared Error |

## 📊 Example Results

### Modality Synthesis Examples

```
Input: T1, T2          → Output: FLAIR     (2-to-1)
Input: T1, T1CE, T2    → Output: FLAIR     (3-to-1)
Input: FLAIR           → Output: T2        (1-to-1)
```

### Example Measured Quality (BraTS, S_2 checkpoint)

Measured by actually running `checkpoints/brats.pth` through `src/inference.py`'s evaluation
pipeline on a small sample of real BraTS test slices (not a full benchmark run — just a
sanity check that the pipeline produces sensible image quality):

| Synthesis Task | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---------------|--------|--------|---------|
| T1+T1CE+T2→FLAIR | 24.7 ± 1.8 | 0.87 ± 0.03 | 0.086 ± 0.016 |

For a full evaluation across every modality combination in `config.yaml`, run:
```bash
python src/inference.py --config config/config.yaml --dataset brats --checkpoint checkpoints/brats.pth
```

## 🔧 Advanced Usage

### Custom Modality Combinations

Edit `config.yaml` to test specific combinations:

```yaml
inference:
  modality_tests:
    brats:
      custom:
        - [['t1', 't2'], 'flair']        # Your custom combinations
        - [['t1'], 't1ce']
```

### Adjusting ODE Solver

The project includes three ODE solvers:

- **Euler** (fastest, least accurate): Use for quick testing
- **Heun** (balanced): Default, good accuracy-speed tradeoff
- **RK4** (slowest, most accurate): Use for best quality results

Change solver in `inference.py`:
```python
ode_solver=heun_ode  # Change to euler_ode or rk4_ode
```

### Model Checkpointing Strategy

- **Periodic**: Saves every N epochs (set `save_interval`)
- **Best Model**: Automatically saves when validation loss improves
- **Resumable**: All checkpoints include optimizer state for seamless resuming

## 🐛 Troubleshooting

### Common Issues

**Out of Memory (OOM)**
```
Solution:
1. Reduce batch_size in config.yaml
2. Use smaller model (S_2 instead of B_2)
3. Reduce img_size in config
```

**CUDA Error**
```
Solution:
1. Update CUDA and PyTorch versions
2. Check GPU compatibility
3. Try: export CUDA_LAUNCH_BLOCKING=1
```

**ModuleNotFoundError: medvae**
```
Solution:
pip install medvae
```

**Low Metrics**
```
Possible causes:
1. Insufficient training (try more epochs)
2. Learning rate too high/low
3. Data quality issues
4. Check data normalization
```

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- **DiT (Diffusion Transformer)**: Original architecture from Meta AI
- **MedVAE**: Medical image VAE for latent encoding
- **Rectified Flow**: Optimal transport-based diffusion approach
- **Dataset Providers**: BraTS, IXI, UCSF, UPENN for MRI data

## 🗺️ Roadmap

- [ ] 3D volume synthesis support
- [ ] Additional modality types (CT, PET)
- [ ] Conditional synthesis with clinical metadata
- [ ] Web demo interface
- [ ] Train and validate the B/L/XL architectures (currently S_2-only)
- [ ] Multi-GPU training support
- [ ] Docker containerization

---

**Note**: This is research code. Always validate generated medical images with domain experts before clinical use.
