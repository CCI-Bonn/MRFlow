# Quick Start Guide

Get your MRI synthesis model running in 5 minutes!

## 1. Installation (2 minutes)

```bash
# Clone and setup
git clone https://github.com/yourusername/mri-synthesis.git
cd mri-synthesis
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Prepare Your Data (varies)

Your data should be organized as:
```
your_dataset/
├── train/
│   └── patient_*/slice_*.npy  # Shape: [num_modalities, H, W]
├── val/
└── test/
```

Update paths in `config/config.yaml`:
```yaml
dataset:
  paths:
    brats:
      train: "/your/path/train"
      val: "/your/path/val"
      test: "/your/path/test"
```

## 3. Train (1 command)

```bash
python src/train.py --config config/config.yaml --dataset brats
```

Monitor progress in terminal. Checkpoints saved to `checkpoints/`.

## 4. Inference (1 command)

```bash
python src/inference.py \
    --config config/config.yaml \
    --dataset brats \
    --checkpoint checkpoints/best.pth \
    --n_steps 10
```

Results saved to `outputs/inference/`.

## 5. View Results

Check `outputs/inference/`:
- `result.txt` - Per-sample metrics
- `*.png` - Visual comparisons
- Console output - Summary statistics

## Common Modifications

### Change Model Size
Edit `config/config.yaml`:
```yaml
model:
  architecture: 'B_2'  # S_2 (small), B_2 (base), L_2 (large), XL_2 (xlarge)
```

### Reduce Memory Usage
```yaml
training:
  batch_size: 4  # Reduce from 8
dataset:
  img_size: [128, 128]  # Reduce from [256, 256]
```

### Faster Training
```yaml
training:
  validate_interval: 10  # Validate less frequently
  save_interval: 10  # Save less frequently
```

## Troubleshooting

**Out of Memory?**
- Reduce `batch_size` to 4 or 2
- Use smaller model (`S_2`)
- Reduce `img_size`

**Model not learning?**
- Check data normalization ([0, 1] range)
- Try different `learning_rate` (1e-5 to 1e-3)
- Train for more `epochs`

**Slow inference?**
- Reduce `n_steps` (minimum 5, good quality at 10-20)
- Use `euler_ode` instead of `heun_ode`

## Next Steps

- Read full README.md for detailed documentation
- Experiment with different modality combinations
- Try different ODE solvers
- Tune hyperparameters for your dataset

## Support

- Issues: GitHub Issues
- Questions: Open a discussion
- Bugs: Submit a detailed issue with error logs

Happy synthesizing! 🎉
