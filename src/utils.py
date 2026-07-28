"""
Shared utility helpers for training and inference scripts.

Centralizes config loading and model construction so both entry points
build the exact same architecture from the same config keys.
"""

import yaml

from model import (
    DiT_S_2, DiT_S_4, DiT_S_8,
    DiT_B_2, DiT_B_4, DiT_B_8,
    DiT_L_2, DiT_L_4, DiT_L_8,
    DiT_XL_2, DiT_XL_4, DiT_XL_8,
)

# Registry maps the "{SIZE}_{PATCH}" architecture string from config.yaml
# (prefixed with "DiT_") to its constructor.
MODEL_REGISTRY = {
    'DiT_S_2': DiT_S_2, 'DiT_S_4': DiT_S_4, 'DiT_S_8': DiT_S_8,
    'DiT_B_2': DiT_B_2, 'DiT_B_4': DiT_B_4, 'DiT_B_8': DiT_B_8,
    'DiT_L_2': DiT_L_2, 'DiT_L_4': DiT_L_4, 'DiT_L_8': DiT_L_8,
    'DiT_XL_2': DiT_XL_2, 'DiT_XL_4': DiT_XL_4, 'DiT_XL_8': DiT_XL_8,
}

# Number of MRI modalities per dataset (BraTS/UCSF/UPENN: T1, T1CE, T2, FLAIR; IXI: PD, T1, T2)
NUM_MODALITIES = {'brats': 4, 'UCSF': 4, 'UPENN': 4, 'ixi': 3}


def load_config(path: str) -> dict:
    """Load a YAML config file into a dict."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def get_dataset_paths(config: dict, dataset_name: str) -> dict:
    """Return the {train, val, test} path dict for a dataset from config."""
    return config['dataset']['paths'][dataset_name]


def create_model(config: dict, dataset_name: str = None):
    """
    Build a DiT model from config['model'], sized for the given dataset.

    dataset_name defaults to config['dataset']['name'] but can be overridden
    (e.g. inference on a dataset different from the one named in the config).
    """
    dataset_name = dataset_name or config['dataset']['name']
    if dataset_name not in NUM_MODALITIES:
        raise ValueError(f"Unsupported dataset name: {dataset_name}")

    architecture = config['model']['architecture']
    model_name = f"DiT_{architecture}"
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model architecture: {architecture!r}. "
            f"Choose from {sorted(k.replace('DiT_', '') for k in MODEL_REGISTRY)}."
        )

    model_fn = MODEL_REGISTRY[model_name]
    return model_fn(
        input_size=config['model']['input_size'],
        in_channels=config['model']['in_channels'],
        num_modalities=NUM_MODALITIES[dataset_name],
        dataset_name=dataset_name,
    )
