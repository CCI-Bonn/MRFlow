"""
Diffusion Transformer (DiT) Model for MRI Modality Synthesis

This module implements a Diffusion Transformer architecture for synthesizing
MRI modalities using rectified flow. The model uses cross-attention to fuse
information from multiple input modalities and generates target modalities
in the latent space.

Based on: Scalable Diffusion Models with Transformers (DiT)
Modified for multi-modal MRI synthesis with latent space operation.
"""

import torch
from torch import Tensor
from typing import Tuple
import torch.nn as nn
import numpy as np
import math

# Must run before importing timm: timm pulls in wandb (via timm.utils.summary),
# and older wandb releases touch np.float_/np.complex_, which NumPy >=2.0 removed.
np.float_ = np.float64
np.complex_ = np.complex128

from timm.models.vision_transformer import Attention as SelfAttention
from timm.models.vision_transformer import PatchEmbed, Mlp
import torch.nn.functional as F


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    """
    Apply adaptive layer norm modulation.
    
    Args:
        x: Input tensor of shape [N, T, D]
        shift: Shift parameter of shape [N, D]
        scale: Scale parameter of shape [N, D]
        
    Returns:
        Modulated tensor of shape [N, T, D]
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# =============================================================================
# Embedding Layers for Timesteps and Modalities
# =============================================================================

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations using sinusoidal encoding.
    """
    
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: Tensor, dim: int, max_period: int = 10000) -> Tensor:
        """
        Create sinusoidal timestep embeddings.
        
        Args:
            t: Timestep tensor of shape [N]
            dim: Dimension of the output embedding
            max_period: Maximum period for sinusoidal encoding
            
        Returns:
            Embedding tensor of shape [N, dim]
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: Tensor) -> Tensor:
        """
        Forward pass for timestep embedding.
        
        Args:
            t: Timestep tensor of shape [N]
            
        Returns:
            Embedded timestep of shape [N, hidden_size]
        """
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


# =============================================================================
# Core DiT Building Blocks
# =============================================================================

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm (adaLN) conditioning.
    Combines self-attention and MLP with conditioning from timestep and modality.
    """
    
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.final_attn = SelfAttention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0
        )
        
        # AdaLN modulation generates 6 parameters: scale/shift for attention and MLP, plus gates
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        """
        Forward pass through DiT block.
        
        Args:
            x: Input tensor of shape [N, T, D]
            c: Conditioning vector of shape [N, D]
            
        Returns:
            Output tensor of shape [N, T, D]
        """
        # Generate adaLN parameters
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)
        
        # Self-attention with adaLN
        x = x + gate_msa.unsqueeze(1) * self.final_attn(modulate(self.norm1(x), shift_msa, scale_msa))
        
        # MLP with adaLN
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        
        return x


class FinalLayer(nn.Module):
    """
    Final layer of DiT that projects back to image space.
    Uses adaLN conditioning before the final linear projection.
    """
    
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        """
        Forward pass through final layer.
        
        Args:
            x: Input tensor of shape [N, T, D]
            c: Conditioning vector of shape [N, D]
            
        Returns:
            Output tensor of shape [N, T, patch_size^2 * out_channels]
        """
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class Attention(nn.Module):
    """
    Multi-head attention module for cross-attention fusion.
    Used to fuse information from input modalities into the target representation.
    """
    
    def __init__(self, embedding_dim: int, num_heads: int, downsample_rate: int = 1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        """Reshape tensor to separate attention heads."""
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # [B, num_heads, N, C_per_head]

    def _recombine_heads(self, x: Tensor) -> Tensor:
        """Recombine attention heads back to original shape."""
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, attn_mask: Tensor = None) -> Tensor:
        """
        Cross-attention forward pass.
        
        Args:
            q: Query tensor of shape [N, T_q, D]
            k: Key tensor of shape [N, T_k, D]
            v: Value tensor of shape [N, T_k, D]
            attn_mask: Optional attention mask
            
        Returns:
            Output tensor of shape [N, T_q, D]
        """
        # Project Q, K, V
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Compute attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # [B, heads, T_q, T_k]
        attn = attn / math.sqrt(c_per_head)

        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask == 0, -1e9)

        attn = torch.softmax(attn, dim=-1)
        out = attn @ v  # [B, heads, T_q, C_per_head]
        
        # Recombine heads and project
        out = self._recombine_heads(out)
        out = self.out_proj(out)
        return out


# =============================================================================
# Positional Embedding Functions
# =============================================================================

def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int, 
                            cls_token: bool = False, extra_tokens: int = 0) -> np.ndarray:
    """
    Generate 2D sinusoidal positional embeddings.
    
    Args:
        embed_dim: Embedding dimension
        grid_size: Grid size (assuming square grid)
        cls_token: Whether to include class token
        extra_tokens: Number of extra tokens to prepend
        
    Returns:
        Positional embedding array of shape [grid_size^2, embed_dim]
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    """Generate 2D positional embedding from grid coordinates."""
    assert embed_dim % 2 == 0
    
    # Use half of dimensions for height, half for width
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """
    Generate 1D sinusoidal positional embedding.
    
    Args:
        embed_dim: Output dimension for each position
        pos: Position array
        
    Returns:
        Positional embedding of shape [M, embed_dim]
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega

    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


# =============================================================================
# Main DiT Model
# =============================================================================

class DiT(nn.Module):
    """
    Diffusion Transformer for Multi-Modal MRI Synthesis.
    
    This model operates in the latent space and uses cross-attention to fuse
    information from multiple input modalities to synthesize a target modality.
    The model is conditioned on timestep and target modality identity.
    
    Architecture:
        1. Patch embedding for noisy target and input modalities
        2. Cross-attention fusion stack to integrate input information
        3. Stack of DiT blocks with adaLN conditioning
        4. Final layer to project back to latent space
    """
    
    def __init__(
        self,
        input_size: int = 256,
        patch_size: int = 2,
        in_channels: int = 1,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        num_modalities: int = 4,
        class_dropout_prob: float = 0.1,
        dataset_name: str = 'brats',
    ):
        """
        Initialize DiT model.
        
        Args:
            input_size: Input latent size (H, W)
            patch_size: Patch size for tokenization
            in_channels: Number of input channels in latent space
            hidden_size: Hidden dimension
            depth: Number of transformer blocks
            num_heads: Number of attention heads
            mlp_ratio: MLP expansion ratio
            num_modalities: Number of modalities (3 for IXI, 4 for BraTS)
            class_dropout_prob: Dropout probability (unused, kept for compatibility)
            dataset_name: Dataset name ('brats', 'ixi', 'UCSF', 'UPENN')
        """
        super().__init__()
        self.dataset_name = dataset_name
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.num_fusion_layers = 3  # Number of cross-attention fusion layers

        # ─────────────────────────────────────────────────────────────────────
        # Patch Embedders
        # ─────────────────────────────────────────────────────────────────────
        # Separate embedders for noisy target and each input slot
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.y1_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.y2_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        
        # Third embedder only for datasets with 4 modalities (BraTS, UCSF, UPENN)
        if self.dataset_name in ['brats', 'UCSF', 'UPENN']:
            self.y3_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        else:
            self.y3_embedder = None

        # ─────────────────────────────────────────────────────────────────────
        # Timestep Embedder
        # ─────────────────────────────────────────────────────────────────────
        self.t_embedder = TimestepEmbedder(hidden_size)

        # ─────────────────────────────────────────────────────────────────────
        # Modality Embeddings
        # ─────────────────────────────────────────────────────────────────────
        # Embedding for target modality identity
        self.target_embed = nn.Embedding(num_modalities, hidden_size)
        
        # Embedding for input modality identities (num_modalities + 1 for NA/missing)
        self.input_modalities_embed = nn.Embedding(num_modalities + 1, hidden_size)
        
        # Embedding to indicate presence (1) or absence (0) of a modality
        self.presence_embed = nn.Embedding(2, hidden_size)

        # ─────────────────────────────────────────────────────────────────────
        # Positional Embedding
        # ─────────────────────────────────────────────────────────────────────
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        # ─────────────────────────────────────────────────────────────────────
        # Cross-Attention Fusion Stack
        # ─────────────────────────────────────────────────────────────────────
        # These layers fuse input modality information into the target representation
        self.cross_attn_fusion = nn.ModuleList([
            Attention(embedding_dim=hidden_size, num_heads=num_heads)
            for _ in range(self.num_fusion_layers)
        ])
        
        self.cross_attn_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            for _ in range(self.num_fusion_layers)
        ])
        
        # MLP after each cross-attention layer
        mlp_hidden_dim = int(hidden_size * 4)
        self.fusion_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, mlp_hidden_dim),
                nn.GELU(),
                nn.Linear(mlp_hidden_dim, hidden_size)
            ) for _ in range(self.num_fusion_layers)
        ])
        
        self.mlp_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            for _ in range(self.num_fusion_layers)
        ])

        # ─────────────────────────────────────────────────────────────────────
        # DiT Blocks
        # ─────────────────────────────────────────────────────────────────────
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])

        # ─────────────────────────────────────────────────────────────────────
        # Final Layer
        # ─────────────────────────────────────────────────────────────────────
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        
        # Initialize weights
        self.initialize_weights()

    def initialize_weights(self):
        """Initialize model weights following DiT initialization scheme."""
        
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
        self.apply(_basic_init)

        # Initialize positional embedding with sinusoidal encoding
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.x_embedder.num_patches ** 0.5)
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch embedding weights
        for embedder in [self.x_embedder, self.y1_embedder, self.y2_embedder]:
            w = embedder.proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            nn.init.constant_(embedder.proj.bias, 0)
        
        if self.y3_embedder is not None:
            w = self.y3_embedder.proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            nn.init.constant_(self.y3_embedder.proj.bias, 0)

        # Initialize timestep embedder
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Initialize modality embeddings
        nn.init.normal_(self.target_embed.weight, std=0.02)
        nn.init.normal_(self.input_modalities_embed.weight, std=0.02)
        nn.init.normal_(self.presence_embed.weight, std=0.02)

        # Zero-out adaLN modulation layers (standard DiT initialization)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out final layer
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

        # Initialize fusion MLPs
        for mlp in self.fusion_mlps:
            nn.init.xavier_uniform_(mlp[0].weight)
            nn.init.constant_(mlp[0].bias, 0)
            nn.init.xavier_uniform_(mlp[2].weight)
            nn.init.constant_(mlp[2].bias, 0)

    def unpatchify(self, x: Tensor) -> Tensor:
        """
        Convert sequence of patch tokens back to image format.
        
        Args:
            x: Patch tokens of shape [N, num_patches, patch_size^2 * C]
            
        Returns:
            Image tensor of shape [N, C, H, W]
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def _build_conditioning(self, y: Tensor, input_imgs_mods: Tensor, num_slots: int) -> Tensor:
        """
        Embed and concatenate input modality latents into a single conditioning sequence.
        Each slot is embedded with modality identity and presence/absence signals.
        
        Args:
            y: Concatenated input modalities [N, num_slots*in_channels, H, W]
            input_imgs_mods: Modality indices for each slot [N, num_slots]
                            Value of 4 indicates NA/missing modality
            num_slots: Number of input slots (2 for IXI, 3 for BraTS)
            
        Returns:
            Conditioning sequence [N, num_slots*T, D] where T is number of patches
        """
        embedders = [self.y1_embedder, self.y2_embedder]
        if self.y3_embedder is not None:
            embedders.append(self.y3_embedder)

        y_slots = []
        for i in range(num_slots):
            # Extract latent for this slot
            yi_raw = y[:, i * self.in_channels:(i + 1) * self.in_channels, :, :]
            
            # Embed patches and add positional encoding
            yi = embedders[i](yi_raw) + self.pos_embed  # [N, T, D]

            # Add modality identity embedding
            mod_emb = self.input_modalities_embed(input_imgs_mods[:, i])  # [N, D]
            yi = yi + mod_emb.unsqueeze(1)

            # Add presence/absence embedding (0 if missing/NA, 1 if present)
            presence_idx = (input_imgs_mods[:, i] != 4).long()  # [N]
            pres_emb = self.presence_embed(presence_idx)         # [N, D]
            yi = yi + pres_emb.unsqueeze(1)

            y_slots.append(yi)

        # Concatenate all input slots into single sequence
        y_seq = torch.cat(y_slots, dim=1)  # [N, num_slots*T, D]
        return y_seq

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        y: Tensor,
        one_hot_target_modality: Tensor,
        input_imgs_mods: Tensor = None
    ) -> Tensor:
        """
        Forward pass of DiT model.
        
        Args:
            x: Noisy target latent [N, in_channels, H, W]
            t: Timesteps [N], scaled by 999 by the caller
            y: Concatenated input modality latents [N, num_slots*in_channels, H, W]
            one_hot_target_modality: One-hot encoding of target modality [N, num_modalities]
            input_imgs_mods: Modality indices for each input slot [N, num_slots]
                           Value of 4 indicates NA/missing modality
            
        Returns:
            Predicted velocity field in latent space [N, in_channels, H, W]
        """
        # ─────────────────────────────────────────────────────────────────────
        # Embed noisy target latent
        # ─────────────────────────────────────────────────────────────────────
        x = self.x_embedder(x) + self.pos_embed  # [N, T, D]

        # ─────────────────────────────────────────────────────────────────────
        # Embed timestep and target modality
        # ─────────────────────────────────────────────────────────────────────
        t_emb = self.t_embedder(t)  # [N, D]
        target_modality_emb = self.target_embed(
            torch.argmax(one_hot_target_modality, dim=1)
        )  # [N, D]

        # ─────────────────────────────────────────────────────────────────────
        # Build conditioning sequence from input modalities
        # ─────────────────────────────────────────────────────────────────────
        num_slots = 3 if self.dataset_name in ['brats', 'UCSF', 'UPENN'] else 2
        y_seq = self._build_conditioning(y, input_imgs_mods, num_slots)  # [N, num_slots*T, D]

        # Inject target modality information into conditioning sequence
        y_seq = y_seq + target_modality_emb.unsqueeze(1)

        # ─────────────────────────────────────────────────────────────────────
        # Cross-attention fusion: integrate input information into target
        # ─────────────────────────────────────────────────────────────────────
        for i in range(self.num_fusion_layers):
            # Cross-attention: target attends to input modalities
            x = x + self.cross_attn_fusion[i](
                self.cross_attn_norms[i](x),  # Query from target
                y_seq,                         # Key from inputs
                y_seq                          # Value from inputs
            )
            # MLP refinement
            x = x + self.fusion_mlps[i](self.mlp_norms[i](x))

        # ─────────────────────────────────────────────────────────────────────
        # AdaLN conditioning combines timestep and target modality
        # ─────────────────────────────────────────────────────────────────────
        # Input content is already integrated via cross-attention fusion above
        c = t_emb + target_modality_emb  # [N, D]

        # ─────────────────────────────────────────────────────────────────────
        # Process through DiT blocks
        # ─────────────────────────────────────────────────────────────────────
        for block in self.blocks:
            x = block(x, c)

        # ─────────────────────────────────────────────────────────────────────
        # Final projection back to latent space
        # ─────────────────────────────────────────────────────────────────────
        x = self.final_layer(x, c)  # [N, T, patch_size^2 * out_channels]
        x = self.unpatchify(x)       # [N, out_channels, H, W]
        
        return x


# =============================================================================
# Model Configuration Helpers
# =============================================================================

def DiT_XL_2(**kwargs): 
    """DiT-XL/2: Extra Large model with patch size 2."""
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_XL_4(**kwargs): 
    """DiT-XL/4: Extra Large model with patch size 4."""
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiT_XL_8(**kwargs): 
    """DiT-XL/8: Extra Large model with patch size 8."""
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiT_L_2(**kwargs):
    """DiT-L/2: Large model with patch size 2."""
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiT_L_4(**kwargs):
    """DiT-L/4: Large model with patch size 4."""
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def DiT_L_8(**kwargs):
    """DiT-L/8: Large model with patch size 8."""
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def DiT_B_2(**kwargs):
    """DiT-B/2: Base model with patch size 2."""
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_B_4(**kwargs):
    """DiT-B/4: Base model with patch size 4."""
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiT_B_8(**kwargs):
    """DiT-B/8: Base model with patch size 8."""
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiT_S_2(**kwargs):
    """DiT-S/2: Small model with patch size 2."""
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_S_4(**kwargs):
    """DiT-S/4: Small model with patch size 4."""
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiT_S_8(**kwargs):
    """DiT-S/8: Small model with patch size 8."""
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)
