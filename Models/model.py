import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Type


# Configuration
LATENT_DIM = 256

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

TASKS = list(MODALITY_SUFFIXES.keys())



##################################################################################################################################################################################################################



class LatentFusionModel(nn.Module):
    def __init__(self, input_channels=32, num_modalities=4, hidden_dim=128, num_channels_latent_space=4):
        super().__init__()
        self.input_channels = input_channels  # mean + std = 32
        self.num_modalities = num_modalities  # 4 modalities
        self.num_channels_latent_space = num_channels_latent_space

        # CNN Backbone
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels + num_modalities + num_modalities, hidden_dim, kernel_size=3, padding=1),  # + one-hot inputs
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Heads to predict latent mean & std
        self.mean_head = nn.Conv2d(hidden_dim, self.num_channels_latent_space, kernel_size=1)
        self.std_head = nn.Conv2d(hidden_dim, self.num_channels_latent_space, kernel_size=1)

    def forward(self, latent_mean, latent_std, one_hot_input_mod, one_hot_target_mod):
        """
        latent_mean, latent_std: [B, 16, 64, 64]
        one_hot_input_mod, one_hot_target_mod: [B, 4]
        """
        
        B, _, H, W = latent_mean.shape

        # Concatenate latent stats
        x = torch.cat([latent_mean, latent_std], dim=1)  # [B, 32, H, W]

        # Expand one-hot vectors into spatial maps
        input_mod_map = one_hot_input_mod.view(B, self.num_modalities, 1, 1).expand(-1, -1, H, W)
        target_mod_map = one_hot_target_mod.view(B, self.num_modalities, 1, 1).expand(-1, -1, H, W)

        # Concatenate everything
        x = torch.cat([x, input_mod_map, target_mod_map], dim=1)  # [B, 32+4+4=40, H, W]

        # Pass through CNN
        features = self.encoder(x)

        # Predict modality-wise outputs
        pred_mean = self.mean_head(features)  # [B, 4, H, W]
        pred_std = F.softplus(self.std_head(features))  # [B, 4, H, W], enforce positivity

        return pred_mean, pred_std







#######################################################################################################################################################################################################





class LatentFusionTransformerModel(nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        num_modalities: int = 4,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
    ) -> None:
        """
        A transformer decoder that attends to an input image using
        queries whose positional embedding is supplied.

        Args:
          depth (int): number of layers in the transformer
          embedding_dim (int): the channel dimension for the input embeddings
          num_heads (int): the number of heads for multihead attention. Must
            divide embedding_dim
          activation (nn.Module): the activation to use in the MLP block
        """
        super().__init__()
        
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.layers = nn.ModuleList()
        
        '''
        for i in range(depth):
            self.layers.append(
                ModifiedTwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                )
            )
        '''
        
        self.num_modalities = 4
        self.latent_channels_per_modality = 4
        
        # Modality embedding layer
        self.modality_embedding = nn.Embedding(num_modalities, self.latent_channels_per_modality)
        
        # Add MLP for target modality => MLP(4, 4*4)
        self.target_modality_embedding = nn.Linear(self.num_modalities, self.num_modalities*self.latent_channels_per_modality)
        
        self.norm = nn.LayerNorm(embedding_dim)
        
        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                Attention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate),
                
            )
        
        self.MLP = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim // 2),
                                 nn.ReLU(),
                                 nn.Linear(self.embedding_dim // 2, self.embedding_dim))
        
        self.mean_projector = nn.Conv2d(self.num_modalities*self.latent_channels_per_modality, self.latent_channels_per_modality, kernel_size=1)
        self.std_projector = nn.Conv2d(self.num_modalities*self.latent_channels_per_modality, self.latent_channels_per_modality, kernel_size=1)
        
        
    def forward(
        self,
        latent_mean,
        latent_std,
        one_hot_input_mod, 
        one_hot_target_mod,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
          latent_mean (torch.Tensor): latent means of input modalities (B x 16 x 32 x 32)  :: 16==> 4 x 32 x 32 (for 1 modality) ::: total 4 modalities
          latent_std (torch.Tensor): latent std of input modalities (B x 16 x 32 x 32)  :: 16==> 4 x 32 x 32 (for 1 modality) ::: total 4 modalities
          one_hot_input_mod (torch.Tensor): one hot vector denoting the input modalities (B x 4)  :: B x [T1, T1ce, T2, flair]
          one_hot_target_mod (torch.Tensor): one hot vector denoting the output modality (B x 4)  :: B x [T1, T1ce, T2, flair]
          
          C = 16
          H = 32
          W = 32
          H and W depend on the size of the image.
          If original img size is 512 then, H,W = 64,64
          
        Returns:
          torch.Tensor: pred latent mean of target modality 
          torch.Tensor: pred latent std of target modality
        """
        
        bs, c, h, w = latent_mean.shape
        
        # Initialize embedding map to add
        embedding_map = torch.zeros_like(latent_mean)  # [B, C, H, W]
        
        # Loop through modalities
        for mod_idx in range(self.num_modalities):
            # Mask: [B, 1, 1, 1]
            mask = one_hot_input_mod[:, mod_idx].view(bs, 1, 1, 1)

            # Embedding for this modality: [1, 4]
            emb = self.modality_embedding(torch.tensor(mod_idx, device=latent_mean.device)).view(1, self.latent_channels_per_modality, 1, 1)
            
            # Broadcast embedding to match shape [B, 4, H, W]
            emb = emb.expand(bs, -1, h, w)

            # Apply mask
            emb = emb * mask  # Zero out for missing modalities

            # Insert into correct channel slice (e.g., 0–3 for mod 0, 4–7 for mod 1, etc.)
            start = mod_idx * self.latent_channels_per_modality
            end = start + self.latent_channels_per_modality
            embedding_map[:, start:end, :, :] = emb
        
        # Get target modality embedding
        # one_hot_target_mod: shape [B, 4]
        target_emb = self.target_modality_embedding(one_hot_target_mod)  # [B, C]
        
        #. Reshape and broadcast for adding to latent maps
        target_emb = target_emb.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        target_emb = target_emb.expand(-1, -1, h, w)         # [B, C, H, W]
        
        # Add to latent tensors
        latent_mean = latent_mean + embedding_map + target_emb
        latent_std = latent_std + embedding_map + target_emb
        
        # Concatenate along the channel dimension (dim=1)
        latent_combined = torch.cat([latent_mean, latent_std], dim=1)  # [B, 2C, H, W]
        
        # B x 2C x H x W -> B x H W x 2C == B x N_image_tokens x 2C
        latent_combined = latent_combined.flatten(2).permute(0, 2, 1)
        
        # Apply attention blocks
        for layer in self.layers:
            attn_out = layer(q=latent_combined, k=latent_combined, v=latent_combined)
            latent_combined = latent_combined + attn_out
            latent_combined = self.norm(latent_combined)
            mlp_out = self.MLP(latent_combined)
            latent_combined = latent_combined + mlp_out
            latent_combined = self.norm(latent_combined)  # [B, N=H*W, 2C]

        latent_combined = latent_combined.permute(0, 2, 1).contiguous()  # [B, 2C, N=H*W]
        latent_combined = latent_combined.view(bs, 2*c, h, w)  # [B, 2C, H, W]
        
        mean, std = torch.chunk(latent_combined, chunks=2, dim=1)  # [B, C, H, W] each
        
        mean = self.mean_projector(mean)  # [B, 4, H, W]
        std  = self.std_projector(std)  # [B, 4, H, W]
        
        std = torch.sigmoid(std)
        
        return mean, std






########################################################################################################################################################################################################





class PositionalEncoding2D(torch.nn.Module):
    def __init__(self, dim, height, width):
        super().__init__()
        self.height = height
        self.width = width
        self.dim = dim
        self.register_buffer("pos_encoding", self.create_sinusoidal_encoding(dim, height, width))

    def create_sinusoidal_encoding(self, dim, height, width):
        if dim % 4 != 0:
            raise ValueError("Dimension must be divisible by 4 for 2D positional encoding.")

        pe = torch.zeros(dim, height, width)

        div_term = torch.exp(torch.arange(0, dim // 2, 2) * (-math.log(10000.0) / (dim // 2)))

        pos_w = torch.arange(0, width).unsqueeze(1)
        pos_h = torch.arange(0, height).unsqueeze(1)

        for i in range(dim // 4):
            pe[2*i, :, :] = torch.sin(pos_w * div_term[i]).transpose(0, 1).repeat(height, 1)
            pe[2*i+1, :, :] = torch.cos(pos_w * div_term[i]).transpose(0, 1).repeat(height, 1)
            pe[dim//2 + 2*i, :, :] = torch.sin(pos_h * div_term[i]).repeat(1, width)
            pe[dim//2 + 2*i+1, :, :] = torch.cos(pos_h * div_term[i]).repeat(1, width)

        return pe.unsqueeze(0)  # [1, dim, H, W]

    def forward(self, x):
        return x + self.pos_encoding  # Assumes x is [B, dim, H, W]


class ConvProjectorHead(nn.Module):
    def __init__(self, in_channels=16, out_channels=4):
        super(ConvProjectorHead, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.net(x)



class LatentFusionTransformerModel_Cross(nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        num_modalities: int = 4,
        height:int = 32,
        width: int = 32,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
    ) -> None:
        """
        A transformer decoder that attends to an input image using
        queries whose positional embedding is supplied.

        Args:
          depth (int): number of layers in the transformer
          embedding_dim (int): the channel dimension for the input embeddings
          num_heads (int): the number of heads for multihead attention. Must
            divide embedding_dim
          activation (nn.Module): the activation to use in the MLP block
        """
        super().__init__()
        
        self.depth = depth
        self.embedding_dim = embedding_dim  # 32
        self.num_heads = num_heads  # 4
        self.num_modalities = 4
        self.latent_channels_per_modality = 4
        
        self.height = height
        self.width = width
        
        self.postional_encoding = PositionalEncoding2D(self.embedding_dim, self.height, self.width)  # [32, H, W]
        
        # Modality embedding layer for Input Modalities
        self.input_modality_embedding = nn.Embedding(num_modalities, self.latent_channels_per_modality)  # [4, 4]
        
        # Modality embedding layer for Output Modality
        self.target_modality_embedding = nn.Embedding(num_modalities, self.latent_channels_per_modality)  # [4, 4]
        
        self.norm = nn.LayerNorm(embedding_dim)
        
        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(TwoWayAttentionBlock(embedding_dim, num_heads, attention_downsample_rate=attention_downsample_rate))
        
        self.MLP_for_target_emb = nn.Linear(num_modalities * self.latent_channels_per_modality, 2 * num_modalities * self.latent_channels_per_modality)  # MLP(16,32)
        
        self.mean_projector = ConvProjectorHead()
        self.std_projector = ConvProjectorHead()
        
    def get_input_embedding_map(self, latent_mean, one_hot_input_mod, bs, h, w):
        # Initialize embedding map to add
        input_embedding_map = torch.zeros_like(latent_mean)  # [B, C, H, W]
        
        # Loop through modalities
        for mod_idx in range(self.num_modalities):
            # Mask: [B, 1, 1, 1]
            mask = one_hot_input_mod[:, mod_idx].view(bs, 1, 1, 1)

            # Embedding for this modality: [1, 4]
            emb = self.input_modality_embedding(torch.tensor(mod_idx, device=latent_mean.device)).view(1, self.latent_channels_per_modality, 1, 1)
            
            # Broadcast embedding to match shape [B, 4, H, W]
            emb = emb.expand(bs, -1, h, w)

            # Apply mask
            emb = emb * mask  # Zero out for missing modalities

            # Insert into correct channel slice (e.g., 0–3 for mod 0, 4–7 for mod 1, etc.)
            start = mod_idx * self.latent_channels_per_modality
            end = start + self.latent_channels_per_modality
            input_embedding_map[:, start:end, :, :] = emb
        
        return input_embedding_map  
    
    def get_target_embedding_map(self, one_hot_target_mod, latent_mean, bs, h, w):
        # Step 1: Get the modality index from one-hot
        target_mod_idx = one_hot_target_mod.argmax(dim=1)  # Shape: [B]

        # Step 2: Lookup the embedding vector for each target modality
        target_embeddings = self.target_modality_embedding(target_mod_idx)  # [B, C]

        # Step 3: Expand to spatial dimensions
        target_embeddings = target_embeddings.view(bs, self.latent_channels_per_modality, 1, 1)               # [B, 4, 1, 1]
        target_embeddings = target_embeddings.expand(-1, -1, h, w)          # [B, 4, H, W]

        # Step 4: Initialize a full zero tensor to hold the full 16-channel map
        target_embedding_map = torch.zeros_like(latent_mean)  # [B, 16, H, W]

        # Step 5: Fill in the correct 4-channel slice per sample
        for b in range(bs):
            mod_idx = target_mod_idx[b].item()
            start = mod_idx * self.latent_channels_per_modality
            end = start + self.latent_channels_per_modality
            target_embedding_map[b, start:end, :, :] = target_embeddings[b]
        
        return target_embedding_map
    
    def forward(
        self,
        latent_mean,
        latent_std,
        one_hot_input_mod, 
        one_hot_target_mod,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
          latent_mean (torch.Tensor): latent means of input modalities (B x 16 x 32 x 32)  :: 16==> 4 x 32 x 32 (for 1 modality) ::: total 4 modalities
          latent_std (torch.Tensor): latent std of input modalities (B x 16 x 32 x 32)  :: 16==> 4 x 32 x 32 (for 1 modality) ::: total 4 modalities
          one_hot_input_mod (torch.Tensor): one hot vector denoting the input modalities (B x 4)  :: B x [T1, T1ce, T2, flair]
          one_hot_target_mod (torch.Tensor): one hot vector denoting the output modality (B x 4)  :: B x [T1, T1ce, T2, flair]
          
          C = 16
          H = 32
          W = 32
          H and W depend on the size of the image.
          If original img size is 512 then, H,W = 64,64
          
        Returns:
          torch.Tensor: pred latent mean of target modality 
          torch.Tensor: pred latent std of target modality
        """
        
        bs, c, h, w = latent_mean.shape  # [B, C=16, H, W]
        
        assert self.height == h and self.width == w, 'Model\'s height and width do not match input\'s height and width'
        
        input_embedding_map = self.get_input_embedding_map(latent_mean, one_hot_input_mod, bs, h, w)  # [B, C, H, W]
        target_embedding_map = self.get_target_embedding_map(one_hot_target_mod, latent_mean, bs, h, w)  # [B, C, H, W]
        
        # Add the input embeddings
        latent_mean = latent_mean + input_embedding_map
        latent_std = latent_std + input_embedding_map
        
        # Concatenate along the channel dimension (dim=1)
        latent_combined = torch.cat([latent_mean, latent_std], dim=1)  # [B, 2C, H, W]
        
        # Add positional encoding
        latent_combined = self.postional_encoding(latent_combined)
        
        # Convert combined latent into tokens
        latent_combined = latent_combined.flatten(2).permute(0, 2, 1)  # [B, 2C, H, W] -> [B, H*W, 2C] ==> [B, N_image_tokens=H*W, 2C]
        
        # Convert target embeddings into token and use MLP to expand them
        target_embedding_map = target_embedding_map.flatten(2).permute(0, 2, 1)  # [B, C, H, W] -> [B, H*W, 2C] ==> [B, N_image_tokens=H*W, C]
        target_embedding_map = self.MLP_for_target_emb(target_embedding_map)  # [B, N_image_tokens=H*W, C] ==> [B, N_image_tokens=H*W, 2*C]
        
        target_emb_attn_out = target_embedding_map
        latent_cross_attn_out = latent_combined
        
        # Apply attention blocks
        for layer in self.layers:
            target_emb_attn_out, latent_cross_attn_out = layer(latent_combined=latent_cross_attn_out, target_embedding=target_emb_attn_out)

        latent_combined = latent_combined + latent_cross_attn_out
        
        # Correct the shape and split the mean and std
        latent_combined = latent_combined.permute(0, 2, 1).contiguous()  # [B, 2C, N=H*W]
        latent_combined = latent_combined.view(bs, 2*c, h, w)  # [B, 2C, H, W]
        mean, std = torch.chunk(latent_combined, chunks=2, dim=1)  # [B, C, H, W] each
        
        mean = self.mean_projector(mean)  # [B, 4, H, W]
        std  = self.std_projector(std)  # [B, 4, H, W]
        
        std = torch.sigmoid(std)
        
        return mean, std


class TwoWayAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        """
        A transformer block with four layers: (1) self-attention of sparse
        inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross attention of dense inputs to sparse
        inputs.

        Arguments:
          embedding_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
          skip_first_layer_pe (bool): skip the PE on the first layer
        """
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_targetEmb_to_latent = Attention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = nn.Sequential(nn.Linear(embedding_dim, embedding_dim // 2),
                                 nn.ReLU(),
                                 nn.Linear(embedding_dim // 2, embedding_dim),
                                 nn.ReLU())
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.cross_attn_latent_to_targetEmb = Attention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
        self.norm4 = nn.LayerNorm(embedding_dim)

    def forward(self, latent_combined, target_embedding) -> Tuple[Tensor, Tensor]:
        
        latent_self_attn_out = self.self_attn(q=latent_combined, k=latent_combined, v=latent_combined)
        latent_self_attn_out = latent_combined + latent_self_attn_out
        latent_self_attn_out = self.norm1(latent_self_attn_out)  # [B, N_tokens, 2C]
        
        target_emb_out = self.cross_attn_targetEmb_to_latent(q=target_embedding, k=latent_combined, v=latent_combined)
        target_emb_out = target_emb_out + target_embedding
        target_emb_out = self.norm2(target_emb_out)  # [B, N_tokens, 2C]
        
        target_emb_out = self.mlp(target_emb_out)  # [B, N_tokens, 2C]
        target_emb_out = self.norm3(target_emb_out)
        
        latent_cross_attn_out = self.cross_attn_latent_to_targetEmb(q=latent_self_attn_out, k=target_emb_out, v=target_emb_out)
        latent_cross_attn_out = latent_cross_attn_out + latent_self_attn_out
        latent_cross_attn_out = self.norm4(latent_cross_attn_out)  # [B, N_tokens, 2C]
        
        return target_emb_out, latent_cross_attn_out
        

class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert (
            self.internal_dim % num_heads == 0
        ), "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor, attn_mask=None) -> Tensor:
        """
        Perform Attention

        Args:
            q (Tensor): query
            k (Tensor): key
            v (Tensor): value
            attn_mask (Tensor, optional): Attention mask. Defaults to None. Shape: [B, N_Tokens]

        Returns:
            Tensor: Attention output
        """
        
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        
        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask == 0, -1e9)
        
        # Apply softmax
        attn = torch.softmax(attn, dim=-1)
        
        '''
        if attn.shape[3] == 4096:
            fig, axes = plt.subplots(8, 3, figsize=(15, 15))
            for idx, ax in enumerate(axes.flat):
                tmp = copy.copy(attn[0,idx//8,idx%3,:].cpu())
                tmp = tmp.view(64,64)
                tmp = tmp.detach().cpu().numpy()
                im = ax.imshow(tmp, cmap='magma')
            plt.title('Attention map prompts on image: 8 x 3 x 64 x 64')
            plt.show()
        '''
        
        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out