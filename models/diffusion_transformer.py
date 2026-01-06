"""
Diffusion Transformer for Music Generation
Alternative to U-Net: Uses pure transformer architecture for diffusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm
import numpy as np


class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal positional encoding for timesteps"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class PatchEmbedding(nn.Module):
    """Convert mel-spectrogram to patches and embed them"""
    def __init__(self, patch_size=8, embed_dim=256, n_mels=128, n_frames=216):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.n_mels = n_mels
        self.n_frames = n_frames
        
        # Calculate number of patches
        self.n_mel_patches = n_mels // patch_size
        self.n_frame_patches = n_frames // patch_size
        self.num_patches = self.n_mel_patches * self.n_frame_patches
        
        # Patch embedding
        self.patch_embed = nn.Linear(patch_size * patch_size, embed_dim)
        
        # Positional embeddings for patches
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim) * 0.02)
        
    def forward(self, x):
        """
        x: [batch, 1, n_mels, n_frames]
        Output: [batch, num_patches, embed_dim]
        """
        batch_size = x.shape[0]
        
        # Reshape to patches
        x = x.squeeze(1)  # [batch, n_mels, n_frames]
        
        # Extract patches
        patches = []
        for i in range(0, self.n_mels, self.patch_size):
            for j in range(0, self.n_frames, self.patch_size):
                patch = x[:, i:i+self.patch_size, j:j+self.patch_size]
                patch = patch.reshape(batch_size, -1)
                patches.append(patch)
        
        patches = torch.stack(patches, dim=1)  # [batch, num_patches, patch_size^2]
        
        # Embed patches
        embeddings = self.patch_embed(patches)  # [batch, num_patches, embed_dim]
        
        # Add positional embeddings
        embeddings = embeddings + self.pos_embed
        
        return embeddings


class TransformerBlock(nn.Module):
    """Transformer block with time conditioning"""
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1, time_embed_dim=256):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Self-attention
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        
        # Time conditioning
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Feed-forward
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, time_embed):
        """
        x: [batch, num_patches, embed_dim]
        time_embed: [batch, time_embed_dim]
        """
        # Self-attention with residual
        x_norm = self.norm1(x)
        attn_out, _ = self.mha(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # Time conditioning
        time_feature = self.time_mlp(time_embed)  # [batch, embed_dim]
        time_feature = time_feature.unsqueeze(1)  # [batch, 1, embed_dim]
        x = x + time_feature
        
        # Feed-forward with residual
        x_norm = self.norm2(x)
        mlp_out = self.mlp(x_norm)
        x = x + mlp_out
        
        return x


class MelSpectrogramDiT(nn.Module):
    """
    Diffusion Transformer for Mel-Spectrogram Generation
    Uses pure transformer architecture instead of U-Net
    """
    def __init__(self, 
                 n_mels=128,
                 n_frames=216,
                 patch_size=8,
                 embed_dim=256,
                 num_layers=12,
                 num_heads=8,
                 mlp_dim=1024,
                 time_embed_dim=256,
                 dropout=0.1):
        super().__init__()
        
        self.n_mels = n_mels
        self.n_frames = n_frames
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(patch_size, embed_dim, n_mels, n_frames)
        
        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionalEmbedding(time_embed_dim // 4),
            nn.Linear(time_embed_dim // 4, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        
        # Transformer layers
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_dim, dropout, time_embed_dim)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.norm_out = nn.LayerNorm(embed_dim)
        self.out_proj = nn.Linear(embed_dim, patch_size * patch_size)
        
    def forward(self, x, timestep):
        """
        x: noisy mel-spectrogram [batch, 1, n_mels, n_frames]
        timestep: diffusion timestep [batch]
        Output: noise prediction [batch, 1, n_mels, n_frames]
        """
        batch_size = x.shape[0]
        
        # Patch embedding
        x_embed = self.patch_embed(x)  # [batch, num_patches, embed_dim]
        
        # Time embedding
        time_embed = self.time_embed(timestep)  # [batch, time_embed_dim]
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x_embed = block(x_embed, time_embed)
        
        # Output projection
        x_out = self.norm_out(x_embed)  # [batch, num_patches, embed_dim]
        x_out = self.out_proj(x_out)  # [batch, num_patches, patch_size^2]
        
        # Reshape back to mel-spectrogram
        n_mel_patches = self.n_mels // self.patch_size
        n_frame_patches = self.n_frames // self.patch_size
        
        output = torch.zeros(batch_size, 1, self.n_mels, self.n_frames, 
                            device=x.device, dtype=x.dtype)
        
        patch_idx = 0
        for i in range(n_mel_patches):
            for j in range(n_frame_patches):
                patch = x_out[:, patch_idx, :]  # [batch, patch_size^2]
                patch = patch.reshape(batch_size, self.patch_size, self.patch_size)
                output[:, 0, 
                      i*self.patch_size:(i+1)*self.patch_size,
                      j*self.patch_size:(j+1)*self.patch_size] = patch
                patch_idx += 1
        
        return output


class NoiseScheduler:
    """Noise scheduler for diffusion process"""
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=2e-2, 
                 schedule_type="cosine", device="cpu"):
        self.timesteps = timesteps
        self.device = device
        
        if schedule_type == "linear":
            self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        elif schedule_type == "cosine":
            s = 0.008
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps, device=device)
            alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clip(betas, 0.0001, 0.9999)
        
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
    
    def to(self, device):
        """Move all scheduler tensors to device"""
        self.device = device
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.sqrt_recip_alphas = self.sqrt_recip_alphas.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        return self

    def add_noise(self, x_start, t, noise=None):
        """Add Gaussian noise according to timestep t"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def _extract(self, arr, timesteps, broadcast_shape):
        """Extract values for given timesteps"""
        batch_size = timesteps.shape[0]
        out = arr.gather(-1, timesteps)
        return out.reshape(batch_size, *((1,) * (len(broadcast_shape) - 1)))


class MusicDiffusionTransformer(nn.Module):
    """Diffusion Transformer model for music generation"""
    def __init__(self,
                 n_mels=128,
                 n_frames=216,
                 patch_size=8,
                 embed_dim=256,
                 num_layers=12,
                 num_heads=8,
                 mlp_dim=1024,
                 timesteps=1000,
                 schedule_type="cosine"):
        super().__init__()
        
        self.dit = MelSpectrogramDiT(
            n_mels=n_mels,
            n_frames=n_frames,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            time_embed_dim=256
        )
        
        self.noise_scheduler = NoiseScheduler(
            timesteps=timesteps,
            schedule_type=schedule_type,
            device='cpu'
        )
        
        self.timesteps = timesteps
    
    def to(self, device):
        """Override to() to move scheduler tensors to device"""
        super().to(device)
        self.noise_scheduler = self.noise_scheduler.to(device)
        return self
    
    def forward(self, x, noise=None):
        """
        Forward pass for training
        x: mel-spectrogram [batch, 1, n_mels, n_frames]
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Sample random timesteps
        t = torch.randint(0, self.timesteps, (batch_size,), device=device, dtype=torch.long)
        
        if noise is None:
            noise = torch.randn_like(x)
        
        # Add noise
        x_noisy = self.noise_scheduler.add_noise(x, t, noise)
        
        # Predict noise
        noise_pred = self.dit(x_noisy, t)
        
        return noise_pred, noise, t
    
    def compute_loss(self, x, reduction="mean"):
        """Compute denoising loss"""
        noise_pred, noise_target, t = self.forward(x)
        
        # MSE loss
        loss = F.mse_loss(noise_pred, noise_target, reduction='none')
        
        # Optional: weight by timestep
        if hasattr(self, 'loss_weighting') and self.loss_weighting:
            weights = 1 / (1 + t.float())
            weights = weights.view(-1, 1, 1, 1)
            loss = loss * weights
        
        if reduction == "mean":
            return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        else:
            return loss
    
    @torch.no_grad()
    def sample(self, shape, device, num_inference_steps=50, eta=0.0, return_intermediates=False):
        """
        Generate mel-spectrograms using DDIM sampling
        shape: (batch_size, 1, n_mels, n_frames)
        """
        batch_size = shape[0]
        img = torch.randn(shape, device=device)
        
        # Timesteps for inference
        timesteps = torch.linspace(self.timesteps-1, 0, num_inference_steps, dtype=torch.long, device=device)
        
        intermediates = []
        
        for i, t in enumerate(tqdm(timesteps, desc="Sampling")):
            t_batch = t.repeat(batch_size)
            
            # Predict noise
            noise_pred = self.dit(img, t_batch)
            
            # DDIM sampling
            alpha_prod_t = self.noise_scheduler.alphas_cumprod[t]
            alpha_prod_t_prev = self.noise_scheduler.alphas_cumprod[max(0, t - self.timesteps // num_inference_steps)]
            
            beta_prod_t = 1 - alpha_prod_t
            beta_prod_t_prev = 1 - alpha_prod_t_prev
            
            # Predict x_0
            pred_original_sample = (img - beta_prod_t ** 0.5 * noise_pred) / alpha_prod_t ** 0.5
            
            # Direction towards x_t
            pred_sample_direction = (1 - alpha_prod_t_prev - eta**2 * beta_prod_t_prev) ** 0.5 * noise_pred
            
            # Previous sample
            prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
            
            if eta > 0:
                noise = torch.randn_like(img)
                variance = eta**2 * beta_prod_t_prev
                prev_sample = prev_sample + variance ** 0.5 * noise
            
            img = prev_sample
            
            if return_intermediates:
                intermediates.append(img.cpu())
        
        if return_intermediates:
            return img, intermediates
        return img


def test_diffusion_transformer():
    """Quick test of the diffusion transformer"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = MusicDiffusionTransformer(
        n_mels=128,
        n_frames=216,
        patch_size=8,
        embed_dim=256,
        num_layers=4,  # Smaller for testing
        num_heads=8,
        mlp_dim=1024,
        timesteps=1000,
        schedule_type="cosine"
    ).to(device)
    
    # Test forward pass
    batch_size = 2
    x = torch.randn(batch_size, 1, 128, 216).to(device)
    
    loss = model.compute_loss(x)
    print(f"Training loss: {loss.item():.4f}")
    
    # Test sampling
    with torch.no_grad():
        samples = model.sample(shape=(1, 1, 128, 216), device=device, num_inference_steps=20)
        print(f"Generated sample shape: {samples.shape}")
    
    print(f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    return model


if __name__ == "__main__":
    model = test_diffusion_transformer()
