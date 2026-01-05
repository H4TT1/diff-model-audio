import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPositionalEmbedding(nn.Module):
    """Positional encoding pour le timestep dans le processus de diffusion"""
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

class SimpleResidualBlock(nn.Module):
    """Bloc résiduel simplifié avec injection de timestep"""
    def __init__(self, channels, time_emb_dim):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, channels)
        )
        
        self.block = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1)
        )
        
    def forward(self, x, time_emb):
        # Injection du timestep embedding
        time_emb = self.time_mlp(time_emb)[:, :, None, None]
        
        return x + self.block(x) + time_emb

class SimpleMelUNet(nn.Module):
    """
    U-Net simplifié pour les mel-spectrogrammes (1, 128, 216)
    Version plus robuste avec moins de problèmes de dimensions
    """
    def __init__(self, 
                 input_channels=1,
                 output_channels=1, 
                 base_channels=32,
                 time_emb_dim=128):
        super().__init__()
        
        self.time_emb_dim = time_emb_dim
        
        # Embedding temporel
        self.time_mlp = nn.Sequential(
            SinusoidalPositionalEmbedding(time_emb_dim // 4),
            nn.Linear(time_emb_dim // 4, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # Convolution initiale
        self.init_conv = nn.Conv2d(input_channels, base_channels, 3, padding=1)
        
        # Encoder
        self.down1 = nn.Sequential(
            SimpleResidualBlock(base_channels, time_emb_dim),
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1)  # 128x216 -> 64x108
        )
        
        self.down2 = nn.Sequential(
            SimpleResidualBlock(base_channels * 2, time_emb_dim),
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1)  # 64x108 -> 32x54
        )
        
        self.down3 = nn.Sequential(
            SimpleResidualBlock(base_channels * 4, time_emb_dim),
            nn.Conv2d(base_channels * 4, base_channels * 8, 4, 2, 1)  # 32x54 -> 16x27
        )
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            SimpleResidualBlock(base_channels * 8, time_emb_dim),
            SimpleResidualBlock(base_channels * 8, time_emb_dim)
        )
        
        # Decoder
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 4, 2, 1),  # 16x27 -> 32x54
            SimpleResidualBlock(base_channels * 4, time_emb_dim)
        )
        
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, 2, 1),  # 32x54 -> 64x108
            SimpleResidualBlock(base_channels * 2, time_emb_dim)
        )
        
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, 2, 1),  # 64x108 -> 128x216
            SimpleResidualBlock(base_channels, time_emb_dim)
        )
        
        # Convolution finale
        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, output_channels, 3, padding=1)
        )
        
    def forward(self, x, timestep):
        """
        x: mel-spectrogramme bruité [batch, 1, 128, 216]
        timestep: pas de temps dans le processus de diffusion [batch]
        """
        # Embedding temporel
        time_emb = self.time_mlp(timestep)
        
        # Initial conv
        x = self.init_conv(x)
        
        # Encoder (sans skip connections pour simplifier)
        x = self.down1[0](x, time_emb)  # ResBlock
        x = self.down1[1](x)            # Downsample
        
        x = self.down2[0](x, time_emb)
        x = self.down2[1](x)
        
        x = self.down3[0](x, time_emb)
        x = self.down3[1](x)
        
        # Bottleneck
        x = self.bottleneck[0](x, time_emb)
        x = self.bottleneck[1](x, time_emb)
        
        # Decoder
        x = self.up3[0](x)              # Upsample
        x = self.up3[1](x, time_emb)    # ResBlock
        
        x = self.up2[0](x)
        x = self.up2[1](x, time_emb)
        
        x = self.up1[0](x)
        x = self.up1[1](x, time_emb)
        
        # Final conv
        x = self.final_conv(x)
        
        return x

def test_simple_unet():
    """Test de l'architecture simplifiée"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test avec les dimensions du preprocessing
    batch_size = 2
    x = torch.randn(batch_size, 1, 128, 216).to(device)
    timestep = torch.randint(0, 1000, (batch_size,)).to(device)
    
    model = SimpleMelUNet(base_channels=32).to(device)
    
    with torch.no_grad():
        output = model(x, timestep)
        
    print(f"✅ Input shape: {x.shape}")
    print(f"✅ Output shape: {output.shape}")
    print(f"✅ Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    return model

if __name__ == "__main__":
    model = test_simple_unet()