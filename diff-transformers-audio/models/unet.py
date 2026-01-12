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

class AttentionBlock(nn.Module):
    """Bloc d'attention pour améliorer la qualité des spectrogrammes"""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        
    def forward(self, x):
        b, c, h, w = x.shape
        residual = x
        
        x = self.norm(x)
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)
        
        # Reshape pour l'attention
        q = q.view(b, c, h * w).transpose(1, 2)
        k = k.view(b, c, h * w).transpose(1, 2)
        v = v.view(b, c, h * w).transpose(1, 2)
        
        # Calcul de l'attention
        scale = (c ** -0.5)
        attention = torch.softmax(torch.bmm(q, k.transpose(1, 2)) * scale, dim=-1)
        out = torch.bmm(attention, v)
        
        out = out.transpose(1, 2).view(b, c, h, w)
        out = self.proj(out)
        
        return out + residual

class ResidualBlock(nn.Module):
    """Bloc résiduel avec injection de timestep"""
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )
        
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )
        
        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x, time_emb):
        residual = self.residual_conv(x)
        
        x = self.block1(x)
        
        # Injection du timestep embedding
        time_emb = self.time_mlp(time_emb)[:, :, None, None]
        x = x + time_emb
        
        x = self.block2(x)
        
        return x + residual

class DownBlock(nn.Module):
    """Bloc de descente dans l'encoder"""
    def __init__(self, in_channels, out_channels, time_emb_dim, use_attention=False):
        super().__init__()
        self.res_block1 = ResidualBlock(in_channels, out_channels, time_emb_dim)
        self.res_block2 = ResidualBlock(out_channels, out_channels, time_emb_dim)
        self.attention = AttentionBlock(out_channels) if use_attention else None
        self.downsample = nn.Conv2d(out_channels, out_channels, 4, 2, 1)
        
    def forward(self, x, time_emb):
        x = self.res_block1(x, time_emb)
        x = self.res_block2(x, time_emb)
        
        if self.attention:
            x = self.attention(x)
            
        skip = x
        x = self.downsample(x)
        
        return x, skip

class UpBlock(nn.Module):
    """Bloc de montée dans le decoder"""
    def __init__(self, in_channels, out_channels, time_emb_dim, use_attention=False):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, 4, 2, 1)
        self.res_block1 = ResidualBlock(in_channels + out_channels, out_channels, time_emb_dim)
        self.res_block2 = ResidualBlock(out_channels, out_channels, time_emb_dim)
        self.attention = AttentionBlock(out_channels) if use_attention else None
        
    def forward(self, x, skip, time_emb):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.res_block1(x, time_emb)
        x = self.res_block2(x, time_emb)
        
        if self.attention:
            x = self.attention(x)
            
        return x

class MelSpectrogramUNet(nn.Module):
    """
    U-Net optimisé pour les mel-spectrogrammes
    Architecture adaptée aux dimensions du preprocessing: (1, 128, 216) 
    - 128 bins mel x 216 frames temporelles (5 secondes à 22050 Hz)
    """
    def __init__(self, 
                 input_channels=1,  # Spectrogramme mono
                 output_channels=1, 
                 base_channels=64,
                 time_emb_dim=256,
                 num_res_blocks=2):
        super().__init__()
        
        self.time_emb_dim = time_emb_dim
        
        # Embedding temporel pour les timesteps de diffusion
        self.time_mlp = nn.Sequential(
            SinusoidalPositionalEmbedding(time_emb_dim // 4),
            nn.Linear(time_emb_dim // 4, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # Convolution initiale
        self.init_conv = nn.Conv2d(input_channels, base_channels, 3, padding=1)
        
        # Encoder (downsampling)
        self.down1 = DownBlock(base_channels, base_channels * 2, time_emb_dim)
        self.down2 = DownBlock(base_channels * 2, base_channels * 4, time_emb_dim)
        self.down3 = DownBlock(base_channels * 4, base_channels * 8, time_emb_dim, use_attention=True)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResidualBlock(base_channels * 8, base_channels * 8, time_emb_dim),
            AttentionBlock(base_channels * 8),
            ResidualBlock(base_channels * 8, base_channels * 8, time_emb_dim)
        )
        
        # Decoder (upsampling)
        self.up3 = UpBlock(base_channels * 8, base_channels * 4, time_emb_dim, use_attention=True)
        self.up2 = UpBlock(base_channels * 4, base_channels * 2, time_emb_dim)
        self.up1 = UpBlock(base_channels * 2, base_channels, time_emb_dim)
        
        # Convolution finale
        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, output_channels, 3, padding=1)
        )
        
    def forward(self, x, timestep):
        """
        x: mel-spectrogramme bruité [batch, channels, freq_bins, time] 
           Format attendu: [batch, 1, 128, 216] (du preprocessing)
        timestep: pas de temps dans le processus de diffusion [batch]
        """
        # Embedding temporel
        time_emb = self.time_mlp(timestep)
        
        # Convolution initiale
        x = self.init_conv(x)
        
        # Encoder avec skip connections
        x1, skip1 = self.down1(x, time_emb)
        x2, skip2 = self.down2(x1, time_emb)
        x3, skip3 = self.down3(x2, time_emb)
        
        # Bottleneck
        for layer in self.bottleneck:
            if isinstance(layer, ResidualBlock):
                x3 = layer(x3, time_emb)
            else:
                x3 = layer(x3)
        
        # Decoder avec skip connections
        x = self.up3(x3, skip3, time_emb)
        x = self.up2(x, skip2, time_emb)
        x = self.up1(x, skip1, time_emb)
        
        # Sortie finale
        x = self.final_conv(x)
        
        return x

def test_unet():
    """Test rapide de l'architecture"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Paramètres correspondant au preprocessing du notebook
    batch_size = 2
    channels = 1
    freq_bins = 128  # bins mel (du preprocessing)
    time_steps = 216  # frames temporelles (calculé du preprocessing 5 sec)
    
    model = MelSpectrogramUNet().to(device)
    x = torch.randn(batch_size, channels, freq_bins, time_steps).to(device)
    timestep = torch.randint(0, 1000, (batch_size,)).to(device)
    
    with torch.no_grad():
        output = model(x, timestep)
        
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    return model

if __name__ == "__main__":
    model = test_unet()
