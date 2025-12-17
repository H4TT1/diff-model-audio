import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from unet import MelSpectrogramUNet

class NoiseScheduler:
    """Scheduleur de bruit pour le processus de diffusion"""
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=2e-2, schedule_type="linear"):
        self.timesteps = timesteps
        
        if schedule_type == "linear":
            self.betas = torch.linspace(beta_start, beta_end, timesteps)
        elif schedule_type == "cosine":
            # Scheduleur cosinus pour une meilleure qualité
            s = 0.008
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps)
            alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clip(betas, 0.0001, 0.9999)
        
        # Précalcul des valeurs utiles
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Pour le reparametrization trick
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Pour le reverse process
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def add_noise(self, x_start, t, noise=None):
        """Ajoute du bruit gaussien selon le timestep t (forward process)"""
        if noise is None:
            noise = torch.randn_like(x_start)
            
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def _extract(self, arr, timesteps, broadcast_shape):
        """Extrait les valeurs pour les timesteps donnés"""
        batch_size = timesteps.shape[0]
        out = arr.gather(-1, timesteps)
        return out.reshape(batch_size, *((1,) * (len(broadcast_shape) - 1)))

class MusicDiffusionModel(nn.Module):
    """Modèle de diffusion pour la génération de musique via mel-spectrogrammes"""
    def __init__(self, 
                 input_channels=1,
                 base_channels=64,
                 timesteps=1000,
                 schedule_type="cosine"):
        super().__init__()
        
        self.unet = MelSpectrogramUNet(
            input_channels=input_channels,
            output_channels=input_channels,
            base_channels=base_channels
        )
        
        self.noise_scheduler = NoiseScheduler(
            timesteps=timesteps,
            schedule_type=schedule_type
        )
        
        self.timesteps = timesteps
        
    def forward(self, x, noise=None):
        """
        Forward pass pour l'entraînement
        x: mel-spectrogramme original [batch, channels, freq, time]
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Sample random timesteps
        t = torch.randint(0, self.timesteps, (batch_size,), device=device, dtype=torch.long)
        
        if noise is None:
            noise = torch.randn_like(x)
            
        # Add noise selon le timestep
        x_noisy = self.noise_scheduler.add_noise(x, t, noise)
        
        # Prédire le bruit
        noise_pred = self.unet(x_noisy, t)
        
        return noise_pred, noise, t
    
    def compute_loss(self, x, reduction="mean"):
        """Calcule la loss de débruitage"""
        noise_pred, noise_target, t = self.forward(x)
        
        # Mean Squared Error loss
        loss = F.mse_loss(noise_pred, noise_target, reduction='none')
        
        # Pondération optionnelle par timestep (P2 weighting)
        # Plus de poids sur les timesteps difficiles
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
        Génère de nouveaux mel-spectrogrammes (reverse process)
        
        shape: (batch_size, channels, freq_bins, time_steps)
               Format attendu: (batch, 1, 128, 216) correspondant au preprocessing
        num_inference_steps: nombre de pas de débruitage
        eta: paramètre pour DDIM (0 = déterministe, 1 = DDPM)
        """
        batch_size, channels, height, width = shape
        
        # Commence par du bruit pur
        img = torch.randn(shape, device=device)
        
        # Timesteps pour l'inférence (peut être moins que l'entraînement)
        timesteps = torch.linspace(self.timesteps-1, 0, num_inference_steps, dtype=torch.long, device=device)
        
        intermediates = []
        
        for i, t in enumerate(tqdm(timesteps, desc="Sampling")):
            t_batch = t.repeat(batch_size)
            
            # Prédiction du bruit
            noise_pred = self.unet(img, t_batch)
            
            # DDIM sampling
            alpha_prod_t = self.noise_scheduler.alphas_cumprod[t]
            alpha_prod_t_prev = self.noise_scheduler.alphas_cumprod[max(0, t - self.timesteps // num_inference_steps)]
            
            beta_prod_t = 1 - alpha_prod_t
            beta_prod_t_prev = 1 - alpha_prod_t_prev
            
            # Prédiction de x_0
            pred_original_sample = (img - beta_prod_t ** 0.5 * noise_pred) / alpha_prod_t ** 0.5
            
            # Direction pointant vers x_t
            pred_sample_direction = (1 - alpha_prod_t_prev - eta**2 * beta_prod_t_prev) ** 0.5 * noise_pred
            
            # Sample précédent
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
    
    @torch.no_grad()
    def interpolate(self, x1, x2, num_steps=10, t_start=500):
        """
        Interpole entre deux mel-spectrogrammes dans l'espace latent
        """
        device = x1.device
        batch_size = x1.shape[0]
        
        # Encode vers l'espace de bruit
        t = torch.full((batch_size,), t_start, device=device, dtype=torch.long)
        noise1 = torch.randn_like(x1)
        noise2 = torch.randn_like(x2)
        
        x1_noisy = self.noise_scheduler.add_noise(x1, t, noise1)
        x2_noisy = self.noise_scheduler.add_noise(x2, t, noise2)
        
        results = []
        for alpha in torch.linspace(0, 1, num_steps):
            # Interpolation linéaire
            x_interp = alpha * x2_noisy + (1 - alpha) * x1_noisy
            
            # Débruitage
            denoised = self.sample(x_interp.shape, device, num_inference_steps=t_start//10)
            results.append(denoised)
            
        return torch.stack(results)

def test_diffusion_model():
    """Test du modèle de diffusion"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = MusicDiffusionModel(
        input_channels=1,
        base_channels=64,
        timesteps=1000,
        schedule_type="cosine"
    ).to(device)
    
    # Test forward pass avec les dimensions du preprocessing
    batch_size = 2
    x = torch.randn(batch_size, 1, 128, 216).to(device)  # Format du preprocessing
    
    loss = model.compute_loss(x)
    print(f"Training loss: {loss.item():.4f}")
    
    # Test sampling avec les bonnes dimensions
    with torch.no_grad():
        samples = model.sample(shape=(1, 1, 128, 216), device=device, num_inference_steps=20)
        print(f"Generated sample shape: {samples.shape}")
    
    print(f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    return model

if __name__ == "__main__":
    model = test_diffusion_model()