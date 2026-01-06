#!/usr/bin/env python3
"""
Unified inference script for both U-Net and Diffusion Transformer models
Automatically detects model architecture from checkpoint configuration
"""

import torch
import torch.nn.functional as F
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
import argparse
import json
from typing import Optional, Tuple, List, Union
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.diffusion_model import MusicDiffusionModel
from models.diffusion_transformer import MusicDiffusionTransformer
from models.unet import MelSpectrogramUNet


def create_mel_spectrogram(audio_chunk, sr, n_mels=128):
    """Create mel-spectrogram from audio"""
    mel_spec = librosa.feature.melspectrogram(y=audio_chunk, sr=sr, n_mels=n_mels)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    normalized_spec = (log_mel_spec + 40) / 40
    return normalized_spec


def denormalize_mel_spectrogram(normalized_spec):
    """Reverse normalization"""
    log_mel_spec = normalized_spec * 40 - 40
    return log_mel_spec


def mel_to_audio_notebook_style(mel_spec_normalized, sr=22050, duration_sec=5):
    """Convert mel-spectrogram to audio"""
    log_mel_spec = denormalize_mel_spectrogram(mel_spec_normalized)
    mel_spec = librosa.db_to_power(log_mel_spec, ref=1.0)
    
    audio = librosa.feature.inverse.mel_to_audio(
        mel_spec, 
        sr=sr, 
        n_fft=2048,
        hop_length=512,
        n_iter=32,
        length=int(duration_sec * sr)
    )
    
    return audio


class UnifiedMusicGenerator:
    """Unified music generator supporting both U-Net and Transformer architectures"""
    
    def __init__(self,
                 model_path: str,
                 device: str = 'auto',
                 force_architecture: Optional[str] = None):
        
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        self.model, self.architecture = self.load_model(model_path, force_architecture)
        self.model.eval()
        
        print(f"Architecture: {self.architecture}")
        
        self.sample_rate = 22050
        self.n_mels = 128
        self.n_fft = 2048
        self.hop_length = 512
        self.duration = 5.0
        
        print("Model loaded successfully!")
    
    def load_model(self, model_path: str, force_architecture: Optional[str] = None):
        """Load model from checkpoint (auto-detects architecture)"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Get config
        config = {}
        if 'model_config' in checkpoint:
            config = checkpoint['model_config']
            print("✓ Model configuration found")
        else:
            print("⚠ Model configuration not found, inferring architecture...")
        
        # Determine architecture
        if force_architecture:
            architecture = force_architecture
            print(f"Using forced architecture: {architecture}")
        else:
            architecture = config.get('architecture', 'unet')
        
        # Create and load model
        if architecture == 'diffusion_transformer':
            print("Loading Diffusion Transformer model...")
            model = self._load_transformer(checkpoint, config)
        else:
            print("Loading U-Net model...")
            model = self._load_unet(checkpoint, config)
        
        return model, architecture
    
    def _load_transformer(self, checkpoint, config):
        """Load Diffusion Transformer model"""
        model = MusicDiffusionTransformer(
            n_mels=config.get('n_mels', 128),
            n_frames=config.get('n_frames', 216),
            patch_size=config.get('patch_size', 8),
            embed_dim=config.get('embed_dim', 256),
            num_layers=config.get('num_layers', 12),
            num_heads=config.get('num_heads', 8),
            mlp_dim=config.get('mlp_dim', 1024),
            timesteps=config.get('timesteps', 1000),
            schedule_type=config.get('schedule_type', 'cosine')
        ).to(self.device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        return model
    
    def _load_unet(self, checkpoint, config):
        """Load U-Net model"""
        from models.unet import MelSpectrogramUNet
        
        unet = MelSpectrogramUNet(
            input_channels=config.get('input_channels', 1),
            base_channels=config.get('base_channels', 64),
            n_mels=config.get('n_mels', 128)
        ).to(self.device)
        
        model = MusicDiffusionModel(
            unet=unet,
            timesteps=config.get('timesteps', 1000),
            schedule_type=config.get('schedule_type', 'cosine')
        ).to(self.device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Legacy format - direct checkpoint load
            model.load_state_dict(checkpoint)
        
        return model
    
    @torch.no_grad()
    def generate(self,
                 batch_size: int = 1,
                 num_inference_steps: int = 50,
                 eta: float = 0.0,
                 seed: Optional[int] = None,
                 return_intermediates: bool = False) -> torch.Tensor:
        """Generate mel-spectrograms"""
        if seed is not None:
            torch.manual_seed(seed)
        
        if self.architecture == 'diffusion_transformer':
            shape = (batch_size, 1, self.n_mels, 216)
        else:  # unet
            shape = (batch_size, 1, self.n_mels, 216)
        
        spectrograms = self.model.sample(
            shape=shape,
            device=self.device,
            num_inference_steps=num_inference_steps,
            eta=eta,
            return_intermediates=return_intermediates
        )
        
        return spectrograms
    
    def mel_to_waveform(self, mel_spec: torch.Tensor) -> np.ndarray:
        """Convert mel-spectrogram to waveform"""
        if mel_spec.dim() == 4:
            mel_spec = mel_spec.squeeze(0)
        if mel_spec.dim() == 3:
            mel_spec = mel_spec.squeeze(0)
        
        mel_np = mel_spec.cpu().numpy()
        audio = mel_to_audio_notebook_style(mel_np, sr=self.sample_rate, duration_sec=self.duration)
        
        return audio
    
    def generate_music(self,
                      output_path: str,
                      num_samples: int = 1,
                      num_inference_steps: int = 50,
                      eta: float = 0.0,
                      seed: Optional[int] = None,
                      save_spectrograms: bool = True) -> List[str]:
        """Generate and save music"""
        print(f"Generating {num_samples} samples ({self.architecture})...")
        
        output_dir = Path(output_path)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        generated_files = []
        
        for i in range(num_samples):
            print(f"Sample {i+1}/{num_samples}")
            
            if seed is not None:
                sample_seed = seed + i
            else:
                sample_seed = None
            
            spectrograms = self.generate(
                batch_size=1,
                num_inference_steps=num_inference_steps,
                eta=eta,
                seed=sample_seed
            )
            
            mel_spec = spectrograms[0]
            
            # Convert to audio
            audio = self.mel_to_waveform(mel_spec)
            audio = audio / np.max(np.abs(audio)) * 0.8
            
            # Save audio
            audio_path = output_dir / f"generated_music_{i:03d}.wav"
            sf.write(audio_path, audio, self.sample_rate)
            generated_files.append(str(audio_path))
            
            # Save spectrogram
            if save_spectrograms:
                spec_path = output_dir / f"spectrogram_{i:03d}.pt"
                torch.save(mel_spec.cpu(), spec_path)
                
                self.plot_spectrogram(mel_spec, output_dir / f"spectrogram_{i:03d}.png")
        
        print(f"Generation complete! Files saved to {output_dir}")
        return generated_files
    
    def plot_spectrogram(self, mel_spec: torch.Tensor, save_path: Optional[str] = None):
        """Visualize mel-spectrogram"""
        spec_np = mel_spec.squeeze().cpu().numpy()
        
        plt.figure(figsize=(12, 6))
        librosa.display.specshow(
            spec_np,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            x_axis='time',
            y_axis='mel',
            cmap='viridis'
        )
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Generated Mel Spectrogram ({self.architecture.title()})')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def model_info(self) -> dict:
        """Get model information"""
        info = {
            'architecture': self.architecture,
            'device': str(self.device),
            'parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }
        return info


def main():
    parser = argparse.ArgumentParser(
        description='Generate music using U-Net or Diffusion Transformer'
    )
    
    # Model
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model checkpoint')
    
    # Architecture override
    parser.add_argument('--architecture', type=str, default=None, 
                       choices=['unet', 'diffusion_transformer'],
                       help='Force architecture (auto-detect if not specified)')
    
    # Generation
    parser.add_argument('--output_dir', type=str, default='generated_music',
                       help='Output directory')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of samples to generate')
    parser.add_argument('--num_inference_steps', type=int, default=50,
                       help='Number of inference steps')
    parser.add_argument('--eta', type=float, default=0.0,
                       help='DDIM parameter')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed')
    
    # Options
    parser.add_argument('--save_spectrograms', action='store_true',
                       help='Save spectrograms')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (auto, cuda, cpu)')
    parser.add_argument('--info', action='store_true',
                       help='Show model info and exit')
    
    args = parser.parse_args()
    
    # Verify model exists
    if not Path(args.model_path).exists():
        raise FileNotFoundError(f"Model not found: {args.model_path}")
    
    # Create generator
    generator = UnifiedMusicGenerator(
        model_path=args.model_path,
        device=args.device,
        force_architecture=args.architecture
    )
    
    # Show model info if requested
    if args.info:
        info = generator.model_info()
        print("\nModel Information:")
        print(f"  Architecture: {info['architecture']}")
        print(f"  Device: {info['device']}")
        print(f"  Parameters: {info['parameters']:,}")
        print(f"  Trainable: {info['trainable_parameters']:,}")
        return
    
    # Generate music
    generated_files = generator.generate_music(
        output_path=args.output_dir,
        num_samples=args.num_samples,
        num_inference_steps=args.num_inference_steps,
        eta=args.eta,
        seed=args.seed,
        save_spectrograms=args.save_spectrograms
    )
    
    print(f"\nGenerated files:")
    for file_path in generated_files:
        print(f"  {file_path}")


if __name__ == "__main__":
    main()
