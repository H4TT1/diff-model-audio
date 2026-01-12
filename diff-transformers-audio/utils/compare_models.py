#!/usr/bin/env python3
"""
Model comparison utility
Compare training performance, inference speed, and quality between U-Net and Transformer
"""

import torch
import torch.nn as nn
import time
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, Tuple, Optional
import argparse
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from models.diffusion_model import MusicDiffusionModel
    from models.diffusion_transformer import MusicDiffusionTransformer
    from models.unet import MelSpectrogramUNet
except ImportError as e:
    print(f"Error importing models: {e}")
    print("Make sure all model files are in the same directory")
    exit(1)


class ModelComparator:
    """Compare U-Net and Diffusion Transformer models"""
    
    def __init__(self, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
    
    def count_parameters(self, model: nn.Module) -> Tuple[int, int]:
        """Count total and trainable parameters"""
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total, trainable
    
    def estimate_memory(self, model: nn.Module, batch_size: int = 1) -> Dict[str, float]:
        """Estimate GPU memory usage"""
        # Model parameters
        total_params, _ = self.count_parameters(model)
        param_memory = total_params * 4 / (1024**3)  # 4 bytes per float32
        
        # Activation memory (rough estimate: 2x model size)
        activation_memory = param_memory * 2
        
        # Batch memory
        batch_memory = batch_size * 1 * 128 * 216 * 4 / (1024**3)  # Input
        batch_memory += batch_size * 1 * 128 * 216 * 4 / (1024**3)  # Output
        
        return {
            'model': param_memory,
            'activations': activation_memory,
            'batch': batch_memory,
            'total': param_memory + activation_memory + batch_memory
        }
    
    def benchmark_forward_pass(self, model: nn.Module, batch_size: int = 1, 
                               num_iterations: int = 10) -> float:
        """Benchmark forward pass speed"""
        model.eval()
        x = torch.randn(batch_size, 1, 128, 216, device=self.device)
        t = torch.randint(0, 1000, (batch_size,), device=self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = model(x, t)
        
        # Benchmark
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()
        
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(x, t)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.time() - start
        
        return elapsed / num_iterations
    
    def benchmark_inference(self, model_path: str, architecture: str, 
                           num_samples: int = 1, num_steps: int = 50) -> Dict:
        """Benchmark inference speed"""
        # Load model
        if architecture == 'unet':
            model = self._load_unet_model(model_path)
        else:
            model = self._load_transformer_model(model_path)
        
        model.eval()
        
        # Warm up
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        start = time.time()
        
        with torch.no_grad():
            for i in range(num_samples):
                shape = (1, 1, 128, 216)
                _ = model.sample(
                    shape=shape,
                    device=self.device,
                    num_inference_steps=num_steps,
                    eta=0.0
                )
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.time() - start
        
        return {
            'total_time': elapsed,
            'per_sample': elapsed / num_samples,
            'per_step': elapsed / (num_samples * num_steps)
        }
    
    def _load_unet_model(self, model_path: str) -> MusicDiffusionModel:
        """Load U-Net model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        config = checkpoint.get('model_config', {})
        
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
            model.load_state_dict(checkpoint)
        
        return model
    
    def _load_transformer_model(self, model_path: str) -> MusicDiffusionTransformer:
        """Load Transformer model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        config = checkpoint.get('model_config', {})
        
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
    
    def create_dummy_models(self) -> Tuple[MusicDiffusionModel, MusicDiffusionTransformer]:
        """Create dummy models for comparison"""
        # U-Net model
        unet = MelSpectrogramUNet(input_channels=1, base_channels=64, n_mels=128).to(self.device)
        unet_model = MusicDiffusionModel(unet=unet).to(self.device)
        
        # Transformer model
        transformer_model = MusicDiffusionTransformer(
            n_mels=128, n_frames=216, patch_size=8, embed_dim=256,
            num_layers=12, num_heads=8, mlp_dim=1024
        ).to(self.device)
        
        return unet_model, transformer_model
    
    def compare_architectures(self, unet_path: Optional[str] = None,
                             transformer_path: Optional[str] = None) -> Dict:
        """Compare U-Net and Transformer architectures"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'device': str(self.device),
            'comparison': {}
        }
        
        # Load or create models
        if unet_path and Path(unet_path).exists():
            print("Loading U-Net model...")
            unet_model = self._load_unet_model(unet_path)
        else:
            print("Creating dummy U-Net model...")
            unet_model, _ = self.create_dummy_models()
        
        if transformer_path and Path(transformer_path).exists():
            print("Loading Transformer model...")
            transformer_model = self._load_transformer_model(transformer_path)
        else:
            print("Creating dummy Transformer model...")
            _, transformer_model = self.create_dummy_models()
        
        # Compare U-Net
        print("\n" + "="*50)
        print("U-Net Architecture")
        print("="*50)
        
        unet_params = self.count_parameters(unet_model)
        unet_memory = self.estimate_memory(unet_model)
        unet_forward = self.benchmark_forward_pass(unet_model)
        
        results['comparison']['unet'] = {
            'parameters': {
                'total': unet_params[0],
                'trainable': unet_params[1],
                'formatted': f"{unet_params[0]/1e6:.1f}M"
            },
            'memory': {
                'model_gb': unet_memory['model'],
                'activations_gb': unet_memory['activations'],
                'batch_gb': unet_memory['batch'],
                'total_gb': unet_memory['total']
            },
            'forward_pass_ms': unet_forward * 1000
        }
        
        print(f"Parameters: {unet_params[0]:,} ({unet_params[0]/1e6:.1f}M)")
        print(f"Memory (estimated):")
        print(f"  Model: {unet_memory['model']:.3f} GB")
        print(f"  Activations: {unet_memory['activations']:.3f} GB")
        print(f"  Batch: {unet_memory['batch']:.3f} GB")
        print(f"  Total: {unet_memory['total']:.3f} GB")
        print(f"Forward pass: {unet_forward*1000:.2f} ms")
        
        # Compare Transformer
        print("\n" + "="*50)
        print("Diffusion Transformer Architecture")
        print("="*50)
        
        transformer_params = self.count_parameters(transformer_model)
        transformer_memory = self.estimate_memory(transformer_model)
        transformer_forward = self.benchmark_forward_pass(transformer_model)
        
        results['comparison']['transformer'] = {
            'parameters': {
                'total': transformer_params[0],
                'trainable': transformer_params[1],
                'formatted': f"{transformer_params[0]/1e6:.1f}M"
            },
            'memory': {
                'model_gb': transformer_memory['model'],
                'activations_gb': transformer_memory['activations'],
                'batch_gb': transformer_memory['batch'],
                'total_gb': transformer_memory['total']
            },
            'forward_pass_ms': transformer_forward * 1000
        }
        
        print(f"Parameters: {transformer_params[0]:,} ({transformer_params[0]/1e6:.1f}M)")
        print(f"Memory (estimated):")
        print(f"  Model: {transformer_memory['model']:.3f} GB")
        print(f"  Activations: {transformer_memory['activations']:.3f} GB")
        print(f"  Batch: {transformer_memory['batch']:.3f} GB")
        print(f"  Total: {transformer_memory['total']:.3f} GB")
        print(f"Forward pass: {transformer_forward*1000:.2f} ms")
        
        # Summary
        print("\n" + "="*50)
        print("Comparison Summary")
        print("="*50)
        
        param_ratio = transformer_params[0] / unet_params[0]
        memory_ratio = transformer_memory['total'] / unet_memory['total']
        speed_ratio = transformer_forward / unet_forward
        
        results['summary'] = {
            'parameter_ratio': param_ratio,
            'memory_ratio': memory_ratio,
            'speed_ratio': speed_ratio,
            'transformer_larger': {
                'parameters': f"{(param_ratio-1)*100:.1f}%",
                'memory': f"{(memory_ratio-1)*100:.1f}%",
                'slower': f"{(speed_ratio-1)*100:.1f}%"
            }
        }
        
        print(f"Transformer vs U-Net:")
        print(f"  Parameters: {param_ratio:.2f}x ({(param_ratio-1)*100:+.1f}%)")
        print(f"  Memory: {memory_ratio:.2f}x ({(memory_ratio-1)*100:+.1f}%)")
        print(f"  Speed: {speed_ratio:.2f}x slower ({(speed_ratio-1)*100:+.1f}%)")
        
        return results
    
    def save_comparison(self, results: Dict, output_path: str = 'model_comparison.json'):
        """Save comparison results"""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nComparison saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Compare U-Net and Diffusion Transformer models'
    )
    
    parser.add_argument('--unet_path', type=str, default=None,
                       help='Path to U-Net checkpoint')
    parser.add_argument('--transformer_path', type=str, default=None,
                       help='Path to Transformer checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda, cpu)')
    parser.add_argument('--output', type=str, default='model_comparison.json',
                       help='Output file for comparison results')
    
    args = parser.parse_args()
    
    # Create comparator
    comparator = ModelComparator(device=args.device)
    
    # Run comparison
    results = comparator.compare_architectures(
        unet_path=args.unet_path,
        transformer_path=args.transformer_path
    )
    
    # Save results
    comparator.save_comparison(results, output_path=args.output)


if __name__ == "__main__":
    main()
