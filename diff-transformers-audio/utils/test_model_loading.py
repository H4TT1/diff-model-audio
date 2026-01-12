#!/usr/bin/env python3
"""
Utility script to test model loading and configuration
Useful for debugging model loading issues
"""

import torch
import argparse
from pathlib import Path
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.diffusion_model import MusicDiffusionModel

def test_model_loading(checkpoint_path: str, device: str = 'auto'):
    """Test loading a model checkpoint and verify configuration"""
    
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    print(f"Testing model loading from: {checkpoint_path}")
    print(f"Device: {device}")
    print("=" * 60)
    
    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        print("✓ Checkpoint loaded successfully")
    except FileNotFoundError:
        print("✗ Error: Checkpoint file not found")
        return False
    except Exception as e:
        print(f"✗ Error loading checkpoint: {e}")
        return False
    
    # Check for model configuration
    print("\nConfiguration:")
    if 'model_config' in checkpoint:
        config = checkpoint['model_config']
        print(f"  ✓ model_config found in checkpoint")
        print(f"    - input_channels: {config.get('input_channels', 'N/A')}")
        print(f"    - base_channels: {config.get('base_channels', 'N/A')}")
        print(f"    - timesteps: {config.get('timesteps', 'N/A')}")
        print(f"    - schedule_type: {config.get('schedule_type', 'N/A')}")
    else:
        print(f"  ⚠ model_config NOT found - using defaults")
        config = {
            'input_channels': 1,
            'base_channels': 64,
            'timesteps': 1000,
            'schedule_type': 'cosine'
        }
    
    # Check for training info
    print("\nTraining Information:")
    if 'epoch' in checkpoint:
        print(f"  - Epoch: {checkpoint['epoch']}")
    if 'best_val_loss' in checkpoint:
        print(f"  - Best Val Loss: {checkpoint.get('best_val_loss', 'N/A'):.4f}")
    if 'train_losses' in checkpoint:
        print(f"  - Training Epochs: {len(checkpoint.get('train_losses', []))}")
    
    # Verify state dict keys
    print("\nCheckpoint Contents:")
    for key in checkpoint.keys():
        if key == 'model_state_dict':
            num_params = sum(p.numel() for p in [torch.zeros_like(v) for v in checkpoint[key].values()])
            print(f"  ✓ {key}: {len(checkpoint[key])} parameter groups (~{num_params:,} params)")
        elif key in ['optimizer_state_dict', 'scheduler_state_dict']:
            print(f"  ✓ {key}: present")
        else:
            print(f"  • {key}: {type(checkpoint[key]).__name__}")
    
    # Try to create and load model
    print("\nAttempting to create and load model...")
    try:
        model = MusicDiffusionModel(
            input_channels=config.get('input_channels', 1),
            base_channels=config.get('base_channels', 64),
            timesteps=config.get('timesteps', 1000),
            schedule_type=config.get('schedule_type', 'cosine')
        ).to(device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("✓ Model created and weights loaded successfully")
        else:
            print("⚠ No model_state_dict found in checkpoint")
            return False
            
    except Exception as e:
        print(f"✗ Error creating/loading model: {e}")
        return False
    
    # Test forward pass
    print("\nTesting forward pass...")
    try:
        with torch.no_grad():
            batch_size = 2
            x = torch.randn(batch_size, 1, 128, 216).to(device)
            t = torch.randint(0, config.get('timesteps', 1000), (batch_size,)).to(device)
            
            output = model.unet(x, t)
            
            print(f"✓ Forward pass successful")
            print(f"  - Input shape: {x.shape}")
            print(f"  - Output shape: {output.shape}")
            
            if output.shape == x.shape:
                print("✓ Output shape matches input shape (correct!)")
            else:
                print("✗ Output shape does not match input shape")
                return False
                
    except Exception as e:
        print(f"✗ Error in forward pass: {e}")
        return False
    
    # Test sampling (quick test)
    print("\nTesting sampling (quick 5-step test)...")
    try:
        with torch.no_grad():
            shape = (1, 1, 128, 216)
            samples = model.sample(shape=shape, device=device, num_inference_steps=5)
            print(f"✓ Sampling successful")
            print(f"  - Sample shape: {samples.shape}")
            print(f"  - Value range: [{samples.min():.3f}, {samples.max():.3f}]")
            
    except Exception as e:
        print(f"✗ Error in sampling: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✓ All tests passed! Model is ready for inference.")
    return True

def main():
    parser = argparse.ArgumentParser(
        description='Test model checkpoint loading and configuration'
    )
    parser.add_argument('model_path', type=str,
                       help='Path to the model checkpoint')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cuda, cpu)')
    
    args = parser.parse_args()
    
    # Verify file exists
    if not Path(args.model_path).exists():
        print(f"Error: File not found: {args.model_path}")
        exit(1)
    
    # Run tests
    success = test_model_loading(args.model_path, args.device)
    exit(0 if success else 1)

if __name__ == "__main__":
    main()
