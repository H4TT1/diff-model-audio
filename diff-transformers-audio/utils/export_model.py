#!/usr/bin/env python3
"""
Model export utility for production deployment
Converts a training checkpoint to an optimized inference model
"""

import torch
import argparse
from pathlib import Path
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.diffusion_model import MusicDiffusionModel

def export_model(checkpoint_path: str, 
                 output_path: str,
                 device: str = 'auto',
                 verify: bool = True) -> bool:
    """
    Export a model checkpoint to an optimized inference model
    
    Args:
        checkpoint_path: Path to training checkpoint
        output_path: Path to save the inference model
        device: Device to use for verification
        verify: Whether to verify the model before saving
    
    Returns:
        True if export was successful
    """
    
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    print(f"Exporting model from: {checkpoint_path}")
    print(f"Output path: {output_path}")
    print(f"Device: {device}")
    print("=" * 60)
    
    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        print("✓ Checkpoint loaded")
    except FileNotFoundError:
        print(f"✗ Error: Checkpoint file not found: {checkpoint_path}")
        return False
    except Exception as e:
        print(f"✗ Error loading checkpoint: {e}")
        return False
    
    # Extract configuration
    if 'model_config' in checkpoint:
        config = checkpoint['model_config']
        print("✓ Model configuration found")
    else:
        print("⚠ Model configuration not found, using defaults")
        config = {
            'input_channels': 1,
            'base_channels': 64,
            'timesteps': 1000,
            'schedule_type': 'cosine'
        }
    
    print(f"  Configuration:")
    print(f"    - input_channels: {config.get('input_channels')}")
    print(f"    - base_channels: {config.get('base_channels')}")
    print(f"    - timesteps: {config.get('timesteps')}")
    print(f"    - schedule_type: {config.get('schedule_type')}")
    
    # Create and load model
    try:
        model = MusicDiffusionModel(
            input_channels=config.get('input_channels', 1),
            base_channels=config.get('base_channels', 64),
            timesteps=config.get('timesteps', 1000),
            schedule_type=config.get('schedule_type', 'cosine')
        ).to(device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print("✓ Model created and weights loaded")
    except Exception as e:
        print(f"✗ Error creating/loading model: {e}")
        return False
    
    # Verify model (optional)
    if verify:
        print("\nVerifying model...")
        try:
            with torch.no_grad():
                x = torch.randn(1, 1, 128, 216).to(device)
                t = torch.zeros(1, dtype=torch.long).to(device)
                output = model.unet(x, t)
                
                if output.shape == x.shape:
                    print("✓ Forward pass verification passed")
                else:
                    print(f"✗ Output shape mismatch: {output.shape} vs {x.shape}")
                    return False
        except Exception as e:
            print(f"✗ Verification failed: {e}")
            return False
    
    # Prepare export data
    export_data = {
        'model_state_dict': model.state_dict(),
        'model_config': config,
        'export_info': {
            'from_checkpoint': str(checkpoint_path),
            'epoch': checkpoint.get('epoch', 'unknown'),
            'best_val_loss': checkpoint.get('best_val_loss', None),
            'export_device': str(device)
        }
    }
    
    # Save model
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save(export_data, output_path)
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"\n✓ Model exported successfully")
        print(f"  Size: {file_size_mb:.2f} MB")
        print(f"  Location: {output_path}")
    except Exception as e:
        print(f"✗ Error saving model: {e}")
        return False
    
    # Summary
    print("\n" + "=" * 60)
    print("Export Summary:")
    print(f"  ✓ Original checkpoint: {Path(checkpoint_path).stat().st_size / (1024*1024):.2f} MB")
    print(f"  ✓ Exported model: {file_size_mb:.2f} MB")
    print(f"  ✓ Epoch: {export_data['export_info']['epoch']}")
    if export_data['export_info']['best_val_loss']:
        print(f"  ✓ Best Val Loss: {export_data['export_info']['best_val_loss']:.4f}")
    print("\n✓ Model is ready for inference!")
    print(f"\nUsage:")
    print(f"  python inference.py --model_path {output_path} --num_samples 5")
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description='Export a training checkpoint to an optimized inference model'
    )
    
    parser.add_argument('checkpoint', type=str,
                       help='Path to the training checkpoint')
    parser.add_argument('-o', '--output', type=str, default=None,
                       help='Output path (default: checkpoint_dir/exported_model.pt)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device for verification (auto, cuda, cpu)')
    parser.add_argument('--no-verify', action='store_true',
                       help='Skip model verification')
    
    args = parser.parse_args()
    
    # Verify input exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        exit(1)
    
    # Determine output path
    if args.output is None:
        output_path = checkpoint_path.parent / 'exported_model.pt'
    else:
        output_path = args.output
    
    # Export
    success = export_model(
        checkpoint_path=str(checkpoint_path),
        output_path=str(output_path),
        device=args.device,
        verify=not args.no_verify
    )
    
    exit(0 if success else 1)

if __name__ == "__main__":
    main()
