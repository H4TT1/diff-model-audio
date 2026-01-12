#!/usr/bin/env python3
"""
Training script for Diffusion Transformer model
Alternative to U-Net version: Uses pure transformer architecture
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.diffusion_transformer import MusicDiffusionTransformer


class SpectrogramDataset(Dataset):
    """Load preprocessed mel-spectrograms"""
    
    def __init__(self, data_path: str, split: str = "train"):
        """
        Args:
            data_path: Path to folder with .npy files
            split: "train", "val", or "test"
        """
        self.data_path = Path(data_path)
        
        npy_file = self.data_path / f"{split}_specs.npy"
        if not npy_file.exists():
            raise FileNotFoundError(f"File not found: {npy_file}")
        
        print(f"Loading {npy_file}...")
        self.spectrograms = np.load(npy_file)
        print(f"Loaded {len(self.spectrograms)} spectrograms of shape {self.spectrograms.shape[1:]}")
        
    def __len__(self):
        return len(self.spectrograms)
    
    def __getitem__(self, idx):
        spec = torch.from_numpy(self.spectrograms[idx]).float()
        
        if spec.min() < -1.1 or spec.max() > 1.1:
            print(f"Warning: spectrogram {idx} not normalized: [{spec.min():.3f}, {spec.max():.3f}]")
        
        return spec


def create_dataloaders(data_dir: str, batch_size: int = 16, num_workers: int = 4):
    """Create train/val/test dataloaders"""
    
    train_dataset = SpectrogramDataset(data_dir, "train")
    val_dataset = SpectrogramDataset(data_dir, "val") 
    test_dataset = SpectrogramDataset(data_dir, "test")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


class DiffusionTransformerTrainer:
    """Trainer for Diffusion Transformer model"""
    
    def __init__(self,
                 model,
                 train_dataloader,
                 val_dataloader=None,
                 optimizer=None,
                 scheduler=None,
                 device='auto',
                 save_dir='checkpoints_transformer',
                 log_wandb=False):
        
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        self.model = model.to(self.device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        
        if optimizer is None:
            self.optimizer = optim.AdamW(
                model.parameters(),
                lr=1e-4,
                weight_decay=1e-2,
                betas=(0.9, 0.95)
            )
        else:
            self.optimizer = optimizer
        
        self.scheduler = scheduler
        
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        
        self.log_wandb = log_wandb
        if log_wandb:
            try:
                import wandb
                wandb.init(
                    project="music-diffusion-transformer",
                    config={
                        "architecture": "Diffusion Transformer",
                        "dataset": "FMA-Small",
                        "optimizer": "AdamW",
                        "lr": self.optimizer.param_groups[0]['lr']
                    }
                )
            except:
                print("Could not initialize wandb, logging locally only")
                self.log_wandb = False
        
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_dataloader)
        
        pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            batch = batch.to(self.device)
            
            self.optimizer.zero_grad()
            loss = self.model.compute_loss(batch)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{avg_loss:.4f}'
            })
            
            if self.log_wandb and batch_idx % 50 == 0:
                import wandb
                wandb.log({
                    'train_loss_step': loss.item(),
                    'epoch': epoch,
                    'step': epoch * num_batches + batch_idx
                })
        
        avg_train_loss = total_loss / num_batches
        self.train_losses.append(avg_train_loss)
        
        return avg_train_loss
    
    @torch.no_grad()
    def validate_epoch(self, epoch):
        """Validate the model"""
        if self.val_dataloader is None:
            return None
        
        self.model.eval()
        total_loss = 0
        num_batches = len(self.val_dataloader)
        
        pbar = tqdm(self.val_dataloader, desc=f"Validation {epoch}")
        
        for batch in pbar:
            batch = batch.to(self.device)
            loss = self.model.compute_loss(batch)
            total_loss += loss.item()
            
            pbar.set_postfix({'Val Loss': f'{loss.item():.4f}'})
        
        avg_val_loss = total_loss / num_batches
        self.val_losses.append(avg_val_loss)
        
        return avg_val_loss
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'model_config': {
                'n_mels': 128,
                'n_frames': 216,
                'patch_size': 8,
                'embed_dim': 256,
                'num_layers': 12,
                'num_heads': 8,
                'mlp_dim': 1024,
                'timesteps': 1000,
                'schedule_type': 'cosine',
                'architecture': 'diffusion_transformer'
            }
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        checkpoint_path = self.save_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = self.save_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"✓ New best model saved! Loss: {self.best_val_loss:.4f}")
    
    def save_model_for_inference(self, epoch=None, save_name='model_for_inference.pt'):
        """Save clean inference model"""
        inference_checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'n_mels': 128,
                'n_frames': 216,
                'patch_size': 8,
                'embed_dim': 256,
                'num_layers': 12,
                'num_heads': 8,
                'mlp_dim': 1024,
                'timesteps': 1000,
                'schedule_type': 'cosine',
                'architecture': 'diffusion_transformer'
            },
            'epoch': epoch if epoch else 'final'
        }
        
        save_path = self.save_dir / save_name
        torch.save(inference_checkpoint, save_path)
        print(f"✓ Model saved for inference: {save_path}")
        return save_path
    
    def train(self, num_epochs, save_every=10, validate_every=5):
        """Main training loop"""
        print(f"Starting training for {num_epochs} epochs")
        print(f"Dataset size: {len(self.train_dataloader.dataset)}")
        print(f"Batch size: {self.train_dataloader.batch_size}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        
        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch(epoch)
            
            val_loss = None
            if epoch % validate_every == 0:
                val_loss = self.validate_epoch(epoch)
                
                if val_loss is not None and val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(epoch, is_best=True)
            
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    metric = val_loss if val_loss is not None else train_loss
                    self.scheduler.step(metric)
                else:
                    self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch}/{num_epochs} - Train Loss: {train_loss:.4f}", end="")
            if val_loss is not None:
                print(f" - Val Loss: {val_loss:.4f}", end="")
            print(f" - LR: {current_lr:.2e}")
            
            if epoch % save_every == 0:
                self.save_checkpoint(epoch)
        
        print("Training complete!")
        
        self.save_checkpoint(num_epochs)
        
        if self.best_val_loss != float('inf'):
            print(f"\n✓ Best model with Val Loss: {self.best_val_loss:.4f}")
        
        self.save_model_for_inference(epoch=num_epochs, save_name='final_model_for_inference.pt')
        
        self.plot_training_curves()
    
    def plot_training_curves(self):
        """Plot training curves"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        if self.val_losses:
            plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Curves')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.train_losses, label='Train Loss (Log Scale)')
        if self.val_losses:
            plt.plot(self.val_losses, label='Validation Loss (Log Scale)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (Log)')
        plt.yscale('log')
        plt.title('Training Curves (Log Scale)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Train Diffusion Transformer for music generation')
    
    # Data
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument('--data_dir', type=str, default=os.path.join(base_dir, 'outputs', 'processed_data'),
                      help='Directory with preprocessed data (.npy files)')
    
    # Model
    parser.add_argument('--embed_dim', type=int, default=256,
                      help='Embedding dimension')
    parser.add_argument('--num_layers', type=int, default=12,
                      help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=8,
                      help='Number of attention heads')
    parser.add_argument('--patch_size', type=int, default=8,
                      help='Patch size')
    parser.add_argument('--timesteps', type=int, default=1000,
                      help='Number of diffusion timesteps')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=16,
                      help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100,
                      help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                      help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-2,
                      help='Weight decay')
    
    # Saving and logging
    parser.add_argument('--save_dir', type=str, default=os.path.join(base_dir, 'checkpoints_transformer'),
                      help='Directory to save checkpoints')
    parser.add_argument('--save_every', type=int, default=10,
                      help='Save checkpoint every N epochs')
    parser.add_argument('--validate_every', type=int, default=5,
                      help='Validate every N epochs')
    parser.add_argument('--log_wandb', action='store_true',
                      help='Log to Weights & Biases')
    
    # System
    parser.add_argument('--num_workers', type=int, default=4,
                      help='Number of data loader workers')
    parser.add_argument('--device', type=str, default='auto',
                      help='Device to use (auto, cuda, cpu)')
    
    args = parser.parse_args()
    
    # Verify data
    if not os.path.exists(args.data_dir):
        print(f"❌ Directory not found: {args.data_dir}")
        exit(1)
    
    required_files = ['train_specs.npy', 'val_specs.npy', 'test_specs.npy']
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(args.data_dir, f))]
    
    if missing_files:
        print(f"❌ Missing files in {args.data_dir}: {missing_files}")
        exit(1)
    
    # Create dataloaders
    print("📊 Loading data...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    print(f"📈 Train samples: {len(train_loader.dataset)}")
    print(f"📊 Val samples: {len(val_loader.dataset)}")
    print(f"🧪 Test samples: {len(test_loader.dataset)}")
    
    # Create model
    model = MusicDiffusionTransformer(
        n_mels=128,
        n_frames=216,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        mlp_dim=args.embed_dim * 4,
        timesteps=args.timesteps,
        schedule_type="cosine"
    )
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95)
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs,
        eta_min=1e-6
    )
    
    # Trainer
    trainer = DiffusionTransformerTrainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=args.device,
        save_dir=args.save_dir,
        log_wandb=args.log_wandb
    )
    
    # Train
    trainer.train(
        num_epochs=args.num_epochs,
        save_every=args.save_every,
        validate_every=args.validate_every
    )


if __name__ == "__main__":
    main()
