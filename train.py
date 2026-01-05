import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
from pathlib import Path
import json
import argparse
from tqdm import tqdm
import wandb  # Pour le logging (optionnel)
import matplotlib.pyplot as plt

from diffusion_model import MusicDiffusionModel

class SpectrogramDataset(Dataset):
    """Dataset pour charger les spectrogrammes depuis les fichiers .npy du preprocessing"""
    
    def __init__(self, data_path: str, split: str = "train"):
        """
        Args:
            data_path: Chemin vers le dossier contenant les fichiers .npy
            split: "train", "val", ou "test"
        """
        self.data_path = Path(data_path)
        
        # Charge les données selon le split
        npy_file = self.data_path / f"{split}_specs.npy"
        if not npy_file.exists():
            raise FileNotFoundError(f"Fichier {npy_file} introuvable. Assurez-vous d'avoir exécuté le preprocessing.")
        
        print(f"Chargement de {npy_file}...")
        self.spectrograms = np.load(npy_file)
        print(f"Chargé {len(self.spectrograms)} spectrogrammes de forme {self.spectrograms.shape[1:]}")
        
    def __len__(self):
        return len(self.spectrograms)
    
    def __getitem__(self, idx):
        # Convertit en tensor PyTorch et normalise si nécessaire
        spec = torch.from_numpy(self.spectrograms[idx]).float()
        
        # Vérifie que la normalisation est correcte (doit être dans [-1, 1])
        if spec.min() < -1.1 or spec.max() > 1.1:
            print(f"Attention: spectrogramme {idx} mal normalisé: [{spec.min():.3f}, {spec.max():.3f}]")
        
        return spec

def create_dataloaders(data_dir: str, batch_size: int = 16, num_workers: int = 4):
    """Crée les dataloaders train/val/test à partir des fichiers .npy"""
    
    # Datasets
    train_dataset = SpectrogramDataset(data_dir, "train")
    val_dataset = SpectrogramDataset(data_dir, "val") 
    test_dataset = SpectrogramDataset(data_dir, "test")
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True  # Accélère le transfert vers GPU
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

class MusicDiffusionTrainer:
    """Classe principale pour l'entraînement du modèle de diffusion musicale"""
    
    def __init__(self, 
                 model,
                 train_dataloader,
                 val_dataloader=None,
                 optimizer=None,
                 scheduler=None,
                 device='auto',
                 save_dir='checkpoints',
                 log_wandb=False):
        
        # Device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        print(f"Utilisation du device: {self.device}")
        
        # Modèle
        self.model = model.to(self.device)
        
        # Assure-toi que le noise scheduler est aussi sur le bon device
        if hasattr(self.model, 'noise_scheduler'):
            self.model.noise_scheduler.to(self.device)
        
        # Data loaders
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        
        # Optimiseur par défaut
        if optimizer is None:
            self.optimizer = optim.AdamW(
                model.parameters(),
                lr=1e-4,
                weight_decay=1e-2,
                betas=(0.9, 0.95)
            )
        else:
            self.optimizer = optimizer
            
        # Scheduler
        self.scheduler = scheduler
        
        # Sauvegarde
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        
        # Logging
        self.log_wandb = log_wandb
        if log_wandb:
            try:
                wandb.init(
                    project="music-diffusion",
                    config={
                        "architecture": "UNet",
                        "dataset": "FMA-Small",
                        "optimizer": "AdamW",
                        "lr": self.optimizer.param_groups[0]['lr']
                    }
                )
            except:
                print("Impossible d'initialiser wandb, logging local uniquement")
                self.log_wandb = False
        
        # Métriques
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
    def train_epoch(self, epoch):
        """Entraîne le modèle pour une époque"""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_dataloader)
        
        pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            batch = batch.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            loss = self.model.compute_loss(batch)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping pour la stabilité
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{avg_loss:.4f}'
            })
            
            # Log to wandb
            if self.log_wandb and batch_idx % 50 == 0:
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
        """Valide le modèle"""
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
        """Sauvegarde un checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Sauvegarde checkpoint courant
        checkpoint_path = self.save_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Sauvegarde meilleur modèle
        if is_best:
            best_path = self.save_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"Nouveau meilleur modèle sauvegardé! Loss: {self.best_val_loss:.4f}")
    
    def load_checkpoint(self, checkpoint_path):
        """Charge un checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        return checkpoint['epoch']
    
    def train(self, num_epochs, save_every=10, validate_every=5):
        """Boucle d'entraînement principale"""
        print(f"Début de l'entraînement pour {num_epochs} époques")
        print(f"Taille du dataset: {len(self.train_dataloader.dataset)}")
        print(f"Batch size: {self.train_dataloader.batch_size}")
        print(f"Paramètres du modèle: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        
        start_epoch = 1
        
        for epoch in range(start_epoch, num_epochs + 1):
            # Entraînement
            train_loss = self.train_epoch(epoch)
            
            # Validation
            val_loss = None
            if epoch % validate_every == 0:
                val_loss = self.validate_epoch(epoch)
                
                # Check si c'est le meilleur modèle
                if val_loss is not None and val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(epoch, is_best=True)
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    metric = val_loss if val_loss is not None else train_loss
                    self.scheduler.step(metric)
                else:
                    self.scheduler.step()
            
            # Logging
            current_lr = self.optimizer.param_groups[0]['lr']
            log_dict = {
                'epoch': epoch,
                'train_loss': train_loss,
                'learning_rate': current_lr
            }
            
            if val_loss is not None:
                log_dict['val_loss'] = val_loss
            
            if self.log_wandb:
                wandb.log(log_dict)
            
            print(f"Epoch {epoch}/{num_epochs} - Train Loss: {train_loss:.4f}", end="")
            if val_loss is not None:
                print(f" - Val Loss: {val_loss:.4f}", end="")
            print(f" - LR: {current_lr:.2e}")
            
            # Sauvegarde périodique
            if epoch % save_every == 0:
                self.save_checkpoint(epoch)
        
        print("Entraînement terminé!")
        
        # Sauvegarde finale
        self.save_checkpoint(num_epochs)
        self.plot_training_curves()
    
    def plot_training_curves(self):
        """Plot les courbes d'entraînement"""
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
    parser = argparse.ArgumentParser(description='Entraînement du modèle de diffusion musicale')
    
    # Données
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Répertoire des données préprocessées (.npy)')
    
    # Modèle
    parser.add_argument('--base_channels', type=int, default=64,
                      help='Nombre de channels de base du U-Net')
    parser.add_argument('--timesteps', type=int, default=1000,
                      help='Nombre de timesteps de diffusion')
    
    # Entraînement
    parser.add_argument('--batch_size', type=int, default=16,
                      help='Taille de batch')
    parser.add_argument('--num_epochs', type=int, default=100,
                      help='Nombre d\'époques')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                      help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-2,
                      help='Weight decay')
    
    # Sauvegarde et logging
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                      help='Répertoire de sauvegarde')
    parser.add_argument('--save_every', type=int, default=10,
                      help='Sauvegarder tous les N époques')
    parser.add_argument('--validate_every', type=int, default=5,
                      help='Valider tous les N époques')
    parser.add_argument('--log_wandb', action='store_true',
                      help='Activer le logging wandb')
    
    # Système
    parser.add_argument('--num_workers', type=int, default=4,
                      help='Nombre de workers pour le DataLoader')
    parser.add_argument('--device', type=str, default='auto',
                      help='Device à utiliser (auto, cuda, cpu)')
    
    args = parser.parse_args()
    
    # Vérification des données préprocessées
    if not os.path.exists(args.data_dir):
        print(f"❌ Répertoire {args.data_dir} introuvable.")
        print("📋 Veuillez d'abord exécuter le notebook de preprocessing pour créer les fichiers .npy")
        print("   Les fichiers attendus sont: train_specs.npy, val_specs.npy, test_specs.npy")
        exit(1)
    
    # Vérification des fichiers requis
    required_files = ['train_specs.npy', 'val_specs.npy', 'test_specs.npy']
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(args.data_dir, f))]
    
    if missing_files:
        print(f"❌ Fichiers manquants dans {args.data_dir}: {missing_files}")
        print("📋 Exécutez d'abord le notebook de preprocessing pour créer ces fichiers")
        exit(1)
    
    # Création des dataloaders
    print("📊 Chargement des données...")
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    print(f"📈 Dataset train: {len(train_dataloader.dataset)} samples")
    print(f"📊 Dataset validation: {len(val_dataloader.dataset)} samples") 
    print(f"🧪 Dataset test: {len(test_dataloader.dataset)} samples")
    
    # Création du modèle (adapté aux dimensions du preprocessing)
    model = MusicDiffusionModel(
        input_channels=1,
        base_channels=args.base_channels,
        timesteps=args.timesteps,
        schedule_type="cosine"
    )
    
    # Optimiseur et scheduler
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
    trainer = MusicDiffusionTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=args.device,
        save_dir=args.save_dir,
        log_wandb=args.log_wandb
    )
    
    # Entraînement
    trainer.train(
        num_epochs=args.num_epochs,
        save_every=args.save_every,
        validate_every=args.validate_every
    )

if __name__ == "__main__":
    main()