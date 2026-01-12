#!/usr/bin/env python3
"""
Script d'inférence pour Diffusion Transformer : génère spectrogrammes ET audio
"""

import torch
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import argparse
from typing import Optional
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.diffusion_transformer import MusicDiffusionTransformer


class MusicGenerator:
    """Générateur de musique complet avec Diffusion Transformer"""
    
    def __init__(self, model_path: str, device: str = 'cpu'):
        self.device = torch.device(device)
        print(f"📱 Device: {self.device}")
        
        # Charge le modèle avec détection auto des paramètres
        self.model = self.load_model(model_path)
        self.model.eval()
        
        # Paramètres audio (correspondant au preprocessing)
        self.sample_rate = 22050
        self.n_mels = 128
        self.n_fft = 2048
        self.hop_length = 512
        self.duration = 5.0
        
        print("✅ Modèle chargé avec succès!")
    
    def load_model(self, model_path: str) -> MusicDiffusionTransformer:
        """Charge le modèle DiT avec détection automatique des paramètres"""
        print(f"📂 Chargement de {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Détecte les paramètres automatiquement du state_dict
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # Détecte embed_dim depuis patch_embed
        patch_embed_weight = state_dict['dit.patch_embed.patch_embed.weight']
        embed_dim = patch_embed_weight.shape[0]
        
        # Détecte num_layers (compte les transformer blocks)
        num_layers = sum(1 for k in state_dict.keys() if 'transformer_blocks' in k and 'norm1.weight' in k)
        
        # Détecte num_heads depuis la première couche d'attention
        # MultiheadAttention stocke les poids comme (embed_dim * 3, embed_dim) pour Q, K, V
        first_mha_weight = state_dict['dit.transformer_blocks.0.mha.in_proj_weight']
        # num_heads peut être déduit mais on utilise une valeur par défaut
        num_heads = 8  # Valeur standard, difficile à détecter précisément
        
        # Détecte mlp_dim depuis la première couche MLP
        first_mlp_weight = state_dict['dit.transformer_blocks.0.mlp.0.weight']
        mlp_dim = first_mlp_weight.shape[0]
        
        # Détecte patch_size depuis les embeddings
        pos_embed = state_dict['dit.patch_embed.pos_embed']
        num_patches = pos_embed.shape[1]
        # Pour 128x216 avec patch_size=8: (128/8) * (216/8) = 16 * 27 = 432 patches
        patch_size = 8  # Valeur standard
        
        # Affiche les infos
        epoch = checkpoint.get('epoch', 'N/A')
        train_loss = checkpoint.get('train_losses', [])[-1] if checkpoint.get('train_losses') else 'N/A'
        val_loss = checkpoint.get('val_losses', [])[-1] if checkpoint.get('val_losses') else 'N/A'
        
        print(f"🔧 Architecture détectée:")
        print(f"   embed_dim={embed_dim}, num_layers={num_layers}")
        print(f"   num_heads={num_heads}, mlp_dim={mlp_dim}")
        print(f"   patch_size={patch_size}, num_patches={num_patches}")
        print(f"📊 Checkpoint: époque={epoch}")
        if train_loss != 'N/A':
            print(f"   Loss: train={train_loss:.4f}, val={val_loss if val_loss != 'N/A' else 'N/A'}")
        
        # Crée et charge le modèle
        model = MusicDiffusionTransformer(
            n_mels=128,
            n_frames=216,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            timesteps=1000,
            schedule_type="cosine"
        ).to(self.device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    
    @torch.no_grad()
    def generate_spectrograms(self, 
                             num_samples: int = 1,
                             num_steps: int = 50,
                             eta: float = 0.0,
                             seed: Optional[int] = None) -> np.ndarray:
        """Génère des spectrogrammes avec le DiT"""
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        print(f"\n🎨 Génération de {num_samples} spectrogrammes ({num_steps} steps, eta={eta})...")
        
        spectrograms = []
        for i in range(num_samples):
            sample = self.model.sample(
                shape=(1, 1, 128, 216),
                device=self.device,
                num_inference_steps=num_steps,
                eta=eta
            )
            spec = sample.cpu().numpy()[0, 0]
            spectrograms.append(spec)
            print(f"   ✓ Échantillon {i+1}/{num_samples} [min: {spec.min():.1f}, max: {spec.max():.1f}]")
        
        return np.array(spectrograms)
    
    def normalize_spectrogram(self, spec: np.ndarray, method: str = 'clip') -> np.ndarray:
        """Normalise un spectrogramme pour éviter les valeurs extrêmes"""
        
        if method == 'clip':
            # Clip puis normalise vers [-1, 1]
            spec_clipped = np.clip(spec, -10, 10)
            return spec_clipped
        
        elif method == 'minmax':
            # Min-max vers [-1, 1]
            spec_min, spec_max = spec.min(), spec.max()
            if spec_max - spec_min > 0:
                return 2 * (spec - spec_min) / (spec_max - spec_min) - 1
            return np.zeros_like(spec)
        
        elif method == 'standard':
            # Standardisation + clip
            mean, std = spec.mean(), spec.std()
            if std > 0:
                spec_norm = (spec - mean) / std
                return np.clip(spec_norm, -3, 3)
            return np.zeros_like(spec)
        
        return spec
    
    def spectrogram_to_audio(self, spec: np.ndarray, normalize: str = 'clip') -> np.ndarray:
        """Convertit un spectrogramme en audio"""
        
        # Normalise si nécessaire
        spec_norm = self.normalize_spectrogram(spec, method=normalize)
        
        # Dénormalise (inverse du preprocessing: (log_mel + 40) / 40)
        log_mel = spec_norm * 40 - 40
        
        # Convertit en magnitude
        mel_mag = librosa.db_to_power(log_mel)
        
        # Nettoie les valeurs invalides
        mel_mag = np.nan_to_num(mel_mag, nan=0.0, posinf=1e10, neginf=0.0)
        
        # Griffin-Lim pour reconstruction
        try:
            audio = librosa.feature.inverse.mel_to_audio(
                mel_mag,
                sr=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_iter=32,
                length=int(self.duration * self.sample_rate)
            )
            
            # Normalise l'audio
            if np.abs(audio).max() > 0:
                audio = audio / np.abs(audio).max() * 0.9
            
            return audio
        
        except Exception as e:
            print(f"        ⚠️  Erreur conversion: {e}")
            return None
    
    def save_spectrogram_image(self, spec: np.ndarray, filepath: Path, 
                              show_normalized: bool = False, 
                              normalized_spec: np.ndarray = None):
        """Sauvegarde un spectrogramme en image"""
        
        if show_normalized and normalized_spec is not None:
            # Comparaison original vs normalisé
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
            
            im1 = ax1.imshow(spec, aspect='auto', origin='lower', cmap='viridis')
            ax1.set_title(f'Original [min={spec.min():.1f}, max={spec.max():.1f}]')
            ax1.set_xlabel('Temps (frames)')
            ax1.set_ylabel('Mel bins')
            plt.colorbar(im1, ax=ax1)
            
            im2 = ax2.imshow(normalized_spec, aspect='auto', origin='lower', cmap='viridis')
            ax2.set_title(f'Normalisé [min={normalized_spec.min():.2f}, max={normalized_spec.max():.2f}]')
            ax2.set_xlabel('Temps (frames)')
            ax2.set_ylabel('Mel bins')
            plt.colorbar(im2, ax=ax2)
        else:
            # Simple image
            plt.figure(figsize=(10, 4))
            plt.imshow(spec, aspect='auto', origin='lower', cmap='viridis')
            plt.colorbar(label='Amplitude')
            plt.xlabel('Temps (frames)')
            plt.ylabel('Mel bins')
            plt.title(f'Spectrogramme [min={spec.min():.1f}, max={spec.max():.1f}]')
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
    
    def generate(self, 
                output_dir: str = 'generated_music_dit',
                num_samples: int = 3,
                num_steps: int = 1000,
                eta: float = 0.0,
                normalize_method: str = 'clip',
                save_spectrograms: bool = True,
                seed: Optional[int] = None):
        """
        Génère spectrogrammes ET audio avec le Diffusion Transformer
        
        Args:
            output_dir: Dossier de sortie
            num_samples: Nombre d'échantillons à générer
            num_steps: Steps de diffusion (plus = meilleur qualité mais plus lent)
            eta: Stochasticity pour DDIM (0=déterministe, 1=DDPM)
            normalize_method: Méthode de normalisation (clip/minmax/standard)
            save_spectrograms: Sauvegarder les images de spectrogrammes
            seed: Seed pour reproductibilité
        """
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        print(f"\n📁 Dossier de sortie: {output_dir}/")
        print(f"🔧 Normalisation: {normalize_method}")
        
        # Génère les spectrogrammes
        spectrograms = self.generate_spectrograms(
            num_samples=num_samples,
            num_steps=num_steps,
            eta=eta,
            seed=seed
        )
        
        # Sauvegarde .npy
        npy_path = output_path / "spectrograms.npy"
        np.save(npy_path, spectrograms)
        print(f"\n💾 Sauvegardé: {npy_path}")
        
        # Convertit en audio et sauvegarde tout
        print(f"\n🎵 Conversion en audio...")
        
        generated_files = []
        success_count = 0
        
        for i, spec in enumerate(spectrograms, 1):
            print(f"\n  📝 Échantillon {i}/{num_samples}:")
            
            # Normalise pour audio
            spec_norm = self.normalize_spectrogram(spec, method=normalize_method)
            
            # Convertit en audio
            audio = self.spectrogram_to_audio(spec, normalize=normalize_method)
            
            if audio is not None and len(audio) > 0:
                # Sauvegarde WAV
                wav_path = output_path / f"audio_{i:02d}.wav"
                sf.write(wav_path, audio, self.sample_rate)
                generated_files.append(wav_path)
                success_count += 1
                print(f"    🎧 Audio: {wav_path.name} ({len(audio)/self.sample_rate:.1f}s)")
            else:
                print(f"    ❌ Échec conversion audio")
            
            # Sauvegarde image spectrogramme
            if save_spectrograms:
                img_path = output_path / f"spec_{i:02d}.png"
                self.save_spectrogram_image(
                    spec, img_path,
                    show_normalized=True,
                    normalized_spec=spec_norm
                )
                print(f"    🖼️  Image: {img_path.name}")
        
        # Vue d'ensemble
        if save_spectrograms and len(spectrograms) > 1:
            fig, axes = plt.subplots(len(spectrograms), 1, figsize=(12, 3*len(spectrograms)))
            if len(spectrograms) == 1:
                axes = [axes]
            
            for i, (ax, spec) in enumerate(zip(axes, spectrograms)):
                im = ax.imshow(spec, aspect='auto', origin='lower', cmap='viridis')
                ax.set_xlabel('Temps (frames)')
                ax.set_ylabel('Mel bins')
                ax.set_title(f'Échantillon {i+1} [min: {spec.min():.1f}, max: {spec.max():.1f}]')
                plt.colorbar(im, ax=ax)
            
            plt.tight_layout()
            overview_path = output_path / 'overview.png'
            plt.savefig(overview_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"\n  🎨 Vue d'ensemble: {overview_path.name}")
        
        # Résumé
        print(f"\n{'='*60}")
        print(f"✨ GÉNÉRATION TERMINÉE")
        print(f"{'='*60}")
        print(f"  📊 Spectrogrammes: {len(spectrograms)}")
        print(f"  🎵 Audio réussis: {success_count}/{len(spectrograms)}")
        print(f"  📁 Fichiers dans: {output_dir}/")
        
        return generated_files


def main():
    parser = argparse.ArgumentParser(
        description='🎵 Génération de musique avec Diffusion Transformer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  # Génération rapide (3 samples)
  python inference_dit.py --model checkpoints/dit_best.pt -n 3
  
  # Haute qualité (10 samples, 100 steps)
  python inference_dit.py --model checkpoints/dit_best.pt -n 10 -s 100
  
  # Avec stochasticité (plus de variété)
  python inference_dit.py --model checkpoints/dit_best.pt --eta 0.5
  
  # Essayer différentes normalisations
  python inference_dit.py --model checkpoints/dit_best.pt --normalize minmax
  
  # Reproductible avec seed
  python inference_dit.py --model checkpoints/dit_best.pt --seed 42
  
  # GPU pour génération plus rapide
  python inference_dit.py --model checkpoints/dit_best.pt --device cuda -n 10
        """
    )
    
    parser.add_argument('--model', '--model_path', type=str, 
                       default='checkpoints/dit_best.pt',
                       help='Chemin vers le checkpoint du modèle DiT')
    parser.add_argument('-n', '--num_samples', type=int, default=3,
                       help='Nombre d\'échantillons à générer (défaut: 3)')
    parser.add_argument('-s', '--steps', '--num_steps', type=int, default=50,
                       help='Steps de diffusion, plus = meilleur (défaut: 50)')
    parser.add_argument('--eta', type=float, default=0.0,
                       help='Stochasticity DDIM: 0=déterministe, 1=DDPM (défaut: 0.0)')
    parser.add_argument('-o', '--output', '--output_dir', type=str, 
                       default='generated_music_dit',
                       help='Dossier de sortie (défaut: generated_music_dit)')
    parser.add_argument('--normalize', '--method', type=str, default='clip',
                       choices=['clip', 'minmax', 'standard'],
                       help='Méthode de normalisation (défaut: clip)')
    parser.add_argument('--no-images', action='store_true',
                       help='Ne pas sauvegarder les images de spectrogrammes')
    parser.add_argument('--seed', type=int, default=None,
                       help='Seed pour reproductibilité')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda', 'mps'],
                       help='Device à utiliser (défaut: cpu)')
    
    args = parser.parse_args()
    
    print("🎵" * 30)
    print("    GÉNÉRATION DE MUSIQUE - DIFFUSION TRANSFORMER")
    print("🎵" * 30)
    
    # Crée le générateur
    generator = MusicGenerator(
        model_path=args.model,
        device=args.device
    )
    
    # Génère
    generator.generate(
        output_dir=args.output,
        num_samples=args.num_samples,
        num_steps=args.steps,
        eta=args.eta,
        normalize_method=args.normalize,
        save_spectrograms=not args.no_images,
        seed=args.seed
    )
    
    print(f"\n🎧 Pour écouter:")
    print(f"   ls {args.output}/*.wav")
    print(f"   # Ou ouvrir le dossier:")
    print(f"   xdg-open {args.output}/  # Linux")
    print(f"   open {args.output}/      # macOS")
    
    print("\n⚠️  NOTE: Qualité audio dépend de l'entraînement du modèle.")
    print("   Pour de vrais résultats, entraînez 50-100 époques minimum.")
    print("\n💡 TIPS:")
    print("   - Augmentez --steps (100-200) pour meilleure qualité")
    print("   - Utilisez --eta 0.3-0.5 pour plus de variété")
    print("   - Utilisez --device cuda si vous avez un GPU")


if __name__ == "__main__":
    main()