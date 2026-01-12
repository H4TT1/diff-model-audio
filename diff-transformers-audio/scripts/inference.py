import torch
import torch.nn.functional as F
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
import argparse
import json
from typing import Optional, Tuple, List
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.diffusion_model import MusicDiffusionModel
from models.unet import MelSpectrogramUNet

# Fonctions de conversion compatibles avec le preprocessing du notebook
def create_mel_spectrogram(audio_chunk, sr, n_mels=128):
    """Fonction identique au preprocessing pour cohérence"""
    # Compute Mel Spectrogram
    mel_spec = librosa.feature.melspectrogram(y=audio_chunk, sr=sr, n_mels=n_mels)
    
    # Convert power to decibels (log scale) - crucial for diffusion models
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Normalize to [-1, 1] for the U-Net (même normalisation que le preprocessing)
    normalized_spec = (log_mel_spec + 40) / 40  # Adjust based on your DB range
    return normalized_spec

def denormalize_mel_spectrogram(normalized_spec):
    """Inverse de la normalisation du preprocessing"""
    # Inverse: normalized = (log_mel + 40) / 40
    log_mel_spec = normalized_spec * 40 - 40
    return log_mel_spec

def mel_to_audio_notebook_style(mel_spec_normalized, sr=22050, duration_sec=5):
    """
    Convertit un mel-spectrogramme normalisé en audio
    Utilise la même approche que le preprocessing du notebook
    """
    # Dénormalisation
    log_mel_spec = denormalize_mel_spectrogram(mel_spec_normalized)
    
    # Conversion dB -> power
    mel_spec = librosa.db_to_power(log_mel_spec, ref=1.0)
    
    # Griffin-Lim pour reconstruction (approximation)
    # Note: pour une meilleure qualité, utiliser un vocoder neural
    audio = librosa.feature.inverse.mel_to_audio(
        mel_spec, 
        sr=sr, 
        n_fft=2048,
        hop_length=512,
        n_iter=32,
        length=int(duration_sec * sr)
    )
    
    return audio

class MusicGenerator:
    """Générateur de musique utilisant le modèle de diffusion entraîné"""
    
    def __init__(self, 
                 model_path: str,
                 device: str = 'auto'):
        
        # Device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        print(f"Utilisation du device: {self.device}")
        
        # Charge le modèle
        self.model = self.load_model(model_path)
        self.model.eval()
        
        # Paramètres audio correspondant au preprocessing du notebook
        self.sample_rate = 22050
        self.n_mels = 128
        self.n_fft = 2048
        self.hop_length = 512
        self.duration = 5.0  # secondes (du preprocessing notebook)
        
        print("Modèle chargé avec succès!")
    
    def load_model(self, model_path: str) -> MusicDiffusionModel:
        """Charge un modèle depuis un checkpoint"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Création du modèle avec les mêmes paramètres
        model = MusicDiffusionModel(
            input_channels=1,
            base_channels=64,  # À ajuster selon votre modèle
            timesteps=1000,
            schedule_type="cosine"
        ).to(self.device)
        
        # Charge les poids
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        return model
    
    @torch.no_grad()
    def generate(self, 
                 batch_size: int = 1,
                 num_inference_steps: int = 50,
                 eta: float = 0.0,
                 seed: Optional[int] = None,
                 return_intermediates: bool = False) -> torch.Tensor:
        """
        Génère de nouveaux mel-spectrogrammes
        
        Args:
            batch_size: Nombre d'échantillons à générer
            num_inference_steps: Nombre de pas de débruitage
            eta: Paramètre DDIM (0=déterministe, 1=stochastique)
            seed: Graine aléatoire
            return_intermediates: Retourner les étapes intermédiaires
        """
        if seed is not None:
            torch.manual_seed(seed)
            
        # Forme du mel-spectrogramme (correspondant au preprocessing notebook)
        # 128 mel bins x 216 frames temporelles (~5 sec à 22050 Hz avec hop_length=512)
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
        """
        Convertit un mel-spectrogramme en forme d'onde audio
        Utilise la même logique que le preprocessing du notebook
        """
        # Squeeze pour enlever les dimensions batch et channel si nécessaire
        if mel_spec.dim() == 4:
            mel_spec = mel_spec.squeeze(0)  # Remove batch
        if mel_spec.dim() == 3:
            mel_spec = mel_spec.squeeze(0)  # Remove channel
            
        # Convertit en numpy pour librosa
        mel_np = mel_spec.cpu().numpy()
        
        # Utilise la fonction de conversion du notebook
        audio = mel_to_audio_notebook_style(mel_np, sr=self.sample_rate, duration_sec=self.duration)
        
        return audio
    
    def generate_music(self,
                      output_path: str,
                      num_samples: int = 1,
                      num_inference_steps: int = 50,
                      eta: float = 0.0,
                      seed: Optional[int] = None,
                      save_spectrograms: bool = True) -> List[str]:
        """
        Génère de la musique et sauvegarde les fichiers audio
        
        Returns:
            Liste des chemins des fichiers générés
        """
        print(f"Génération de {num_samples} échantillons...")
        
        output_dir = Path(output_path)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        generated_files = []
        
        for i in range(num_samples):
            print(f"Génération échantillon {i+1}/{num_samples}")
            
            # Génération du spectrogramme
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
            
            mel_spec = spectrograms[0]  # Premier (et seul) échantillon
            
            # Conversion en audio
            audio = self.mel_to_waveform(mel_spec)
            
            # Normalisation audio
            audio = audio / np.max(np.abs(audio)) * 0.8  # Évite la saturation
            
            # Sauvegarde audio
            audio_path = output_dir / f"generated_music_{i:03d}.wav"
            sf.write(audio_path, audio, self.sample_rate)
            generated_files.append(str(audio_path))
            
            # Sauvegarde spectrogramme (optionnel)
            if save_spectrograms:
                spec_path = output_dir / f"spectrogram_{i:03d}.pt"
                torch.save(mel_spec.cpu(), spec_path)
                
                # Visualisation spectrogramme
                self.plot_spectrogram(mel_spec, output_dir / f"spectrogram_{i:03d}.png")
        
        print(f"Génération terminée! Fichiers sauvegardés dans {output_dir}")
        return generated_files
    
    def interpolate_music(self,
                         audio_path1: str,
                         audio_path2: str,
                         output_path: str,
                         num_steps: int = 10,
                         num_inference_steps: int = 50) -> List[str]:
        """
        Interpole entre deux morceaux de musique existants
        Utilise le même preprocessing que le notebook
        """
        print("Chargement des fichiers audio...")
        
        # Charge les audios avec librosa (même approche que le notebook)
        y1, sr1 = librosa.load(audio_path1, sr=self.sample_rate)
        y2, sr2 = librosa.load(audio_path2, sr=self.sample_rate)
        
        # Prend les 5 premières secondes (comme dans le preprocessing)
        samples_per_chunk = int(self.duration * self.sample_rate)
        y1 = y1[:samples_per_chunk]
        y2 = y2[:samples_per_chunk]
        
        # Conversion en spectrogrammes avec la fonction du notebook
        spec1 = create_mel_spectrogram(y1, self.sample_rate, self.n_mels)
        spec2 = create_mel_spectrogram(y2, self.sample_rate, self.n_mels)
        
        # Convertit en tensors PyTorch avec la bonne forme
        spec1 = torch.from_numpy(spec1).unsqueeze(0).unsqueeze(0).float().to(self.device)  # [1, 1, 128, 216]
        spec2 = torch.from_numpy(spec2).unsqueeze(0).unsqueeze(0).float().to(self.device)
        
        print(f"Interpolation en {num_steps} étapes...")
        
        # Interpolation dans l'espace latent
        interpolated_specs = self.model.interpolate(
            spec1, spec2, 
            num_steps=num_steps,
            t_start=500  # Point d'interpolation dans le processus de diffusion
        )
        
        # Conversion en audio et sauvegarde
        output_dir = Path(output_path)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        generated_files = []
        
        for i, spec in enumerate(interpolated_specs):
            audio = self.mel_to_waveform(spec[0])  # Retire la dimension batch
            audio = audio / np.max(np.abs(audio)) * 0.8
            
            audio_path = output_dir / f"interpolation_{i:03d}.wav"
            sf.write(audio_path, audio, self.sample_rate)
            generated_files.append(str(audio_path))
        
        print(f"Interpolation terminée! {len(generated_files)} fichiers créés")
        return generated_files
    
    def plot_spectrogram(self, mel_spec: torch.Tensor, save_path: Optional[str] = None):
        """Visualise un mel-spectrogramme"""
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
        plt.title('Mel Spectrogram Généré')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Génération de musique avec le modèle de diffusion')
    
    # Modèle
    parser.add_argument('--model_path', type=str, required=True,
                       help='Chemin vers le checkpoint du modèle')
    
    # Génération
    parser.add_argument('--output_dir', type=str, default='generated_music',
                       help='Répertoire de sortie')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Nombre d\'échantillons à générer')
    parser.add_argument('--num_inference_steps', type=int, default=50,
                       help='Nombre de pas d\'inférence')
    parser.add_argument('--eta', type=float, default=0.0,
                       help='Paramètre DDIM (0=déterministe)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Graine aléatoire')
    
    # Mode interpolation
    parser.add_argument('--interpolate', action='store_true',
                       help='Mode interpolation entre deux fichiers')
    parser.add_argument('--audio1', type=str,
                       help='Premier fichier audio pour l\'interpolation')
    parser.add_argument('--audio2', type=str,
                       help='Second fichier audio pour l\'interpolation')
    parser.add_argument('--num_interpolation_steps', type=int, default=10,
                       help='Nombre d\'étapes d\'interpolation')
    
    # Options
    parser.add_argument('--save_spectrograms', action='store_true',
                       help='Sauvegarder les spectrogrammes')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (auto, cuda, cpu)')
    
    args = parser.parse_args()
    
    # Vérification du modèle
    if not Path(args.model_path).exists():
        raise FileNotFoundError(f"Modèle introuvable: {args.model_path}")
    
    # Initialisation du générateur
    generator = MusicGenerator(
        model_path=args.model_path,
        device=args.device
    )
    
    # Mode interpolation
    if args.interpolate:
        if not args.audio1 or not args.audio2:
            raise ValueError("Les fichiers --audio1 et --audio2 sont requis pour l'interpolation")
            
        generated_files = generator.interpolate_music(
            audio_path1=args.audio1,
            audio_path2=args.audio2,
            output_path=args.output_dir,
            num_steps=args.num_interpolation_steps,
            num_inference_steps=args.num_inference_steps
        )
    else:
        # Mode génération standard avec les bonnes dimensions
        generated_files = generator.generate_music(
            output_path=args.output_dir,
            num_samples=args.num_samples,
            num_inference_steps=args.num_inference_steps,
            eta=args.eta,
            seed=args.seed,
            save_spectrograms=args.save_spectrograms
        )
    
    print(f"\nFichiers générés:")
    for file_path in generated_files:
        print(f"  {file_path}")

if __name__ == "__main__":
    main()