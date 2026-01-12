import os
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# ---------------- SETTINGS ----------------
input_folder = "generated_music_dit"
output_folder = "processed_wavs_2"
spec_folder = "spectrograms_2"

os.makedirs(output_folder, exist_ok=True)
os.makedirs(spec_folder, exist_ok=True)

n_fft = 2048
hop_length = 512

# -----------------------------------------
def process_audio(y, sr):
    """
    Spectral smoothing + temporal smoothing
    """

    # STFT
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    mag, phase = np.abs(S), np.angle(S)

    # --- Spectral smoothing ---
    # Smooth magnitude in time + frequency
    mag_smooth = gaussian_filter(mag, sigma=(1.0, 2.0))

    # --- Spectral gating (soft) ---
    noise_floor = np.percentile(mag_smooth, 20)
    mag_denoised = np.maximum(mag_smooth - noise_floor, 0.0)

    # Reconstruct STFT
    S_denoised = mag_denoised * np.exp(1j * phase)

    # Inverse STFT
    y_out = librosa.istft(S_denoised, hop_length=hop_length)

    # --- Temporal smoothing ---
    y_out = gaussian_filter(y_out, sigma=2)

    # Normalize
    y_out = y_out / (np.max(np.abs(y_out)) + 1e-9)

    return y_out, mag_denoised


def save_spectrogram(mag, sr, filename):
    plt.figure(figsize=(8, 4))
    librosa.display.specshow(
        librosa.amplitude_to_db(mag, ref=np.max),
        sr=sr,
        hop_length=hop_length,
        y_axis="log",
        x_axis="time"
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title("Processed Spectrogram")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


# ---------------- MAIN LOOP ----------------
for file in os.listdir(input_folder):
    if file.endswith(".wav"):
        in_path = os.path.join(input_folder, file)
        out_path = os.path.join(output_folder, file)
        spec_path = os.path.join(spec_folder, file.replace(".wav", ".png"))

        # Load
        y, sr = librosa.load(in_path, sr=None)

        # Process
        y_out, mag = process_audio(y, sr)

        # Save audio
        sf.write(out_path, y_out, sr)

        # Save spectrogram
        save_spectrogram(mag, sr, spec_path)

        print(f"Processed: {file}")
