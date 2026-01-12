import os
import librosa
import soundfile as sf
import numpy as np
from scipy.signal import butter, filtfilt

# --- SETTINGS ---
input_folder = "generated_music_dit"   # folder with your .wav files
output_folder = "processed_wavs"
os.makedirs(output_folder, exist_ok=True)

# Low-pass filter helper
def lowpass_filter(signal, sr, cutoff=2000):
    nyq = 0.5 * sr
    normal_cutoff = cutoff / nyq
    b, a = butter(N=4, Wn=normal_cutoff, btype='low', analog=False)
    filtered = filtfilt(b, a, signal)
    return filtered

# Process each .wav file
for filename in os.listdir(input_folder):
    if filename.endswith(".wav"):
        path_in = os.path.join(input_folder, filename)
        path_out = os.path.join(output_folder, filename)

        # Load audio
        y, sr = librosa.load(path_in, sr=None)  # keep original sr

        # --- Apply transformations ---
        # 1. Low-pass filter to smooth noise
        y_smooth = lowpass_filter(y, sr, cutoff=2000)

        # 2. Optional: normalize amplitude
        y_smooth = y_smooth / (np.max(np.abs(y_smooth)) + 1e-9)

        # 3. Optional: apply a mild fade in/out to reduce clicks
        fade_len = int(0.01 * sr)  # 10ms fade
        y_smooth[:fade_len] *= np.linspace(0, 1, fade_len)
        y_smooth[-fade_len:] *= np.linspace(1, 0, fade_len)

        # Save processed audio
        sf.write(path_out, y_smooth, sr)
        print(f"Processed {filename} → {path_out}")
