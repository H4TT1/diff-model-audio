import librosa
import numpy as np
import pretty_midi

# ---------------- SETTINGS ----------------
wav_path = "processed_wavs_2/audio_01.wav"
midi_out = "piano_output.mid"

n_fft = 2048
hop_length = 512
min_freq = 80    # ~E2 (low piano limit)
max_freq = 1000  # ~B5 (upper piano limit)
velocity = 80
# ------------------------------------------


def freq_to_midi(freq):
    return int(np.round(librosa.hz_to_midi(freq)))


# Load audio
y, sr = librosa.load(wav_path, sr=None)

# Compute spectrogram
S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))

freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=hop_length)

# Create MIDI
midi = pretty_midi.PrettyMIDI()
piano = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program("Acoustic Grand Piano"))

prev_note = None
note_start = 0.0

for t in range(S.shape[1]):
    spectrum = S[:, t]

    # Restrict frequency range
    valid = np.where((freqs >= min_freq) & (freqs <= max_freq))[0]
    idx = valid[np.argmax(spectrum[valid])]
    freq = freqs[idx]

    midi_note = freq_to_midi(freq)

    if prev_note is None:
        prev_note = midi_note
        note_start = times[t]

    elif midi_note != prev_note:
        note_end = times[t]
        note = pretty_midi.Note(
            velocity=velocity,
            pitch=prev_note,
            start=note_start,
            end=note_end
        )
        piano.notes.append(note)

        prev_note = midi_note
        note_start = times[t]

# Add final note
note = pretty_midi.Note(
    velocity=velocity,
    pitch=prev_note,
    start=note_start,
    end=times[-1]
)
piano.notes.append(note)

midi.instruments.append(piano)
midi.write(midi_out)

print("Piano MIDI saved to:", midi_out)
