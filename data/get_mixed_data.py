import os
import numpy as np
import soundfile as sf
import random

SPEECH_DIR = "speech/wav"
NOISE_DIR = "noise/wav"
OUT_DIR = "mixed_10db"
TARGET_SNR = 10  # dB

os.makedirs(OUT_DIR, exist_ok=True)

speech_files = sorted(os.listdir(SPEECH_DIR))
noise_files = sorted(os.listdir(NOISE_DIR))

for i, speech_file in enumerate(speech_files):
    speech_path = os.path.join(SPEECH_DIR, speech_file)
    noise_file = random.choice(noise_files)
    noise_path = os.path.join(NOISE_DIR, noise_file)

    speech, sr = sf.read(speech_path)
    noise, _ = sf.read(noise_path)

    # Make noise same length
    if len(noise) < len(speech):
        repeat = int(np.ceil(len(speech) / len(noise)))
        noise = np.tile(noise, repeat)

    noise = noise[:len(speech)]

    # Compute powers
    speech_power = np.mean(speech**2)
    noise_power = np.mean(noise**2)

    # Compute scaling factor
    alpha = np.sqrt(speech_power / (noise_power * 10**(TARGET_SNR/10)))

    noise_scaled = alpha * noise
    mixed = speech + noise_scaled

    # Prevent clipping
    mixed = mixed / np.max(np.abs(mixed))

    out_path = os.path.join(OUT_DIR, f"mix_{i:04d}.wav")
    sf.write(out_path, mixed, sr)

print("Done creating 10 dB mixtures.")
