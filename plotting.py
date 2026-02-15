import matplotlib.pyplot as plt
import soundfile as sf
import numpy as np
from create_mfcc import extract_mfcc

speech_path = "data/speech/wav/7367-86737-0000.wav"
noise_path = "data/noise/wav/noise_0.wav"

speech, sr = sf.read(speech_path)
noise, _ = sf.read(noise_path)

mfcc_speech = extract_mfcc(speech, sr)
mfcc_noise = extract_mfcc(noise, sr)

avg_speech = np.mean(mfcc_speech, axis=0)
avg_noise = np.mean(mfcc_noise, axis=0)

# -------------------------
# Plot Speech
# -------------------------

plt.figure()
plt.plot(avg_speech)
plt.title("Average MFCC - Speech")
plt.xlabel("Coefficient Index")
plt.ylabel("Value")
plt.tight_layout()
plt.savefig("speech_mfcc.png", dpi=300)
plt.close()

plt.figure()
plt.plot(avg_noise)
plt.title("Average MFCC - Noise")
plt.xlabel("Coefficient Index")
plt.ylabel("Value")
plt.tight_layout()
plt.savefig("noise_mfcc.png", dpi=300)
plt.close()
