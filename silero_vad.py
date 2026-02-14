import torch
import torchaudio
import json
from pathlib import Path

INPUT_PATH = "jackhammer.wav"
OUTPUT_JSON = "silero_output.json"
TARGET_SAMPLE_RATE = 16000
VAD_THRESHOLD = 0.6


def load_audio(path):
    waveform, sample_rate = torchaudio.load(path)

    # Stereo → mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample to 16k
    if sample_rate != TARGET_SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=TARGET_SAMPLE_RATE
        )
        waveform = resampler(waveform)

    return waveform.squeeze(0), TARGET_SAMPLE_RATE


def main():
    print("Loading Silero VAD...")
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False
    )

    get_speech_timestamps = utils[0]

    print(f"Loading audio: {INPUT_PATH}")
    audio, sr = load_audio(INPUT_PATH)

    print("Running VAD...")
    speech_timestamps = get_speech_timestamps(
        audio,
        model,
        sampling_rate=sr,
        threshold=VAD_THRESHOLD
    )

    output = {"silero": {}}

    for idx, segment in enumerate(speech_timestamps, start=1):
        start_sec = round(segment["start"] / sr, 3)
        end_sec = round(segment["end"] / sr, 3)

        output["silero"][f"chunk_{idx}"] = {
            "start": start_sec,
            "end": end_sec
        }

    # Save JSON
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved VAD output to: {OUTPUT_JSON}")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
