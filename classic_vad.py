import numpy as np
import wave
import json
from pathlib import Path

# -------- CONFIG --------
INPUT_PATH = "jackhammer.wav"
OUTPUT_JSON = "energy_vad_output.json"

FRAME_SIZE_MS = 25        # frame size in ms
HOP_SIZE_MS = 10          # hop size in ms
ENERGY_THRESHOLD_RATIO = 0.6  # % of max energy
MIN_SPEECH_DURATION = 0.2     # seconds
MERGE_GAP = 0.15              # seconds
# ------------------------


def load_wav(path):
    with wave.open(path, 'rb') as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        n_frames = wf.getnframes()
        audio = wf.readframes(n_frames)

    audio = np.frombuffer(audio, dtype=np.int16)

    # Stereo → mono
    if n_channels == 2:
        audio = audio.reshape(-1, 2)
        audio = audio.mean(axis=1)

    # Normalize to [-1, 1]
    audio = audio.astype(np.float32) / 32768.0

    return audio, sr


def frame_signal(signal, frame_size, hop_size):
    num_frames = 1 + int((len(signal) - frame_size) / hop_size)
    frames = np.lib.stride_tricks.as_strided(
        signal,
        shape=(num_frames, frame_size),
        strides=(signal.strides[0] * hop_size, signal.strides[0])
    )
    return frames


def compute_energy(frames):
    return np.sum(frames ** 2, axis=1)


def energy_vad(audio, sr):
    frame_size = int(sr * FRAME_SIZE_MS / 1000)
    hop_size = int(sr * HOP_SIZE_MS / 1000)

    frames = frame_signal(audio, frame_size, hop_size)
    energy = compute_energy(frames)

    threshold = ENERGY_THRESHOLD_RATIO * np.max(energy)
    speech_frames = energy > threshold

    segments = []
    start = None

    for i, is_speech in enumerate(speech_frames):
        if is_speech and start is None:
            start = i
        elif not is_speech and start is not None:
            end = i
            segments.append((start, end))
            start = None

    if start is not None:
        segments.append((start, len(speech_frames)))

    # Convert to seconds
    speech_segments = []
    for start_f, end_f in segments:
        start_sec = start_f * hop_size / sr
        end_sec = end_f * hop_size / sr
        speech_segments.append((start_sec, end_sec))

    return speech_segments


def merge_segments(segments):
    if not segments:
        return []

    merged = [segments[0]]

    for start, end in segments[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end < MERGE_GAP:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))

    # Remove very short segments
    merged = [
        (s, e) for s, e in merged
        if (e - s) >= MIN_SPEECH_DURATION
    ]

    return merged


def main():
    audio, sr = load_wav(INPUT_PATH)

    raw_segments = energy_vad(audio, sr)
    final_segments = merge_segments(raw_segments)

    output = {"energy_vad": {}}

    for idx, (start, end) in enumerate(final_segments, 1):
        output["energy_vad"][f"chunk_{idx}"] = {
            "start": round(start, 3),
            "end": round(end, 3)
        }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved VAD output to: {OUTPUT_JSON}")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
