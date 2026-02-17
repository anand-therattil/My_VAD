import numpy as np
import soundfile as sf
from fastapi import FastAPI, UploadFile, File
from create_mfcc import extract_mfcc
from gmm import DiagonalGMM
from hmm import HMM
import io

app = FastAPI()

# -------------------------
# Load models at startup
# -------------------------

def load_gmm(prefix):
    means = np.load(f"dataset/{prefix}_means.npy")
    vars_ = np.load(f"dataset/{prefix}_vars.npy")
    weights = np.load(f"dataset/{prefix}_weights.npy")

    gmm = DiagonalGMM(n_components=len(weights))
    gmm.means = means
    gmm.vars = vars_
    gmm.weights = weights
    gmm.D = means.shape[1]

    return gmm


mean = np.load("dataset/mean.npy")
std = np.load("dataset/std.npy")

gmm_speech = load_gmm("gmm_speech")
gmm_noise = load_gmm("gmm_noise")

A = np.array([[0.99, 0.01],
              [0.02, 0.98]])
pi = np.array([0.5, 0.5])
hmm = HMM(A, pi)


# -------------------------
# Post-processing
# -------------------------

def post_process_segments(states,
                          frame_shift=0.01,
                          min_speech_duration=0.20,
                          min_silence_duration=0.10,
                          hangover=0.15):

    raw_segments = []
    T = len(states)

    in_speech = False
    start_frame = 0

    for t in range(T):
        if states[t] == 1 and not in_speech:
            in_speech = True
            start_frame = t

        elif states[t] == 0 and in_speech:
            raw_segments.append((start_frame, t - 1))
            in_speech = False

    if in_speech:
        raw_segments.append((start_frame, T - 1))

    segments = []
    for start_f, end_f in raw_segments:
        start_time = start_f * frame_shift
        end_time = (end_f + 1) * frame_shift
        segments.append([start_time, end_time])

    # Remove short speech
    segments = [
        seg for seg in segments
        if (seg[1] - seg[0]) >= min_speech_duration
    ]

    if not segments:
        return []

    # Merge short silence gaps
    merged = [segments[0]]
    for current in segments[1:]:
        prev = merged[-1]
        if current[0] - prev[1] < min_silence_duration:
            prev[1] = current[1]
        else:
            merged.append(current)

    # Hangover
    for seg in merged:
        seg[1] += hangover

    return merged


# -------------------------
# API Endpoint
# -------------------------

@app.post("/vad")
async def run_vad(file: UploadFile = File(...)):

    contents = await file.read()
    audio_buffer = io.BytesIO(contents)
    signal, sr = sf.read(audio_buffer)

    # Extract MFCC
    mfcc = extract_mfcc(signal, sr)
    mfcc = (mfcc - mean) / std

    log_speech = gmm_speech.score_samples(mfcc)
    log_noise = gmm_noise.score_samples(mfcc)

    emissions = np.vstack([log_noise, log_speech]).T
    states = hmm.viterbi(emissions)

    segments = post_process_segments(states)

    return {
        "segments": [
            {"start": float(s), "end": float(e)}
            for s, e in segments
        ]
    }
