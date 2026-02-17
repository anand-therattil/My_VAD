import numpy as np
import soundfile as sf
import io
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from create_mfcc import extract_mfcc
from gmm import DiagonalGMM
from hmm import HMM

# -------------------------
# Configuration
# -------------------------

MAX_FILE_SIZE_MB = 10
EXPECTED_SR = 16000
FRAME_SHIFT = 0.01

MIN_SPEECH_DURATION = 0.20
MIN_SILENCE_DURATION = 0.10
HANGOVER = 0.15

# -------------------------
# Logging
# -------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VAD_SERVER")

app = FastAPI()


# -------------------------
# Load Models at Startup
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

logger.info("Models loaded successfully.")


# -------------------------
# Health Check
# -------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


# -------------------------
# Post Processing
# -------------------------

def post_process_segments(states):

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
        start_time = start_f * FRAME_SHIFT
        end_time = (end_f + 1) * FRAME_SHIFT
        segments.append([start_time, end_time])

    # Remove short speech
    segments = [
        seg for seg in segments
        if (seg[1] - seg[0]) >= MIN_SPEECH_DURATION
    ]

    if not segments:
        return []

    # Merge short silence
    merged = [segments[0]]
    for current in segments[1:]:
        prev = merged[-1]
        if current[0] - prev[1] < MIN_SILENCE_DURATION:
            prev[1] = current[1]
        else:
            merged.append(current)

    # Hangover
    for seg in merged:
        seg[1] += HANGOVER

    return merged


# -------------------------
# VAD Endpoint
# -------------------------

@app.post("/vad")
async def run_vad(file: UploadFile = File(...)):

    try:
        contents = await file.read()

        # File size guard
        size_mb = len(contents) / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            raise HTTPException(status_code=413,
                                detail="File too large")

        audio_buffer = io.BytesIO(contents)
        signal, sr = sf.read(audio_buffer)

        # Validate sampling rate
        if sr != EXPECTED_SR:
            raise HTTPException(
                status_code=400,
                detail=f"Expected {EXPECTED_SR} Hz audio"
            )

        # Convert stereo to mono if needed
        if len(signal.shape) == 2:
            signal = np.mean(signal, axis=1)

        # Feature extraction
        mfcc = extract_mfcc(signal, sr)
        mfcc = (mfcc - mean) / std

        log_speech = gmm_speech.score_samples(mfcc)
        log_noise = gmm_noise.score_samples(mfcc)

        emissions = np.vstack([log_noise, log_speech]).T
        states = hmm.viterbi(emissions)

        segments = post_process_segments(states)

        return JSONResponse(
            content={
                "segments": [
                    {"start": float(s), "end": float(e)}
                    for s, e in segments
                ]
            }
        )

    except Exception as e:
        logger.exception("VAD processing failed.")
        raise HTTPException(status_code=500, detail=str(e))
