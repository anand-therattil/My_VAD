# 🎙 Classical VAD Server (MFCC + GMM + HMM)

A production-ready Voice Activity Detection (VAD) system built from
scratch using:

-   Manual MFCC feature extraction
-   Diagonal Gaussian Mixture Models (EM algorithm)
-   2-state Hidden Markov Model (Viterbi decoding)
-   Post-processing (minimum duration + hangover)
-   FastAPI production server
-   Stress-tested REST API

------------------------------------------------------------------------

## 🚀 Features

-   MFCC implementation from scratch (no librosa)
-   Diagonal covariance GMM (custom EM implementation)
-   2-state HMM with Viterbi decoding
-   Speech segment extraction with duration filtering
-   Hangover extension
-   FastAPI REST API
-   Gunicorn production deployment
-   Stress-tested concurrency
-   Health endpoint

------------------------------------------------------------------------

## 🧠 System Architecture

Audio\
→ Pre-emphasis\
→ Framing (25ms / 10ms hop)\
→ FFT → Power Spectrum\
→ Mel Filterbank\
→ Log\
→ DCT → MFCC\
→ GMM (Speech / Noise)\
→ Log Likelihood Ratio\
→ HMM (Viterbi)\
→ Post-processing\
→ Speech timestamps

------------------------------------------------------------------------

## 📁 Project Structure

    My_VAD/
    │
    ├── data/
    ├── dataset/
    │   ├── gmm_speech_*.npy
    │   ├── gmm_noise_*.npy
    │   ├── mean.npy
    │   ├── std.npy
    │
    ├── create_mfcc.py
    ├── create_dataset.py
    ├── train_gmm.py
    ├── gmm.py
    ├── hmm.py
    ├── test_gmm_vad.py
    ├── vad_server.py
    ├── vad_server_prod.py
    ├── stress_test_client.py
    ├── vad_client.py
    └── README.md

------------------------------------------------------------------------

## 🛠 Installation

``` bash
python -m venv my-vad
source my-vad/bin/activate
pip install -e .
```

------------------------------------------------------------------------

## 🏋️ Training Pipeline

### 1️⃣ Create Dataset

``` bash
python create_dataset.py
```

### 2️⃣ Train GMM Models

``` bash
python train_gmm.py
```

### 3️⃣ Offline Testing

``` bash
python test_gmm_vad.py
```

------------------------------------------------------------------------

## 🌐 Run Production Server

### Development

``` bash
uvicorn vad_server_prod:app --host 0.0.0.0 --port 8000
```

### Production (Recommended)

``` bash
gunicorn -k uvicorn.workers.UvicornWorker vad_server_prod:app -w 4 -b 0.0.0.0:8000
```

------------------------------------------------------------------------

## 🔎 API Endpoints

### Health Check

GET /health

Response:

``` json
{"status": "ok"}
```

------------------------------------------------------------------------

### Run VAD

POST /vad\
Upload 16kHz mono WAV file.

Response:

``` json
{
  "segments": [
    {"start": 0.24, "end": 1.82},
    {"start": 2.24, "end": 3.59}
  ]
}
```

------------------------------------------------------------------------

## 📊 Stress Testing

``` bash
python stress_test_client.py
```

Example:

-   50 requests
-   10 concurrent workers
-   \~170ms average latency
-   \~50+ RPS

------------------------------------------------------------------------

## ⚙️ Production Safeguards

-   File size limit (10MB)
-   Sampling rate validation (16kHz required)
-   Stereo to mono conversion
-   Structured error handling
-   Logging
-   Multi-worker support
-   Health endpoint

------------------------------------------------------------------------

## 📌 Limitations

-   Offline VAD only (no streaming)
-   16kHz audio required
-   Classical model (non-neural)
-   Performance degrades below 5 dB SNR

------------------------------------------------------------------------

## 🚀 Future Improvements

-   WebSocket streaming endpoint
-   Automatic resampling
-   Docker containerization
-   Kubernetes deployment
-   Neural VAD comparison

------------------------------------------------------------------------

