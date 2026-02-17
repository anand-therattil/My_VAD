import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

from create_mfcc import extract_mfcc
from gmm import DiagonalGMM
from hmm import HMM
from extract_segments import extract_segments, post_process_segments

# -----------------------------
# Load GMM From Saved Params
# -----------------------------

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


# -----------------------------
# Compute LLR
# -----------------------------

def compute_llr(signal, sr, gmm_speech, gmm_noise, mean, std):

    mfcc = extract_mfcc(signal, sr)

    # Normalize
    mfcc = (mfcc - mean) / std

    log_speech = gmm_speech.score_samples(mfcc)
    log_noise = gmm_noise.score_samples(mfcc)

    llr = log_speech - log_noise

    return llr


# -----------------------------
# Main Test
# -----------------------------
def main():

    # Load normalization
    mean = np.load("dataset/mean.npy")
    std = np.load("dataset/std.npy")

    # Load GMM models
    gmm_speech = load_gmm("gmm_speech")
    gmm_noise = load_gmm("gmm_noise")

    # Choose test file
    test_path = "data/mixed_10db/mix_0000.wav"
    # test_path = "data/speech/wav/7367-86737-0000.wav"
    # test_path = "data/noise/wav/noise_0.wav"

    signal, sr = sf.read(test_path)

    # -----------------------------
    # Extract + Normalize Once
    # -----------------------------
    mfcc = extract_mfcc(signal, sr)
    mfcc = (mfcc - mean) / std

    # -----------------------------
    # Compute Log Likelihoods
    # -----------------------------
    log_speech = gmm_speech.score_samples(mfcc)
    log_noise = gmm_noise.score_samples(mfcc)

    llr = log_speech - log_noise

    # -----------------------------
    # Plot LLR
    # -----------------------------
    frame_shift = 0.01
    time = np.arange(len(llr)) * frame_shift

    # plt.figure()
    # plt.plot(time, llr)
    # plt.axhline(0)
    # plt.title("Log-Likelihood Ratio (Speech - Noise)")
    # plt.xlabel("Time (seconds)")
    # plt.ylabel("LLR")
    # plt.tight_layout()
    # plt.savefig("llr_plot.png", dpi=300)
    # plt.close()

    print("LLR plot saved as llr_plot.png")
    print("LLR Mean:", np.mean(llr))
    print("LLR Min:", np.min(llr))
    print("LLR Max:", np.max(llr))

    # -----------------------------
    # HMM Emissions
    # -----------------------------
    emissions = np.vstack([log_noise, log_speech]).T

    A = np.array([[0.99, 0.01],
                  [0.02, 0.98]])

    pi = np.array([0.5, 0.5])

    hmm = HMM(A, pi)
    states = hmm.viterbi(emissions)
    segments = extract_segments(states)
    print("\nDetected Speech Segments:")
    for s, e in segments:
        print(f"Start: {s:.2f}s  End: {e:.2f}s  Duration: {e-s:.2f}s")
        
    print("\nPost-Processed Segments:")
    segments = post_process_segments(states)
    print("\nDetected Speech Segments:")
    for s, e in segments:
        print(f"Start: {s:.2f}s  End: {e:.2f}s  Duration: {e-s:.2f}s")

    # -----------------------------
    # Plot HMM result
    # -----------------------------
    # plt.figure()
    # plt.plot(time, llr, label="LLR")
    # plt.plot(time, states * np.max(llr), linestyle="--", label="HMM State")
    # plt.axhline(0)
    # plt.legend()
    # plt.title("LLR and HMM Decoded States")
    # plt.xlabel("Time (seconds)")
    # plt.ylabel("LLR / State")
    # plt.tight_layout()
    # plt.savefig("hmm_vad_plot.png", dpi=300)
    # plt.close()

    # print("HMM VAD plot saved as hmm_vad_plot.png")


if __name__ == "__main__":
    main()
