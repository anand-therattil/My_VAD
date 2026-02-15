import numpy as np
from gmm import DiagonalGMM


def main():

    X_speech = np.load("dataset/X_speech.npy")
    X_noise = np.load("dataset/X_noise.npy")

    print("Training Speech GMM...")
    gmm_speech = DiagonalGMM(n_components=8)
    gmm_speech.fit(X_speech)

    print("Training Noise GMM...")
    gmm_noise = DiagonalGMM(n_components=8)
    gmm_noise.fit(X_noise)

    np.save("dataset/gmm_speech_means.npy", gmm_speech.means)
    np.save("dataset/gmm_speech_vars.npy", gmm_speech.vars)
    np.save("dataset/gmm_speech_weights.npy", gmm_speech.weights)

    np.save("dataset/gmm_noise_means.npy", gmm_noise.means)
    np.save("dataset/gmm_noise_vars.npy", gmm_noise.vars)
    np.save("dataset/gmm_noise_weights.npy", gmm_noise.weights)

    print("GMM models saved.")


if __name__ == "__main__":
    main()
