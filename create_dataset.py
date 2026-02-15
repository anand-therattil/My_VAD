import numpy as np
import os

from extract_feature import build_feature_matrix


def main():

    print("Building Speech Features from mixed_10db...")
    X_speech = build_feature_matrix("data/mixed_10db")

    print("Building Noise Features from pure noise...")
    X_noise = build_feature_matrix("data/noise/wav")

    print("Speech shape:", X_speech.shape)
    print("Noise shape:", X_noise.shape)

    # -------------------------------------
    # Normalize (Global Mean/Std)
    # -------------------------------------

    print("Normalizing features...")

    X_all = np.vstack([X_speech, X_noise])

    mean = np.mean(X_all, axis=0)
    std = np.std(X_all, axis=0) + 1e-10  # avoid divide-by-zero

    X_speech = (X_speech - mean) / std
    X_noise = (X_noise - mean) / std

    # -------------------------------------
    # Create output folder
    # -------------------------------------

    os.makedirs("dataset", exist_ok=True)

    # -------------------------------------
    # Save everything
    # -------------------------------------

    np.save("dataset/X_speech.npy", X_speech)
    np.save("dataset/X_noise.npy", X_noise)
    np.save("dataset/mean.npy", mean)
    np.save("dataset/std.npy", std)

    print("Dataset saved successfully in ./dataset/")
    print("Done.")


if __name__ == "__main__":
    main()
