import os
import numpy as np
import soundfile as sf
from create_mfcc import extract_mfcc

# assuming extract_mfcc() already defined

def build_feature_matrix(folder_path):
    all_features = []

    files = sorted(os.listdir(folder_path))

    for f in files:
        path = os.path.join(folder_path, f)
        signal, sr = sf.read(path)

        mfcc = extract_mfcc(signal, sr)

        all_features.append(mfcc)

    return np.vstack(all_features)
