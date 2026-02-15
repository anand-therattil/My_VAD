import numpy as np 
from scipy.fftpack import dct

def pre_emphasis(signal, coeff=0.97):
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])


def framing(signal, frame_size=400, hop_size=160):
    num_frames = 1 + int((len(signal) - frame_size) / hop_size)
    frames = np.zeros((num_frames, frame_size))

    for i in range(num_frames):
        start = i * hop_size
        frames[i] = signal[start:start+frame_size]

    return frames


def apply_window(frames):
    hamming = np.hamming(frames.shape[1])
    return frames * hamming

def compute_fft(frames, NFFT=512):
    return np.fft.rfft(frames, NFFT)

def power_spectrum(fft_frames, NFFT=512):
    return (1.0 / NFFT) * (np.abs(fft_frames) ** 2)

def mel_filterbank(sr=16000, NFFT=512, n_mels=26):

    def hz_to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    def mel_to_hz(mel):
        return 700 * (10**(mel / 2595) - 1)

    low_mel = hz_to_mel(0)
    high_mel = hz_to_mel(sr / 2)

    mel_points = np.linspace(low_mel, high_mel, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    bins = np.floor((NFFT + 1) * hz_points / sr).astype(int)

    fbank = np.zeros((n_mels, int(NFFT/2 + 1)))

    for m in range(1, n_mels + 1):
        f_m_minus = bins[m - 1]
        f_m = bins[m]
        f_m_plus = bins[m + 1]

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)

        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)

    return fbank

def apply_mel_filterbank(power_frames, fbank):
    return np.dot(power_frames, fbank.T)

def log_mel(mel_energy):
    return np.log(mel_energy + 1e-10)



def compute_mfcc(log_mel_energy, num_ceps=13):
    mfcc = dct(log_mel_energy, type=2, axis=1, norm='ortho')
    return mfcc[:, :num_ceps]

def extract_mfcc(signal, sr=16000):
    signal = pre_emphasis(signal)
    frames = framing(signal)
    frames = apply_window(frames)
    fft_frames = compute_fft(frames)
    power_frames = power_spectrum(fft_frames)
    fbank = mel_filterbank(sr)
    mel_energy = apply_mel_filterbank(power_frames, fbank)
    log_energy = log_mel(mel_energy)
    mfcc = compute_mfcc(log_energy)
    return mfcc
