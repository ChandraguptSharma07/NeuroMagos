import numpy as np
from scipy import signal

def notch(d, fs=512, freq=50.0):
    b, a = signal.iirnotch(freq, 30.0, fs)
    return signal.filtfilt(b, a, d, axis=0) 

def bandpass(d, fs=512, l=20.0, h=250.0):
    nyq = 0.5 * fs
    b, a = signal.butter(4, [l/nyq, h/nyq], btype='band')
    return signal.filtfilt(b, a, d, axis=0)

def norm(d):
    mu = np.mean(d, axis=0)
    std = np.std(d, axis=0)
    std[std == 0] = 1.0
    return (d - mu) / std

def get_spec(d, fs=512):
    # [Time, 8] -> [8, Time]
    d = d.T
    specs = []
    
    for i in range(8):
        f, t, Zxx = signal.stft(d[i], fs=fs, nperseg=64, noverlap=32)
        specs.append(20 * np.log10(np.abs(Zxx) + 1e-6))
        
    s = np.array(specs)
    
    # global norm
    mu, std = s.mean(), s.std()
    return (s - mu) / (std if std != 0 else 1.0)
