import numpy as np
from scipy import signal

def apply_notch_filter(data, fs=512, notch_freq=50.0, quality_factor=30.0):
    """
    Applies a notch filter to remove power line noise (e.g., 50Hz).
    """
    b, a = signal.iirnotch(notch_freq, quality_factor, fs)
    # Apply along the first axis (assuming data is [samples, channels] or 1D)
    # If data is pandas df, we might need to handle it, but scipy works on numpy arrays typically.
    filtered_data = signal.filtfilt(b, a, data, axis=0) 
    return filtered_data

def apply_bandpass_filter(data, fs=512, lowcut=20.0, highcut=250.0, order=4):
    """
    Butterworth bandpass filter to keep EMG frequencies (typically 20-450Hz, 
    but with 512Hz sampling, Nyquist is 256Hz, so highcut MUST be < 256).
    Let's try 20-200Hz to be safe for now.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    filtered_data = signal.filtfilt(b, a, data, axis=0)
    return filtered_data

