import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import preprocessing

# Draft version - just to see the data
def plot_comparison(original, filtered, fs=512, channel_idx=0):
    """
    Plots time domain and frequency domain comparison for a single channel.
    """
    n = len(original)
    xf = np.fft.rfftfreq(n, 1/fs)
    
    plt.figure(figsize=(15, 10))
    
    # Time Domain
    plt.subplot(2, 1, 1)
    plt.plot(original[:, channel_idx], label='Raw', alpha=0.5, color='gray')
    plt.plot(filtered[:, channel_idx], label='Filtered', alpha=0.8, color='blue')
    plt.title(f"Time Domain Comparison (Channel {channel_idx+1})")
    plt.legend()
    plt.grid(True)
    
    # Frequency Domain
    plt.subplot(2, 1, 2)
    yf_raw = np.abs(np.fft.rfft(original[:, channel_idx]))
    yf_clean = np.abs(np.fft.rfft(filtered[:, channel_idx]))
    
    plt.plot(xf, yf_raw, label='Raw FFT', alpha=0.5, color='gray')
    plt.plot(xf, yf_clean, label='Filtered FFT', alpha=0.8, color='blue')
    plt.axvline(x=50, color='red', linestyle='--', label='50Hz Noise')
    
    plt.title(f"Frequency Domain Comparison (Channel {channel_idx+1})")
    plt.xlabel("Frequency (Hz)")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def load_and_view_signals(file_path):
    print(f"Loading: {file_path}")
    
    df = pd.read_csv(file_path)
    
    # Get raw numpy array (all 8 columns)
    raw_data = df.iloc[:, :8].values
    
    print("Applying Filters...")
    # 1. Notch Filter (50Hz)
    filtered_data = preprocessing.apply_notch_filter(raw_data)
    # 2. Bandpass Filter (20-200Hz)
    filtered_data = preprocessing.apply_bandpass_filter(filtered_data)
    
    print("Normalizing...")
    # 3. Z-score Normalization
    normalized_data = preprocessing.normalize_data(filtered_data)
    
    print("Plotting comparison for Channel 1...")
    # Comparing Filtered vs Normalized to see the scale change
    plot_comparison(filtered_data, normalized_data, channel_idx=0)

if __name__ == "__main__":
    # Just for testing
    import sys
    if len(sys.argv) > 1:
        load_and_view_signals(sys.argv[1])
    else:
        print("Pass a CSV file path to run.")
