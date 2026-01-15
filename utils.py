import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Draft version - just to see the data
def plot_frequency_spectrum(signals, fs=512):
    n = len(signals)
    # frequency resolution
    xf = np.fft.rfftfreq(n, 1/fs)
    
    plt.figure(figsize=(15, 10))
    
    for col in signals.columns:
        # data, n points
        yf = np.fft.rfft(signals[col].values)
        magnitude = np.abs(yf)
        plt.plot(xf, magnitude, label=col, alpha=0.7)
        
    plt.title("Frequency Spectrum (Raw)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.grid(True)
    # Mark 50Hz just to see
    plt.axvline(x=50, color='r', linestyle='--', label='50Hz')
    plt.show()

def load_and_view_signals(file_path):
    print(f"Loading: {file_path}")
    
    # The file has a header "ch1,ch2...". 
    df = pd.read_csv(file_path)
    
    print("DEBUG: HEAD of data:")
    print(df.head())
    
    # Plotting first 8 columns
    signals = df.iloc[:, :8]
    
    # Time domain
    signals.plot(subplots=True, figsize=(15, 10), title="Time Domain")
    plt.show()
    
    # Frequency domain
    print("Plotting FFT...")
    plot_frequency_spectrum(signals)

if __name__ == "__main__":
    # Just for testing
    import sys
    if len(sys.argv) > 1:
        load_and_view_signals(sys.argv[1])
    else:
        print("Pass a CSV file path to run.")
