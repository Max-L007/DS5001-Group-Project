import os
import numpy as np
import pywt
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

plt.rcParams['axes.unicode_minus'] = False


def _find(name):
    """Look for a data file in the current directory and its parent directories."""
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(4):
        for cand in (d, os.path.join(d, 'pulse_wave_clustering_analysis')):
            p = os.path.join(cand, name)
            if os.path.exists(p):
                return p
        d = os.path.dirname(d)
    return name


# Wavelet denoising and baseline drift removal
def process_ppg_signal(signal, fs=100):
    """
    Apply wavelet denoising and baseline drift removal to a single PPG signal.

    Parameters:
        signal (np.array): input PPG signal (float64)
        fs (float): sampling rate in Hz, defaults to 100 Hz

    Returns:
        processed_signal (np.array): processed signal (float64)
    """
    # 0. Force double precision so a float32 array passed in from elsewhere
    #    cannot silently degrade accuracy
    signal = np.asarray(signal, dtype=np.float64)

    # 1. Parameters
    wavelet = 'db5'  # wavelet basis
    level = 7

    # 2. Wavelet decomposition
    coeffs = pywt.wavedec(signal, wavelet, level=level)

    # 3. Noise estimation and thresholding
    if len(coeffs) > 1:
        # Estimate noise from the finest detail coefficients
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        # Guard against a degenerate sigma of zero
        if sigma < 1e-6:
            sigma = np.std(coeffs[-1])

        # Universal threshold
        N = len(signal)
        threshold = sigma * np.sqrt(2 * np.log(N))

        # Soft-threshold the high-frequency detail coefficients
        denoised_coeffs = [coeffs[0]]  # keep the approximation coefficients
        for i in range(1, len(coeffs)):
            denoised_coeffs.append(pywt.threshold(coeffs[i], threshold, mode='soft'))
    else:
        denoised_coeffs = coeffs

    # 4. Remove baseline drift
    # Zero out the coarsest approximation coefficients
    denoised_coeffs[0] = np.zeros_like(denoised_coeffs[0])

    # 5. Reconstruct the signal
    processed_signal = pywt.waverec(denoised_coeffs, wavelet)

    # 6. Make sure the output length matches the input length
    if len(processed_signal) > len(signal):
        processed_signal = processed_signal[:len(signal)]
    elif len(processed_signal) < len(signal):
        processed_signal = np.pad(processed_signal, (0, len(signal) - len(processed_signal)),
                                   mode='edge')

    return processed_signal.astype(np.float64, copy=False)


def plot_signal_comparison(raw, processed, fs=100, title="PPG Signal Comparison"):
    """
    Plot the raw signal against the processed signal.
    """
    t = np.arange(len(raw)) / fs

    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(3, 1, figure=fig)

    # Raw signal
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, raw, label='Raw Signal', color='blue', alpha=0.7)
    ax1.set_title(f"{title} - Before")
    ax1.set_xlabel("Sample Point")
    ax1.set_ylabel("Signal Amplitude")
    ax1.legend()
    ax1.grid(True)

    # Processed signal
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(t, processed, label='Processed Signal', color='red', alpha=0.7)
    ax2.set_title(f"{title} - After")
    ax2.set_xlabel("Sample Point")
    ax2.set_ylabel("Signal Amplitude")
    ax2.legend()
    ax2.grid(True)

    # Overlay
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(t, raw, label='Raw Signal', color='blue', alpha=0.5)
    ax3.plot(t, processed, label='Processed Signal', color='red', alpha=0.8)
    ax3.set_title(f"{title} - Overlay")
    ax3.set_xlabel("Sample Point")
    ax3.set_ylabel("Signal Amplitude")
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    return fig


def process_ppg_batch(input_path, output_path, fs=100, show_plots=5):
    """
    Process PPG signals in batch.

    Parameters:
        input_path (str): path to the input .npy file
        output_path (str): path to the output .npy file
        fs (float): sampling rate in Hz
        show_plots (int): number of signal comparison figures to produce
    """
    # 1. Load the data (convert to float64)
    data = np.load(input_path).astype(np.float64)
    print(f"Input data shape: {data.shape}   dtype={data.dtype}")

    # 2. Validate the data shape
    if len(data.shape) != 2:
        raise ValueError("Input data must be a 2-D array (n_samples, signal_length)")

    n_samples, signal_length = data.shape
    processed_data = np.zeros(data.shape, dtype=np.float64)  # explicitly double precision

    # 3. Process every signal
    for i in range(n_samples):
        processed_data[i] = process_ppg_signal(data[i], fs)

        # Progress report (1000 signals, print every 100)
        if (i + 1) % 100 == 0 or i == 0 or i == n_samples - 1:
            print(f"Progress: {i + 1}/{n_samples} ({(i + 1) / n_samples * 100:.1f}%)")

    # 4. Save the processed data
    np.save(output_path, processed_data)
    print(f"Done! Saved to: {output_path}   dtype={processed_data.dtype}")

    # 5. Visualise a few processed signals
    plot_indices = np.linspace(0, n_samples - 1, min(show_plots, n_samples), dtype=int)

    for idx in plot_indices:
        fig = plot_signal_comparison(data[idx], processed_data[idx], fs,
                                     title=f"PPG{idx + 1} Signal Comparison")
        fig.savefig(f"ppg_signal_before_after_{idx + 1}.png")
        plt.close(fig)

    return processed_data


# Usage example
if __name__ == "__main__":
    # Input and output file paths
    input_file = 'cut_pulse_wave_data.npy'      # 1000 cut pulse wave signals
    output_file = 'denoised_pulse_wave_data.npy'  # 1000 denoised pulse wave signals

    # Run the processing (produce comparison figures for 5 signals)
    processed_data = process_ppg_batch(input_file, output_file, fs=100, show_plots=5)
