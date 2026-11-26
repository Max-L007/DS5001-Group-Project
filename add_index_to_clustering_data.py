# Add an index column to the cut pulse wave data
import numpy as np

original_data = np.load('cut_pulse_wave_data.npy').astype(np.float64)
n = original_data.shape[0]

# Prepend an index column (values from 0 to n-1)
indices = np.arange(n).reshape(-1, 1)  # (n, 1)
original_data_with_index = np.hstack([indices, original_data])  # (n, signal_length+1)

print(f"Cut data shape: {original_data.shape} -> with index: {original_data_with_index.shape}")
print(f"dtype: {original_data_with_index.dtype}")

# Save the indexed data
np.save('cut_pulse_wave_data_with_index.npy', original_data_with_index)
print("Saved: cut_pulse_wave_data_with_index.npy")
