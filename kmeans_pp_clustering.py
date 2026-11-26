import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy import stats
import os

plt.rcParams['axes.unicode_minus'] = False

N_CLUSTERS = 6

# 1. Load the data
# Load the raw pulse wave signals (1000, 1501) - the first column is the index
wave_data = np.load('cut_pulse_wave_data_with_index.npy')
print(f"Raw pulse wave data shape: {wave_data.shape}   dtype={wave_data.dtype}")

# Load the feature matrix (1000, 22) - the first column is the index
feature_data = np.load('feature_matrix.npy')
print(f"Feature matrix shape: {feature_data.shape}   dtype={feature_data.dtype}")

# 2. Verify that the indices are consistent
wave_indices = wave_data[:, 0].astype(int)
feature_indices = feature_data[:, 0].astype(int)

if not np.array_equal(wave_indices, feature_indices):
    # If the indices do not match, re-align the two arrays
    wave_data = wave_data[np.argsort(wave_data[:, 0])]
    feature_data = feature_data[np.argsort(feature_data[:, 0])]
    wave_indices = wave_data[:, 0].astype(int)
    feature_indices = feature_data[:, 0].astype(int)
    assert np.array_equal(wave_indices, feature_indices), "The indices cannot be aligned!"

print("Index check passed, the data is aligned.")

# 3. Data preprocessing
# Separate the index from the features
wave_signals = wave_data[:, 1:]  # raw pulse wave signals (1000, 1500)
features = feature_data[:, 1:]  # feature data (1000, 21)

# Standardise the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# 4. K-means++ clustering
kmeans = KMeans(
    n_clusters=N_CLUSTERS,
    init='k-means++',  # use k-means++ initialisation
    n_init=10,  # run 10 times and keep the best result
    random_state=42  # make the result reproducible
)
cluster_labels = kmeans.fit_predict(features_scaled)

# 5. Analyse the clustering result
print("\nClustering result summary:")
print(f"Silhouette score: {metrics.silhouette_score(features_scaled, cluster_labels):.3f}")

# Create the output directory
os.makedirs('clustering_results', exist_ok=True)


def short_list(indices, head=12):
    """Only show the first few indices when there are many, to keep titles short."""
    if len(indices) <= head:
        return ', '.join(map(str, indices))
    return ', '.join(map(str, indices[:head])) + f' ... ({len(indices)} total)'


# Output the sample indices of every cluster
cluster_indices = {}
for cluster_id in range(N_CLUSTERS):
    # Sample indices belonging to the current cluster
    cluster_mask = (cluster_labels == cluster_id)
    indices = wave_indices[cluster_mask]
    cluster_indices[cluster_id] = indices

    # Print the cluster summary
    count = len(indices)
    print(f"\nCluster {cluster_id}: {count} samples ({count / len(cluster_labels) * 100:.1f}%)")
    print(f"Sample indices: {short_list(indices, head=20)}")

    # Save the cluster indices to a file (inside the results folder)
    with open(f'clustering_results/cluster_{cluster_id}_sample_indices.txt', 'w', encoding='utf-8') as f:
        f.write(f"Cluster {cluster_id} contains {count} samples:\n")
        f.write(', '.join(map(str, indices)))

# 6. Plot the first three waveforms of every cluster (one waveform per row)
for cluster_id in range(N_CLUSTERS):
    indices = cluster_indices[cluster_id]
    cluster_signals = wave_signals[cluster_labels == cluster_id]

    # Number of waveforms to plot (at most 3)
    num_signals = min(3, len(cluster_signals))
    if num_signals == 0:
        print(f"Cluster {cluster_id} has no samples to plot")
        continue

    # Create the figure
    plt.figure(figsize=(10, 3 * num_signals))
    time = np.arange(wave_signals.shape[1]) / 100  # sampling rate 100 Hz

    for i in range(num_signals):
        plt.subplot(num_signals, 1, i + 1)
        plt.plot(time, cluster_signals[i])
        plt.title(f'Cluster {cluster_id} - Pulse Wave of Sample {indices[i]}')
        plt.xlabel('Sample Point')
        plt.ylabel('Signal Amplitude')
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'clustering_results/cluster_{cluster_id}_sample_waveforms.png', dpi=300)
    plt.close()


# 7. Safe helper (handles empty arrays)
def safe_mean_ci(signals):
    """Compute the mean signal and its confidence interval safely."""
    if len(signals) == 0:
        return np.array([]), np.array([]), np.array([])

    mean_signal = np.nanmean(signals, axis=0)

    # Confidence interval (needs at least 2 samples)
    if len(signals) > 1:
        sem = stats.sem(signals, axis=0, nan_policy='omit')
        ci = 1.96 * sem
    else:
        ci = np.zeros_like(mean_signal)

    return mean_signal, ci


# 8. Plot the mean signal of every cluster for comparison (one subplot per cluster)
plt.figure(figsize=(15, 18))  # 6 rows, 1 column
time = np.arange(wave_signals.shape[1]) / 100  # sampling rate 100 Hz

for cluster_id in range(N_CLUSTERS):
    cluster_signals = wave_signals[cluster_labels == cluster_id]
    indices = cluster_indices[cluster_id]

    # Compute the mean signal and confidence interval safely
    mean_signal, ci = safe_mean_ci(cluster_signals)

    # Skip clusters without data
    if len(mean_signal) == 0:
        continue

    # One subplot per cluster
    plt.subplot(N_CLUSTERS, 1, cluster_id + 1)

    # Plot the mean signal
    plt.plot(time, mean_signal, 'b-', linewidth=2, label=f'Mean Signal (n={len(cluster_signals)})')

    # Plot the confidence interval (needs at least 2 samples)
    if len(cluster_signals) > 1:
        plt.fill_between(time, mean_signal - ci, mean_signal + ci, color='b', alpha=0.2)

    plt.title(f'Mean Pulse Wave of Cluster {cluster_id} (n={len(indices)}, samples: {short_list(indices, head=8)})')
    plt.xlabel('Sample Point')
    plt.ylabel('Signal Amplitude')
    plt.grid(True)
    plt.legend()

plt.tight_layout()
plt.savefig('clustering_results/cluster_mean_signals_comparison.png', dpi=300)
plt.close()

# 9. Save the clustering result (indices and cluster labels)
results = np.column_stack([wave_indices, cluster_labels])
np.save('clustering_results.npy', results)
print("\nClustering result saved as 'clustering_results.npy'")
print("Analysis complete! All results were saved to the 'clustering_results' folder")
