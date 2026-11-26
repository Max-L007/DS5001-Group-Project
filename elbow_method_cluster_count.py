import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from kneed import KneeLocator

plt.rcParams['axes.unicode_minus'] = False

# 1. Load the data and separate the index column from the features
data_with_index = np.load('feature_matrix.npy')  # shape (1000, 22)
print(f"Raw data shape (with index): {data_with_index.shape}   dtype={data_with_index.dtype}")

# Separate the index (first column) and the features (remaining columns)
sample_indices = data_with_index[:, 0]  # index column
features = data_with_index[:, 1:]       # feature data
print(f"Feature data shape: {features.shape}")  # (1000, 21)

# 2. Standardise the data
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
print(f"Scaled dtype: {features_scaled.dtype}")

# 3. Define the range of K and compute the SSE
k_range = range(1, 11)  # test 1 to 10 clusters
sse = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')  # n_init updated
    kmeans.fit(features_scaled)
    sse.append(kmeans.inertia_)

print("SSE series:", ", ".join(f"{v:.1f}" for v in sse))

# 4. Plot the elbow curve (saved as PNG)
plt.figure(figsize=(10, 6))
plt.plot(k_range, sse, 'bo-', markersize=8, linewidth=2)
plt.xlabel('Number of Clusters (K)', fontsize=12)
plt.ylabel('Within-Cluster Sum of Squares (SSE)', fontsize=12)
plt.title('Elbow Curve (Features Only, n=1000, float64)', fontsize=14)
plt.xticks(k_range)
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('elbow_curve_basic.png', dpi=300, bbox_inches='tight')
plt.close()  # close the figure to avoid memory leaks

# 5. Detect the optimal K automatically
kl = KneeLocator(
    k_range,
    sse,
    curve='convex',
    direction='decreasing'
)
optimal_k = kl.elbow
print(f"Optimal number of clusters: {optimal_k}")

# 6. Plot the curve with the elbow point marked (saved as PNG)
plt.figure(figsize=(10, 6))
plt.plot(k_range, sse, 'bo-', markersize=8, linewidth=2)
plt.vlines(optimal_k, plt.ylim()[0], plt.ylim()[1],
           colors='r', linestyles='dashed', linewidth=2,
           label=f'Optimal K = {optimal_k}')
plt.xlabel('Number of Clusters (K)', fontsize=12)
plt.ylabel('Within-Cluster Sum of Squares (SSE)', fontsize=12)
plt.title(f'Elbow Curve (Optimal K={optimal_k}, n=1000, float64)', fontsize=14)
plt.xticks(k_range)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('elbow_curve_with_optimal_k.png', dpi=300, bbox_inches='tight')
plt.close()

print("Analysis complete. The following figures were saved:")
print("- elbow_curve_basic.png")
print("- elbow_curve_with_optimal_k.png")
