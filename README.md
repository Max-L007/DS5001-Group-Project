# DS5001-Group-Project
Group Project of DS5001 Team at City University of Hong Kong (Dongguan)
# Project Overview

This project performs clustering analysis on pulse wave data.

It denoises the pulse wave data and removes baseline drift, extracts features, determines the optimal number of clusters, and then runs the clustering analysis.

The pipeline works on 1000 pulse wave signals, each 1500 samples long (sampling rate 100 Hz). It extracts 21 time-domain and frequency-domain features and finally splits the data into 6 clusters with K-Means++, together with a comparison of the two initialisation strategies and a ranking of how much each feature influences the result.

# Dependencies

The project is built and debugged on Python 3.12 with PyCharm 2025.3.3.

Required packages: see the image file `软件包.png` (installed packages) in the `README` help folder.

The main dependencies are `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `pywavelets` and `kneed`; everything else is a transitive dependency of those.

# Directory Structure

```
-- Environment and Report/                                    // help folder (软件包.png, project documents)
-- raw_data.npy                               // raw data used by the project
-- cut_raw_data.py                            // extract the part of the raw data that is needed
-- add_index_to_clustering_data.py            // add an index column to the raw data, before processing
-- denoise_and_remove_baseline_drift.py       // process the raw data (wavelet denoising + baseline drift removal)
-- time_frequency_domain_analysis.py          // feature extraction (9 time-domain + 12 frequency-domain features)
-- elbow_method_cluster_count.py              // elbow method to determine the optimal number of clusters
-- kmeans_pp_clustering.py                    // clustering analysis (K-Means++)
-- compare_kmeans_vs_kmeans_pp_iterations.py  // K-Means vs K-Means++ initialisation comparison
-- feature_influence_ranking.py               // feature influence ranking
```

Files generated while the pipeline runs:

```
-- cut_pulse_wave_data.npy                    // cut pulse wave data (1000, 1500)
-- cut_pulse_wave_data_with_index.npy         // cut data with an index column (1000, 1501)
-- denoised_pulse_wave_data.npy               // denoised data with baseline drift removed (1000, 1500)
-- feature_matrix.npy                         // feature matrix with the index column (1000, 22)
-- feature_matrix_feature_names.npy           // feature names (including sample_index)
-- elbow_curve_basic.png                      // basic elbow curve
-- elbow_curve_with_optimal_k.png             // elbow curve with the optimal K marked
-- feature_distributions.png                  // histograms of the 21 features
-- ppg_signal_before_after_*.png              // before/after denoising (5 sampled signals)
-- clustering_results.npy                     // clustering result (sample index + cluster label)
-- clustering_results/                        // clustering result folder
     cluster_{i}_sample_indices.txt           // sample indices of cluster i
     cluster_{i}_sample_waveforms.png         // first three waveforms of cluster i
     cluster_mean_signals_comparison.png       // mean signal of every cluster (95% confidence band)
     init_method_comparison.txt/.csv/.png     // comparison of the two initialisation strategies
     feature_influence_ranking.txt/.csv/.png  // feature influence ranking
```

# Usage

All scripts read their input through relative paths, so set the working directory to the project root and run the scripts in the following order:

1. `cut_raw_data.py` — take all 1000 signals of group 0, channel 0 from `raw_data.npy` and write `cut_pulse_wave_data.npy`
2. `add_index_to_clustering_data.py` — prepend an index column (0-999) and write `cut_pulse_wave_data_with_index.npy`
3. `denoise_and_remove_baseline_drift.py` — 7-level db5 wavelet decomposition, soft-threshold denoising and zeroing of the approximation coefficients to remove the baseline drift, writing `denoised_pulse_wave_data.npy`
4. `time_frequency_domain_analysis.py` — db6 wavelet decomposition, then time-domain and frequency-domain feature extraction, writing `feature_matrix.npy`
5. `elbow_method_cluster_count.py` — compute the SSE for K = 1-10 and locate the elbow automatically; the optimal K is 6
6. `kmeans_pp_clustering.py` — cluster with K = 6, `init='k-means++'`, `n_init=10`, `random_state=42`, writing to `clustering_results/`
7. `compare_kmeans_vs_kmeans_pp_iterations.py` — keep the same feature matrix and change only the initialisation, comparing the number of iterations and the final SSE
8. `feature_influence_ranking.py` — permutation importance to measure how much each of the 21 features influences the clustering

Notes:

- Steps 5-8 only need `feature_matrix.npy` produced by step 4; they cannot run before the first four steps are done.
- Steps 1-4 involve a wavelet transform of 1000 signals and take a while; steps 5-8 take a few seconds.
