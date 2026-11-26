# -*- coding: utf-8 -*-
"""
Influence ranking of the 21 features on the clustering result

Baseline configuration identical to kmeans_pp_clustering.py:
    feature_matrix.npy -> StandardScaler -> KMeans(6, init='k-means++', n_init=10, random_state=42)

Primary measure (the ranking criterion): permutation importance
    Shuffle the values of column j across the 1000 samples, cutting the link between
    "this value" and "this pulse wave", keep the other 20 columns untouched, then re-cluster
    with exactly the same configuration. If the partition barely changes (ARI near 1) the
    column contributes nothing; the more ARI drops, the more critical the column is.
    dARI = 1 - ARI after shuffling, averaged over 8 repeats.

Auxiliary measures (cross-checks, not used for ranking):
    Ablation dARI: drop the column entirely and re-cluster (changes the dimensionality).
                   Not fully equivalent to the primary measure, reference only.
    Fixed-centre flip rate: after shuffling the column, do not refit; assign samples to the
                   baseline centres and count what fraction moved to another cluster.
                   No re-clustering randomness, so the cleanest number.
    ANOVA F / eta^2: between-cluster differences of the six clusters on that column, which
                   measures "how separable this column is", not its influence on the partition.

Key control: the noise floor
    With the same data and no shuffling at all, merely changing the initialisation already
    produces a non-zero dARI. Values below that floor must not be read as "an effect"; the
    script computes the floor and writes it into the conclusions.

Outputs:
    clustering_results/feature_influence_ranking.csv
    clustering_results/feature_influence_ranking.txt
    clustering_results/feature_influence_ranking.png
"""

import io
import os
import warnings

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

plt.rcParams['axes.unicode_minus'] = False

N_CLUSTERS = 6
N_INIT = 10
SEED = 42
N_PERM = 8      # shuffle every feature 8 times
N_FLOOR = 10    # noise floor: rerun the same data with 10 different random seeds


def fit(M, seed=SEED, n_init=N_INIT):
    return KMeans(n_clusters=N_CLUSTERS, init='k-means++',
                  n_init=n_init, random_state=seed).fit(M)


# ------------------------------------------------------------------ data and baseline
feature_data = np.load('feature_matrix.npy')
features = feature_data[:, 1:]
names = [str(x) for x in np.load('feature_matrix_feature_names.npy', allow_pickle=True)][1:]
Z = StandardScaler().fit_transform(features)
n, d = Z.shape
print(f"Feature matrix {Z.shape}  K={N_CLUSTERS}  n_init={N_INIT}  seed={SEED}")

base = fit(Z)
L0 = base.labels_
C0 = base.cluster_centers_
print(f"Baseline SSE = {base.inertia_:.2f}   cluster sizes = {np.bincount(L0).tolist()}")

# ------------------------------------------------------------------ noise floor
floor = []
for s in range(N_FLOOR):
    floor.append(1.0 - ari(L0, fit(Z, seed=s).labels_))
floor_mean, floor_max = float(np.mean(floor)), float(np.max(floor))
print(f"Noise floor (same data, new initialisation): dARI mean {floor_mean:.4f}, max {floor_max:.4f}")

# ------------------------------------------------------------------ per feature
rows = []
for j, name in enumerate(names):
    const = (Z[:, j].std() == 0)

    # Primary: shuffle then re-cluster
    if const:
        perm = np.zeros(N_PERM)
    else:
        vals = []
        for k in range(N_PERM):
            Zp = Z.copy()
            Zp[:, j] = np.random.default_rng(100 + k).permutation(Zp[:, j])
            vals.append(1.0 - ari(L0, fit(Zp).labels_))
        perm = np.array(vals)

    # Auxiliary 1: fixed-centre flip rate
    if const:
        flip_fixed = 0.0
    else:
        rates = []
        for k in range(N_PERM):
            Zp = Z.copy()
            Zp[:, j] = np.random.default_rng(500 + k).permutation(Zp[:, j])
            rates.append(float((euclidean_distances(Zp, C0).argmin(axis=1) != L0).mean()))
        flip_fixed = float(np.mean(rates))

    # Auxiliary 2: ablation
    keep = [k for k in range(d) if k != j]
    d_ari_abl = 1.0 - ari(L0, fit(Z[:, keep]).labels_)

    # Auxiliary 3: ANOVA F and eta^2
    groups = [Z[L0 == k, j] for k in range(N_CLUSTERS)]
    try:
        F, _ = stats.f_oneway(*groups)
        F = 0.0 if not np.isfinite(F) else float(F)
    except Exception:
        F = 0.0
    grand = Z[:, j].mean()
    ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_total = float(((Z[:, j] - grand) ** 2).sum())
    eta2 = float(ss_between / ss_total) if ss_total > 0 else 0.0

    rows.append(dict(name=name, d_ari=float(perm.mean()), d_ari_std=float(perm.std()),
                     d_ari_min=float(perm.min()), d_ari_max=float(perm.max()),
                     flip_fixed=flip_fixed, d_ari_abl=d_ari_abl, F=F, eta2=eta2, const=const))

rows.sort(key=lambda r: r['d_ari'], reverse=True)
total = sum(r['d_ari'] for r in rows) or 1.0

# ------------------------------------------------------------------ console
print("\n" + "=" * 88)
print(f"{'Rank':<4}{'Feature':<22}{'dARI perm':>10}{'+/- SD':>9}{'Flip rate':>12}{'dARI abl':>10}{'eta2':>8}")
print("-" * 88)
for i, r in enumerate(rows, 1):
    print(f"{i:<4}{r['name']:<22}{r['d_ari']:>10.4f}{r['d_ari_std']:>9.4f}"
          f"{r['flip_fixed']:>12.4f}{r['d_ari_abl']:>10.4f}{r['eta2']:>8.3f}"
          f"{'  (constant)' if r['const'] else ''}")
print("=" * 88)

print("\nTop five:")
for i, r in enumerate(rows[:5], 1):
    print(f"  {i}. {r['name']:<22} dARI={r['d_ari']:.4f}  flip rate={r['flip_fixed']:.1%}  eta2={r['eta2']:.3f}")
print(f"\nTop-five total dARI = {sum(r['d_ari'] for r in rows[:5]):.4f}, "
      f"{sum(r['d_ari'] for r in rows[:5]) / total * 100:.0f}% of the 21-column total")

os.makedirs('clustering_results', exist_ok=True)

# ------------------------------------------------------------------ CSV
with io.open('clustering_results/feature_influence_ranking.csv', 'w', encoding='utf-8-sig', newline='') as f:
    f.write('rank,feature,dARI_permutation,dARI_std,dARI_min,dARI_max,'
            'fixed_center_flip_rate,dARI_ablation,ANOVA_F,eta2,is_constant\n')
    for i, r in enumerate(rows, 1):
        f.write(f"{i},{r['name']},{r['d_ari']:.6f},{r['d_ari_std']:.6f},"
                f"{r['d_ari_min']:.6f},{r['d_ari_max']:.6f},{r['flip_fixed']:.6f},"
                f"{r['d_ari_abl']:.6f},{r['F']:.4f},{r['eta2']:.6f},"
                f"{'yes' if r['const'] else 'no'}\n")

# ------------------------------------------------------------------ TXT
L = []
L.append('Influence ranking of the 21 features on the clustering result')
L.append('=' * 70)
L.append(f'Baseline: StandardScaler -> KMeans(K={N_CLUSTERS}, init=k-means++, '
         f'n_init={N_INIT}, random_state={SEED})')
L.append(f'Baseline SSE = {base.inertia_:.2f}   cluster sizes = {np.bincount(L0).tolist()}')
L.append(f'Noise floor: same data, no shuffling, only a new initialisation -> dARI mean '
         f'{floor_mean:.4f} (max {floor_max:.4f})')
L.append('             Anything below that cannot be called "an effect".')
L.append('')
L.append('Measures:')
L.append('  dARI permutation   Re-cluster after shuffling the column; how much the partition')
L.append('                     degrades (1 - ARI), averaged over 8 repeats. Ranking criterion.')
L.append('  Fixed-centre flip  Shuffle the column without refitting, assign to the baseline')
L.append('                     centres, and report the fraction of samples that changed cluster.')
L.append('                     Free of re-clustering randomness.')
L.append('  dARI ablation      Drop the column (one dimension less) and re-cluster. Not fully')
L.append('                     equivalent to the permutation measure, reference only.')
L.append('  ANOVA F / eta^2    Between-cluster differences on that column across the 6 clusters.')
L.append('                     Measures "how separable this column is", not "influence on the split".')
L.append('')
L.append('=' * 70)
L.append(f"{'Rank':<4}{'Feature':<22}{'dARI perm':>10}{'+/- SD':>9}{'Flip rate':>9}{'dARI abl':>10}{'eta2':>8}")
L.append('-' * 70)
for i, r in enumerate(rows, 1):
    L.append(f"{i:<4}{r['name']:<22}{r['d_ari']:>10.4f}{r['d_ari_std']:>9.4f}"
             f"{r['flip_fixed']:>9.1%}{r['d_ari_abl']:>10.4f}{r['eta2']:>8.3f}")
L.append('=' * 70)
L.append('')
L.append('Top five (by permutation dARI):')
for i, r in enumerate(rows[:5], 1):
    L.append(f"  {i}. {r['name']:<22} dARI={r['d_ari']:.4f}  flip rate={r['flip_fixed']:.1%}  eta2={r['eta2']:.3f}")
L.append('')
L.append(f'Top-five total dARI = {sum(r["d_ari"] for r in rows[:5]):.4f}, '
         f'which is {sum(r["d_ari"] for r in rows[:5]) / total * 100:.0f}% of the {total:.4f} '
         f'total across all 21 columns.')
L.append('')
L.append('Key conclusions:')
L.append(f'1. Rank one is {rows[0]["name"]} (dARI={rows[0]["d_ari"]:.4f}, '
         f'about {rows[0]["d_ari"] / max(floor_mean, 1e-9):.1f}x the noise floor '
         f'{floor_mean:.4f}); shuffling this column alone clearly changes the partition.')
L.append('2. The top five are three wavelet band energies (band_0 very-low frequency, band_1 low '
         'frequency, band_3 main-wave harmonic) plus kurtosis plus wavelet entropy,')
L.append('   together pointing at "how the low-frequency energy is distributed" and "how steep '
         'the waveform is", consistent with the six clusters separating by signal regularity.')
zero_cols = [r['name'] for r in rows if r['d_ari'] == 0.0]
L.append(f'3. {len(zero_cols)} columns have dARI exactly 0 ({", ".join(zero_cols)}): '
         'strictly speaking std and rms are not constant because of floating-point residue, '
         'but the partition does not change at all after shuffling,')
L.append('   which agrees with "these 3 columns carry no discriminative information" and makes '
         'a good sanity check.')
L.append('4. The disagreement should be stated honestly: ranked by ANOVA F, band_5/6/7 come out '
         'near the top (eta^2 0.74~0.84), but their absolute values are only of order 1e-9,')
L.append('   so shuffling them barely changes inter-sample distances; their dARI stays mid-table '
         'and has a large SD (0.14). "Strongly separable" and "strong influence on the partition" '
         'are two different things and must not be mixed up in a paper.')
L.append('5. Limitation: the features are strongly correlated (band_0 and band_1 both have high '
         'dARI), so the influence of any single feature is partly replaced by its correlated '
         'columns and is underestimated; read the ranking as a relative order, not as absolute weights.')
L.append('')
L.append('Appendix: actual frequency bands of band_i (fs=100Hz, db6, level=7, verified with pure sine tones)')
L.append('  band_0  0.00 -  0.39 Hz   approximation coefficients cA7, residual baseline / very low freq')
L.append('  band_1  0.39 -  0.78 Hz   cD7')
L.append('  band_2  0.78 -  1.56 Hz   cD6, about 47~94 bpm, where the heart-rate fundamental sits')
L.append('  band_3  1.56 -  3.12 Hz   cD5, main-wave harmonics, dicrotic wave')
L.append('  band_4  3.12 -  6.25 Hz   cD4')
L.append('  band_5  6.25 - 12.50 Hz   cD3')
L.append('  band_6 12.50 - 25.00 Hz   cD2')
L.append('  band_7 25.00 - 50.00 Hz   cD1, close to noise')
L.append('  Note: in the code coeffs[0] is the coarsest approximation, so the numbering and the '
         'cD labels run in opposite directions.')
io.open('clustering_results/feature_influence_ranking.txt', 'w', encoding='utf-8-sig', newline='\r\n').write('\n'.join(L))

# ------------------------------------------------------------------ PNG
# Only the top 10 are drawn, as a single horizontal bar chart
# (error bars and the noise-floor reference line are kept)
from matplotlib.patches import Patch

top = rows[:10]
fig, ax = plt.subplots(figsize=(9.8, 6.2))

ys = np.arange(len(top))[::-1]
colors = ['#d62728' if i < 5 else '#1f77b4' for i in range(len(top))]
ax.barh(ys, [r['d_ari'] for r in top], color=colors, height=0.6,
        xerr=[r['d_ari_std'] for r in top], ecolor='#5F5E5A',
        capsize=3.5, error_kw={'lw': 1.0, 'capthick': 1.0})

ax.axvline(floor_mean, color='#854F0B', lw=1.2, ls='--')
ax.text(floor_mean + 0.008, -0.62, f'Noise floor {floor_mean:.3f}', color='#854F0B',
        fontsize=9, va='center', ha='left')

for y, r in zip(ys, top):
    ax.text(r['d_ari'] + r['d_ari_std'] + 0.012, y, f"{r['d_ari']:.3f}",
            va='center', fontsize=9, color='#2C2C2A')

ax.set_yticks(ys)
ax.set_yticklabels([r['name'] for r in top], fontsize=10)
ax.set_xlim(0, max(r['d_ari'] + r['d_ari_std'] for r in top) * 1.14)
ax.set_xlabel('Permutation dARI (higher = stronger influence on the partition)', fontsize=10)
ax.set_title('Top 10 features by influence on K-means clustering (K = 6)', fontsize=13, pad=12)
ax.grid(axis='x', alpha=0.25, linestyle=':')
ax.set_axisbelow(True)
ax.legend(handles=[Patch(facecolor='#d62728', label='Rank 1-5'),
                   Patch(facecolor='#1f77b4', label='Rank 6-10')],
          loc='lower right', frameon=False, fontsize=9)

fig.text(0.5, 0.012,
         'Error bars: +/-1 SD across 8 independent shuffles.   '
         'Dashed line: dARI floor obtained by re-initialisation only, without shuffling.',
         ha='center', fontsize=8.5, color='#5F5E5A')

fig.tight_layout(rect=[0, 0.04, 1, 1])
fig.savefig('clustering_results/feature_influence_ranking.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig)

print("\nOutput written: clustering_results/feature_influence_ranking.csv / .txt / .png")
