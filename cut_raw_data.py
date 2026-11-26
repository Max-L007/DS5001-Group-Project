import os
import numpy as np


def _find(name):
    """Look for a data file in the current directory and its parent directories,
    so the script still works when it is placed in a sub-folder."""
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(4):
        for cand in (d, os.path.join(d, 'pulse_wave_clustering_analysis')):
            p = os.path.join(cand, name)
            if os.path.exists(p):
                return p
        d = os.path.dirname(d)
    return name


data = np.load(_find('raw_data.npy'))
print('Raw data shape: %s   dtype=%s' % (data.shape, data.dtype))

# Raw data is laid out as (sample, group, channel, sample_point).
# Take all 1000 signals from group 0, channel 0.
cut = data[:, 0, 0, :].astype(np.float64)
print('Cut shape:      %s   dtype=%s (%d signals)' % (cut.shape, cut.dtype, cut.shape[0]))

np.save('cut_pulse_wave_data.npy', cut)
print('Saved: cut_pulse_wave_data.npy')
