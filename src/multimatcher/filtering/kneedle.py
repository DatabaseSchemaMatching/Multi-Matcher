from __future__ import annotations
import numpy as np
from scipy.signal import argrelextrema

def kneedle(sim_scores, S: float = 1.0, D: float = 0.85):
    xs = np.sort(sim_scores)
    ys = np.arange(1, xs.size + 1) / xs.size
    y_diff = ys - xs

    maxima = argrelextrema(y_diff, np.greater)[0]
    minima = argrelextrema(y_diff, np.less)[0]
    maxima = np.append(maxima, len(xs) - 1)
    minima = np.append(minima, len(xs) - 1)

    y_diff_maxima = y_diff[maxima[:-1]]
    Tmx = y_diff_maxima - (S * np.abs(np.diff(xs).mean()))

    knee = None
    max_i = 0
    min_i = 0
    num_peaks = len(maxima) - 1

    threshold = None
    peak_idx = None
    curve = "concave"
    direction = "increasing"

    for idx in range(len(xs)):
        if max_i < num_peaks and idx >= maxima[max_i]:
            threshold = Tmx[max_i]
            peak_idx = idx
            max_i += 1

        if min_i < len(minima) - 1 and idx >= minima[min_i]:
            threshold = 0.0
            min_i += 1

        if idx < maxima[0]:
            continue

        if threshold is not None and y_diff[idx] < threshold:
            if curve == "concave":
                knee = xs[peak_idx] if direction == "increasing" else xs[-(peak_idx + 1)]
            else:
                knee = xs[-(peak_idx + 1)] if direction == "decreasing" else xs[peak_idx]
            break

    return None if knee is None else knee * D