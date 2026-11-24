"""Input-only direction priors used by both pixel models."""
import io
import zipfile
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.special import expit


class Directions:
    def __init__(self, assets, cache):
        self.cache = cache
        table = pd.read_csv(assets / 'fixed_shares.csv.gz', dtype={'event': str})
        self.shares = table.pivot(index=['event', 'target'], columns='sector', values='q')
        assert list(self.shares.columns) == list(range(8))
        assert np.allclose(self.shares.sum(axis=1), 1) and self.shares.notna().all().all()
        self.domains = zipfile.ZipFile(assets / 'domains.zip')

    @lru_cache(maxsize=4)
    def xy(self, event):
        return np.load(self.cache / event / 'xy.npy').reshape(-1, 2).astype(float)

    def get(self, row, past, recent):
        key = (row.event, row.target_date)
        if key in self.shares.index:
            name = f'{row.event}/{row.target_index:03d}.npz'
            with np.load(io.BytesIO(self.domains.read(name))) as z:
                ids, sector = z['pixel'].astype(int), z['sector'].astype(int)
            q, source = self.shares.loc[key].to_numpy(), 'combined'
        elif recent.any():
            grid = self.xy(row.event)
            offset = grid - grid[recent.ravel()].mean(0)
            radius = np.linalg.norm(offset, axis=1)
            ids = np.flatnonzero((radius <= 10) & (radius > 1e-6))
            bearing = np.degrees(np.arctan2(offset[ids, 0], offset[ids, 1]))
            sector = ((bearing + 22.5) % 360 // 45).astype(int)
            counts = np.bincount(sector[recent.ravel()[ids]], minlength=8) + .5
            q, source = counts / counts.sum(), 'previous_smoothed'
        else:
            return np.array([], int), np.array([], int), np.full(8, 1/8), 'neutral'
        keep = ~past.ravel()[ids]
        return ids[keep], sector[keep], q, source


def corrected_scores(raw, ids, sector, q, exact):
    """Keep exterior scores. Inside the prior domain allocate expected new area."""
    scores = raw.ravel().copy()
    if not ids.size:
        return scores
    local = scores[ids].astype(float)
    mass = np.bincount(sector, weights=expit(local), minlength=8)
    total = mass.sum()
    if total <= 0:
        return scores
    if not exact:
        ratio = q / np.maximum(mass / total, 1e-8)
        scores[ids] += .25 * np.log(np.clip(ratio, .25, 4))[sector]
        return scores

    # Allocate total probability subject to the available pixel count per sector.
    capacity = np.bincount(sector, minlength=8)
    available = capacity > 0
    target = np.zeros(8)
    remaining = total
    while available.any():
        proposed = remaining * q[available] / q[available].sum()
        idx = np.flatnonzero(available)
        full = idx[proposed > capacity[idx]]
        if not len(full):
            target[idx] = proposed
            break
        target[full] = capacity[full]
        remaining -= capacity[full].sum()
        available[full] = False

    lo, hi = np.full(8, -80.), np.full(8, 80.)
    for _ in range(30):
        mid = (lo + hi) / 2
        measured = np.bincount(sector, weights=expit(local + mid[sector]), minlength=8)
        lo, hi = np.where(measured < target, mid, lo), np.where(measured >= target, mid, hi)
    scores[ids] = local + ((lo + hi) / 2)[sector]
    return scores
