from __future__ import annotations

import numpy as np


def mad(x):
    med = np.median(x)
    return np.median(np.abs(x - med)) + 1e-12


def scale_residuals(E_res, G_res, mode="mad"):
    if mode == "mad":
        sE = mad(E_res)
        sG = mad(G_res)
    else:
        sE = np.std(E_res) + 1e-12
        sG = np.std(G_res) + 1e-12
    return E_res / sE, G_res / sG, sE, sG
