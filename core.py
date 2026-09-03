"""
Shared utilities for SHAP-based BAG decomposition.

Key ideas:
    - Background extraction: age-matched peer window [y - delta, y + delta].
    - Bootstrap SHAP: K KernelExplainer runs on resampled backgrounds.
      The mean shap_values are reported; the mean per-feature std across
      bootstraps is the instability.
    - Sanity check: local accuracy property sum(phi_i) == f(x) - phi_0,
      within a tolerance. Subjects that fail are flagged.
"""

import numpy as np
import pandas as pd
import shap
import random

SUMMARY_SIZE = 100      # shap.kmeans summary cap for background
K_BOOTSTRAP = 10        # bootstrap reruns for instability estimation
SANITY_TOL = 1e-3       # tolerance on local-accuracy residual


def extract_background(delta, y_subject, X_pool, y_pool):
    """Rows of X_pool whose y_pool is within [y_subject - delta, y_subject + delta]."""
    mask = (y_pool >= (y_subject - delta)) & (y_pool <= (y_subject + delta))
    return X_pool[mask]


def compute_subject_delta(
    model,
    x_subject,
    y_subject,
    background,
    K=K_BOOTSTRAP,
    summary_size=SUMMARY_SIZE,
    seed=0,
):
    """
    For one (subject, delta): run K bootstrap KernelSHAP computations.

    Parameters
    ----------
    model        : Keras model with .predict
    x_subject    : 1D array of scaled features for the target subject
    y_subject    : chronological age of the target subject (scalar)
    background   : DataFrame of scaled features for the age-matched peers
    K            : number of bootstrap reruns
    summary_size : shap.kmeans cluster count cap
    seed         : RNG seed for bootstrap resampling

    Returns
    -------
    dict or None (None if background is empty)
    """
    if len(background) == 0:
        return None

    bg_vals = background.values if hasattr(background, "values") else np.asarray(background)
    n_bg = len(bg_vals)
    n_features = x_subject.shape[-1]
    x_row = x_subject.reshape(1, -1)

    f_x = float(model(x_row)[0, 0])
    print("prediction: ", f_x, "|| Age: ", y_subject)
    rng = np.random.default_rng(seed)
    boot_shap = np.zeros((K, n_features))
    boot_phi_0 = np.zeros(K)

    for k in range(K):
        idx = rng.integers(0, n_bg, size=n_bg)
        bg_k = bg_vals[idx]
        uniques = np.unique(idx) 
        # uniques, _ = np.unique_counts(idx) # per numpy>2.0

        n_summary = min(len(bg_k), summary_size)
        
        if len(uniques) > SUMMARY_SIZE and len(bg_k) > SUMMARY_SIZE:
            # print("[!] Subsampling with k-means.")
            bg_summary = shap.kmeans(bg_k, n_summary)
        elif len(uniques) < SUMMARY_SIZE and len(bg_k) > SUMMARY_SIZE:
            # print("[!] Random sampling.")
            idx_subsampled = np.random.choice(bg_k.shape[0], n_summary, replace=False);
            bg_summary = bg_k[idx_subsampled];
        else:
            # print("[!] No sampling.")
            bg_summary = bg_k;
        explainer = shap.KernelExplainer(model, bg_summary)
        shap_k = explainer.shap_values(x_row, nsamples="auto", silent = True)
        shap_k = np.asarray(shap_k[0] if isinstance(shap_k, list) else shap_k).reshape(-1)
        boot_shap[k] = shap_k
        boot_phi_0[k] = float(np.asarray(explainer.expected_value).reshape(-1)[0])

    mean_shap = boot_shap.mean(axis=0)
    mean_phi_0 = float(boot_phi_0.mean())
    instab = float(boot_shap.std(axis=0).mean())
    residual = float(abs(mean_shap.sum() - (f_x - mean_phi_0)))

    return {
        "phi_0": mean_phi_0,
        "shap_values": mean_shap,
        "f_x": f_x,
        "residual": residual,
        "sanity_ok": residual <= SANITY_TOL,
        "instab": instab,
        "n_background": n_bg,
        "calibration": abs(mean_phi_0 - y_subject),
    }


