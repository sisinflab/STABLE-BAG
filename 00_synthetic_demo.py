import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skewnorm, truncnorm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import shap

# === Configuration ===
SEED = 251299
N_SAMPLES = 4000
TEST_FRAC = 0.2
WIDE_BG_SIZE = 200
LOCAL_DELTA = 10

DISTRIBUTIONS = {
    "uniform": dict(distribution_type="uniform",
                    min_age=6, max_age=86),
    "gaussian": dict(distribution_type="gaussian",
                     mean_age=45, std_dev=15, min_age=6, max_age=86),
    "skewed_gaussian": dict(distribution_type="skewed_gaussian",
                            mean_age=28, std_dev=20, skew_param=5,
                            min_age=6, max_age=86),
}

# Linear coefficients calibrated on real morphometric ranges
COEFFS = {
    "GM_vol_lh-parahippocampal":   (2302.95, -6.59),
    "GM_vol_rh-parahippocampal":   (2094.38, -4.90),
    "GM_vol_rh_sum":               (275783.23, -1047.00),
    "GM_vol_lh_sum":               (276355.84, -1079.25),
    "average_thickness_lh_sum":    (90.17, -0.17),
    "average_thickness_rh_sum":    (90.10, -0.16),
}
NOISE = {f: (0.05 if "thickness" in f else 0.1) for f in COEFFS}

# === Paths ===
base_path = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.environ.get(
    "STABLE_BAG_RESULTS",
    os.path.join(base_path, "results"),
)
FIG_DIR = os.path.join(RESULTS_DIR, "figures", "synthetic")
os.makedirs(FIG_DIR, exist_ok=True)


# === Data generation ===

def generate_age_values(distribution_type, n_samples, **kwargs):
    """Sample ages within [min_age, max_age] from the chosen distribution."""
    min_age = kwargs.get("min_age", 6)
    max_age = kwargs.get("max_age", 86)

    if distribution_type == "uniform":
        return np.random.uniform(min_age, max_age, n_samples)

    if distribution_type == "gaussian":
        m = kwargs.get("mean_age", 45)
        s = kwargs.get("std_dev", 15)
        a, b = (min_age - m) / s, (max_age - m) / s
        return truncnorm.rvs(a, b, loc=m, scale=s, size=n_samples)

    if distribution_type == "skewed_gaussian":
        m = kwargs.get("mean_age", 28)
        s = kwargs.get("std_dev", 20)
        skew = kwargs.get("skew_param", 5)
        out = np.empty(0)
        while len(out) < n_samples:
            extra = skewnorm.rvs(a=skew, loc=m, scale=s, size=n_samples * 2)
            extra = extra[(extra >= min_age) & (extra <= max_age)]
            out = np.concatenate((out, extra))
        return out[:n_samples]

    raise ValueError(f"unknown distribution_type: {distribution_type}")


def generate_synthetic_features(ages, coefficients, noise_factors):
    """Linear y -> x with proportional noise; resample negative entries."""
    n = len(ages)
    data = {"age": ages}
    for feat, (intercept, coef) in coefficients.items():
        base = intercept + coef * ages
        scale = noise_factors.get(feat, 0.1) * np.abs(base)
        values = base + np.random.normal(0, scale, n)
        mask = values < 0
        while mask.any():
            values[mask] = base[mask] + np.random.normal(0, scale[mask], mask.sum())
            mask = values < 0
        data[feat] = values
    return pd.DataFrame(data)


# === Background helpers ===

def make_wide_background(X_train, size, seed):
    return X_train.sample(n=size, random_state=seed)


def make_local_background(X_train, y_train, target_age, delta):
    mask = (y_train >= (target_age - delta)) & (y_train <= (target_age + delta))
    return X_train[mask]


# === Experiment per distribution ===

def run_experiment(name, dist_kwargs):
    np.random.seed(SEED)

    ages = generate_age_values(n_samples=N_SAMPLES, **dist_kwargs)
    df = generate_synthetic_features(ages, COEFFS, NOISE)
    X = df.drop(columns=["age"])
    y = df["age"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_FRAC, random_state=SEED
    )
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    model = LinearRegression().fit(X_train, y_train)
    f_x = model.predict(X_test.values).flatten()

    # WIDE: build ONCE, batch SHAP for all test subjects
    bg_wide = make_wide_background(X_train, WIDE_BG_SIZE, seed=SEED)
    explainer_wide = shap.LinearExplainer(
        model, bg_wide, feature_perturbation="interventional"
    )
    phi_0_wide = float(np.asarray(explainer_wide.expected_value).reshape(-1)[0])
    shap_wide = explainer_wide.shap_values(X_test.values)
    if isinstance(shap_wide, list):
        shap_wide = shap_wide[0]

    # LOCAL: per-subject (background depends on y_subject)
    n_test, n_features = X_test.shape
    shap_local = np.full((n_test, n_features), np.nan)
    phi_0_local = np.full(n_test, np.nan)
    bg_size_local = np.zeros(n_test, dtype=int)
    mean_local_age = np.full(n_test, np.nan)

    for i in range(n_test):
        target = float(y_test.iloc[i])
        bg_local = make_local_background(X_train, y_train, target, LOCAL_DELTA)
        bg_size_local[i] = len(bg_local)
        if len(bg_local) == 0:
            continue
        mean_local_age[i] = float(y_train.loc[bg_local.index].mean())
        explainer_local = shap.LinearExplainer(
            model, bg_local, feature_perturbation="interventional"
        )
        sl = explainer_local.shap_values(X_test.iloc[[i]].values)
        if isinstance(sl, list):
            sl = sl[0]
        shap_local[i, :] = sl[0]
        phi_0_local[i] = float(np.asarray(explainer_local.expected_value).reshape(-1)[0])

    nan_local = np.isnan(shap_local).all(axis=1)

    # Sums
    sum_wide = shap_wide.sum(axis=1)
    sum_local = np.where(nan_local, np.nan, np.nansum(shap_local, axis=1))

    # Local accuracy residuals (should be ~ 1e-12 for LinearExplainer)
    res_wide = np.abs(sum_wide - (f_x - phi_0_wide))
    res_local = np.where(nan_local, np.nan,
                         np.abs(sum_local - (f_x - phi_0_local)))

    # Calibration gap: by local accuracy, |sum(phi) - (f(x)-y)| = |phi_0 - y|.
    # This quantifies how far the SHAP base value sits from the subject's age,
    # which is the only background-induced source of mismatch in the decomposition.
    bag_real = f_x - y_test.values
    cal_gap_wide = float(np.nanmean(np.abs(sum_wide - bag_real)))
    cal_gap_local = float(np.nanmean(np.abs(sum_local - bag_real)))

    return dict(
        name=name,
        ages_train=y_train.values,
        y_test=y_test.values,
        f_x=f_x,
        phi_0_wide=phi_0_wide,
        phi_0_local=phi_0_local,
        sum_wide=sum_wide,
        sum_local=sum_local,
        bag_real=bag_real,
        res_wide_max=float(res_wide.max()),
        res_local_max=float(np.nanmax(res_local)),
        cal_gap_wide=cal_gap_wide,
        cal_gap_local=cal_gap_local,
        bg_size_local=bg_size_local,
        mean_local_age=mean_local_age,
        n_skipped=int(nan_local.sum()),
        shap_wide=shap_wide,
        shap_local=shap_local,
        feature_names=list(X_train.columns),
    )


# === Plots ===

def plot_distribution(r):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(r["ages_train"], bins=40, color="steelblue", edgecolor="white")
    mean_age = float(np.mean(r["ages_train"]))
    ax.axvline(mean_age, color="red", ls="--", label=f"mean = {mean_age:.1f}")
    ax.set_title(f"Training age distribution — {r['name']} (n={len(r['ages_train'])})")
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"fig_distribution_{r['name']}.png"), dpi=150)
    plt.close()


def plot_sumphi_vs_age(r):
    y = r["y_test"]
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.scatter(y, r["sum_wide"], s=18, alpha=0.5, color="C0", label="Wide background")
    ax.scatter(y, r["sum_local"], s=18, alpha=0.5, color="C1", label="Local background")

    y_grid = np.linspace(y.min(), y.max(), 200)
    ax.plot(y_grid, y_grid - r["phi_0_wide"], "C0--", lw=1.3,
            label=fr"theory wide:  $y - \varphi_0^{{wide}}$  ($\varphi_0^{{wide}}$={r['phi_0_wide']:.1f})")
    ax.axhline(0, color="C1", ls="--", lw=1.3,
               label=r"theory local:  $y - \varphi_0^{local} \approx 0$")

    ax.set_xlabel("Test subject age (years)")
    ax.set_ylabel(r"$\sum_i \varphi_i$")
    ax.set_title(f"Sum of SHAP values vs age — {r['name']} distribution")
    ax.legend(fontsize=9, loc="best")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"fig_sumphi_vs_age_{r['name']}.png"), dpi=150)
    plt.close()


def plot_grid(results):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=False)
    for ax, r in zip(axes, results):
        y = r["y_test"]
        ax.scatter(y, r["sum_wide"], s=14, alpha=0.5, color="C0", label="Wide")
        ax.scatter(y, r["sum_local"], s=14, alpha=0.5, color="C1", label="Local")
        y_grid = np.linspace(y.min(), y.max(), 200)
        ax.plot(y_grid, y_grid - r["phi_0_wide"], "C0--", lw=1)
        ax.axhline(0, color="C1", ls="--", lw=1)
        ax.set_title(
            f"{r['name']}\n"
            fr"calibration gap  wide={r['cal_gap_wide']:.2f}, local={r['cal_gap_local']:.2f}"
        )
        ax.set_xlabel("Test subject age (years)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(r"$\sum_i \varphi_i$")
    axes[0].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig_sumphi_comparison.png"), dpi=150)
    plt.close()


def _short_feature_label(name, max_len=14):
    """Compact feature label for tick axes."""
    parts = name.split("_")
    if len(parts) >= 3 and parts[-2] in ("lh", "rh"):
        # e.g. GM_vol_lh-parahippocampal -> lh-paraHipp
        prefix = "_".join(parts[:-2])
        suffix = "-".join(parts[-2:]).replace("hippocampal", "Hipp")
        out = f"{suffix} | {prefix}"
    else:
        out = name
    return out if len(out) <= max_len else out[: max_len - 1] + "…"


def plot_per_subject_explanations(result, target_ages=(15, 45, 75)):
    """For representative test subjects, plot per-feature SHAP under the two
    background strategies side by side. Shows that the SAME subject receives
    materially different explanations depending on the background choice."""
    y = result["y_test"]
    f_x = result["f_x"]
    sw = result["shap_wide"]
    sl = result["shap_local"]
    feats = result["feature_names"]
    short = [_short_feature_label(f) for f in feats]
    n_features = len(feats)
    x_pos = np.arange(n_features)
    width = 0.38

    n_panels = len(target_ages)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 5.5), sharey=True)
    if n_panels == 1:
        axes = [axes]

    for ax, target in zip(axes, target_ages):
        idx = int(np.argmin(np.abs(y - target)))
        if np.isnan(sl[idx]).any():
            ax.set_title(f"target={target} — local bg empty")
            continue
        ax.bar(x_pos - width / 2, sw[idx], width, label="Wide", color="C0")
        ax.bar(x_pos + width / 2, sl[idx], width, label="Local", color="C1")
        ax.axhline(0, color="gray", lw=0.6)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(short, rotation=40, ha="right", fontsize=8)
        ax.set_title(
            f"y = {y[idx]:.1f}    f(x) = {f_x[idx]:.2f}\n"
            fr"$\varphi_0^{{wide}}={result['phi_0_wide']:.1f}$, "
            fr"$\varphi_0^{{local}}={result['phi_0_local'][idx]:.1f}$"
            "\n"
            fr"$\sum\varphi^{{wide}}={result['sum_wide'][idx]:+.2f}$, "
            fr"$\sum\varphi^{{local}}={result['sum_local'][idx]:+.2f}$",
            fontsize=10,
        )
        ax.grid(alpha=0.3, axis="y")

    axes[0].set_ylabel(r"SHAP value $\varphi_i$")
    axes[0].legend(loc="best")
    fig.suptitle(
        f"Per-feature explanations under wide vs local background — {result['name']}",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIG_DIR, f"fig_per_subject_explanations_{result['name']}.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close()


def plot_phi_wide_vs_local(result):
    """Scatter of phi_i^wide vs phi_i^local across all (subject, feature) pairs.
    Deviations from the diagonal quantify per-element explanation variability."""
    sw = result["shap_wide"]
    sl = result["shap_local"]
    valid = ~np.isnan(sl).any(axis=1)
    sw_v = sw[valid].flatten()
    sl_v = sl[valid].flatten()

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(sw_v, sl_v, s=6, alpha=0.3, color="steelblue", edgecolor="none")
    lim = max(np.abs(sw_v).max(), np.abs(sl_v).max()) * 1.05
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=1,
            label=r"parity $\varphi^{wide}=\varphi^{local}$")
    ax.set_xlabel(r"$\varphi_i^{wide}$  (years)")
    ax.set_ylabel(r"$\varphi_i^{local}$  (years)")
    ax.set_title(
        f"Per-feature SHAP: wide vs local — {result['name']}  "
        f"({valid.sum()} subjects × {sw.shape[1]} features)"
    )
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"fig_phi_wide_vs_local_{result['name']}.png"),
                dpi=150)
    plt.close()


def plot_per_feature_shift(result):
    """Per feature, distribution of (phi_wide - phi_local) across subjects.
    Shows which features are most sensitive to background choice."""
    sw = result["shap_wide"]
    sl = result["shap_local"]
    valid = ~np.isnan(sl).any(axis=1)
    diff = sw[valid] - sl[valid]
    feats = result["feature_names"]
    short = [_short_feature_label(f) for f in feats]
    medians = np.median(diff, axis=0)
    order = np.argsort(np.abs(medians))[::-1]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot(
        [diff[:, j] for j in order],
        tick_labels=[short[j] for j in order],
        showfliers=False,
    )
    ax.axhline(0, color="gray", lw=0.7)
    ax.set_ylabel(r"$\varphi_i^{wide} - \varphi_i^{local}$  (years)")
    ax.set_title(f"Per-feature shift induced by background choice — {result['name']}")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"fig_per_feature_shift_{result['name']}.png"),
                dpi=150)
    plt.close()


def plot_metrics(results):
    names = [r["name"] for r in results]
    cal_wide = [r["cal_gap_wide"] for r in results]
    cal_local = [r["cal_gap_local"] for r in results]
    x = np.arange(len(names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w / 2, cal_wide, w, label="Wide", color="C0")
    ax.bar(x + w / 2, cal_local, w, label="Local", color="C1")
    for i, (cw, cl) in enumerate(zip(cal_wide, cal_local)):
        ax.text(i - w / 2, cw + 0.1, f"{cw:.2f}", ha="center", fontsize=9)
        ax.text(i + w / 2, cl + 0.1, f"{cl:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel(r"Mean $|\varphi_0 - y|$  (years)")
    ax.set_title("Calibration gap of the SHAP base value $\\varphi_0$ "
                 "against chronological age, by background strategy")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig_metrics_summary.png"), dpi=150)
    plt.close()


# === Main ===

def main():
    print(f"Saving figures to {FIG_DIR}")
    results = []
    for name, kw in DISTRIBUTIONS.items():
        print(f"\n=== {name} ===")
        r = run_experiment(name, kw)
        results.append(r)
        plot_distribution(r)
        plot_sumphi_vs_age(r)
        plot_per_subject_explanations(r, target_ages=(15, 45, 75))
        plot_phi_wide_vs_local(r)
        plot_per_feature_shift(r)
        print(f"  phi_0 wide:                 {r['phi_0_wide']:.2f}")
        print(f"  Local accuracy res (max):   wide={r['res_wide_max']:.2e}  "
              f"local={r['res_local_max']:.2e}")
        print(f"  Calibration gap |phi_0 - y|: wide={r['cal_gap_wide']:.3f}  "
              f"local={r['cal_gap_local']:.3f}")
        print(f"  Subjects with empty local bg: {r['n_skipped']}")
    plot_grid(results)
    plot_metrics(results)
    print(f"\nDone. Figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
