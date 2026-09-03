import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.models import load_model


K_NN = 10
SUMMARY_SIZE = 10

base_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.environ.get("STABLE_BAG_DATA", os.path.join(base_path, "data"))
RESULTS_DIR = os.environ.get(
    "STABLE_BAG_RESULTS",
    os.path.join(base_path, "results"),
)
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def load():
    df_train = pd.read_excel(os.path.join(data_path, "df_train.xlsx")).drop(
        columns=["participant_id"])
    df_test_full = pd.read_excel(os.path.join(data_path, "df_test.xlsx"))
    test_ids = df_test_full["participant_id"].reset_index(drop=True)
    df_test = df_test_full.drop(columns=["participant_id"]).reset_index(drop=True)

    age_col = df_train.columns.get_loc("age")
    X_pool = df_train.iloc[:, :age_col]
    y_pool = df_train["age"].values
    X_test = df_test.iloc[:, :age_col]
    y_test = df_test["age"].values
    
    """
    scaler = MinMaxScaler().fit(X_pool)
    X_pool_s = scaler.transform(X_pool)
    X_test_s = scaler.transform(X_test)
    """
    X_pool_s = X_pool.reset_index(drop=True).to_numpy();
    X_test_s = X_test.reset_index(drop=True).to_numpy();
    return X_pool_s, y_pool, X_test_s, y_test, test_ids


def shap_on_background(model, x_row, bg_features):
    """One KernelExplainer SHAP computation on a small background."""
    explainer = shap.KernelExplainer(model.predict, bg_features)
    shap_values = explainer.shap_values(x_row, nsamples="auto", silent = True)
    shap_arr = shap_values[0] if isinstance(shap_values, list) else shap_values
    shap_arr = np.asarray(shap_arr).reshape(-1)
    phi_0 = float(np.asarray(explainer.expected_value).reshape(-1)[0])
    return shap_arr, phi_0

def load_custom_model(path: str):
    """Try to load a Keras model from the given path, checking for both .h5 and .keras formats."""
    h5_model_path = os.path.join(path, "model.h5")
    keras_model_path = os.path.join(path, "model.keras")
    
    if os.path.exists(h5_model_path):
        print(f"Loading model from {h5_model_path}")
        return load_model(h5_model_path)
    elif os.path.exists(keras_model_path):
        print(f"Loading model from {keras_model_path}")
        return load_model(keras_model_path)
    else:
        raise FileNotFoundError(f"No model found at {h5_model_path} or {keras_model_path}")
    
def main():
    print(f"Data:    {data_path}")
    print(f"Results: {RESULTS_DIR}")

    X_pool, y_pool, X_test, y_test, test_ids = load()
    model = load_custom_model(data_path)
    print(f"Pool: {len(X_pool)}  Test: {len(X_test)}")

    # kNN in feature space (one global query)
    nn_feat = NearestNeighbors(n_neighbors=K_NN, metric="euclidean").fit(X_pool)
    _, idx_feat = nn_feat.kneighbors(X_test)

    # kNN in age space (argsort per subject)
    age_diff = np.abs(y_pool[None, :] - y_test[:, None])
    idx_age = np.argsort(age_diff, axis=1)[:, :K_NN]

    f_x_test = model.predict(X_test, verbose=0).flatten()
    rows = []
    for i in range(len(X_test)):
        y = float(y_test[i])
        fx = float(f_x_test[i])

        bg_f = X_pool[idx_feat[i]]
        
        shap_f, phi_0_f = shap_on_background(model, X_test[i:i + 1], bg_f)
        ages_f = y_pool[idx_feat[i]]

        bg_a = X_pool[idx_age[i]]
        shap_a, phi_0_a = shap_on_background(model, X_test[i:i + 1], bg_a)
        ages_a = y_pool[idx_age[i]]

        print("prediction: ", fx, "|| Age: ", y)

        rows.append({
            "id": test_ids.iloc[i],
            "y": y,
            "f_x": fx,
            "bag": fx - y,
            "phi_0_NNfeat":    phi_0_f,
            "cal_NNfeat":      abs(phi_0_f - y),
            "sum_shap_NNfeat": float(shap_f.sum()),
            "NNfeat_age_mean": float(ages_f.mean()),
            "NNfeat_age_std":  float(ages_f.std()),
            "NNfeat_age_range": float(ages_f.max() - ages_f.min()),
            "phi_0_NNage":     phi_0_a,
            "cal_NNage":       abs(phi_0_a - y),
            "sum_shap_NNage":  float(shap_a.sum()),
            "NNage_age_mean":  float(ages_a.mean()),
            "NNage_age_std":   float(ages_a.std()),
            "NNage_age_range": float(ages_a.max() - ages_a.min()),
        })
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(X_test)}]")

    nn_df = pd.DataFrame(rows)

    # Merge with STABLE-BAG reference
    sb_path = os.path.join(RESULTS_DIR, "test_results.csv")
    if os.path.exists(sb_path):
        sb = pd.read_csv(sb_path)
        sb = sb[sb["status"] == "ok"][[
            "id", "calibration", "phi_0", "best_delta",
            "local_density", "instab",
        ]].rename(columns={
            "calibration":   "cal_sb",
            "phi_0":         "phi_0_sb",
            "best_delta":    "best_delta_sb",
            "local_density": "density_sb",
            "instab":        "instab_sb",
        })
        comp = nn_df.merge(sb, on="id", how="inner")
    else:
        print("WARNING: no test_results.csv — saving kNN data only, no STABLE-BAG join.")
        comp = nn_df

    out_csv = os.path.join(RESULTS_DIR, "ablation_backgrounds.csv")
    comp.to_csv(out_csv, index=False)
    print(f"Saved {out_csv} ({len(comp)} rows)")

    if "cal_sb" not in comp.columns:
        return

    cals = {
        "STABLE-BAG":  comp["cal_sb"].values,
        "kNN-age":     comp["cal_NNage"].values,
        "kNN-feature": comp["cal_NNfeat"].values,
    }

    # Calibration bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(cals))
    width = 0.35
    means = [v.mean() for v in cals.values()]
    medians = [float(np.median(v)) for v in cals.values()]
    ax.bar(x - width / 2, means, width, label="mean", color="steelblue")
    ax.bar(x + width / 2, medians, width, label="median", color="orange")
    for i, (m, md) in enumerate(zip(means, medians)):
        ax.text(i - width / 2, m + 0.05, f"{m:.2f}", ha="center", fontsize=9)
        ax.text(i + width / 2, md + 0.05, f"{md:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(list(cals.keys()))
    ax.set_ylabel(r"Calibration  $|\varphi_0 - y|$  (years)")
    ax.set_title("Background-strategy ablation: calibration of $\\varphi_0$ against chronological age")
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig_ablation_calibration.png"), dpi=150)
    plt.close()

    # Diagonality plots (both kNN variants)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    for ax, key, label in [(axes[0], "NNfeat", "kNN-feature"),
                           (axes[1], "NNage", "kNN-age")]:
        sc = ax.scatter(comp["y"], comp[f"{key}_age_mean"],
                        c=comp[f"{key}_age_std"], cmap="plasma",
                        alpha=0.7, s=22, edgecolor="none")
        lim_lo = float(min(comp["y"].min(), comp[f"{key}_age_mean"].min())) - 2
        lim_hi = float(max(comp["y"].max(), comp[f"{key}_age_mean"].max())) + 2
        ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", lw=1, label="y = x")
        ax.set_xlabel(r"Chronological age $y$ (years)")
        ax.set_title(f"{label}: mean age of 10 NN vs $y$")
        ax.grid(alpha=0.3)
        plt.colorbar(sc, ax=ax, label="Std of NN ages")
        ax.legend(loc="best")
    axes[0].set_ylabel("Mean age of 10 NN (years)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig_ablation_diagonality.png"), dpi=150)
    plt.close()

    # Age range histograms
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    upper = float(max(comp["NNfeat_age_range"].max(),
                      comp["NNage_age_range"].max())) + 1
    bins = np.linspace(0, upper, 50)
    ax.hist(comp["NNfeat_age_range"], bins=bins, color="C0", alpha=0.5,
            label="kNN-feature")
    ax.hist(comp["NNage_age_range"], bins=bins, color="C2", alpha=0.5,
            label="kNN-age")
    ax.set_xlabel("Age range of 10 NN (max - min) (years)")
    ax.set_ylabel("Test subjects")
    ax.set_title("Age spread of selected peers, by kNN strategy")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig_ablation_age_range.png"), dpi=150)
    plt.close()

    # Console summary
    print("\n=== Calibration summary (test set) ===")
    for k, v in cals.items():
        print(f"  {k:<12}  mean={v.mean():.3f}  median={np.median(v):.3f}  max={v.max():.3f}")
    n_win_f = int((comp["cal_sb"] < comp["cal_NNfeat"]).sum())
    n_win_a = int((comp["cal_sb"] < comp["cal_NNage"]).sum())
    n = len(comp)
    print(f"\nSTABLE-BAG better than kNN-feature: {n_win_f}/{n} ({100*n_win_f/n:.1f}%)")
    print(f"STABLE-BAG better than kNN-age:     {n_win_a}/{n} ({100*n_win_a/n:.1f}%)")


if __name__ == "__main__":
    main()
