"""
Final test run with lambda* fixed.

For each test subject:
    - For each delta in DELTA_GRID, run bootstrap SHAP.
    - Pick best delta via loss = |phi_0 - y| + lambda_star * instab.
    - Save all quantities, SHAP values (long format), density, sanity residual.

The background pool is the FULL df_train (no split) - this is phase 1 baseline.
Phase 2 (UKBB + VAE augmentation) will reuse this script with a different pool.

Outputs
-------
    results/test_results.csv       : per-subject summary
    results/test_shap_long.parquet : per-subject SHAP values in long format
                                     (subject_id, feature, shap_value)
    results/test_checkpoint.pkl    : checkpoint (resumable)
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

from core import extract_background, compute_subject_delta, K_BOOTSTRAP, SUMMARY_SIZE

os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"

SEED = 42
DELTA_GRID = [2, 3, 5, 7, 10, 15, 20]

base_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_path, "data")
results_path = os.path.join(base_path, "results")
os.makedirs(results_path, exist_ok=True)

train_path = os.path.join(data_path, "df_train.xlsx")
test_path = os.path.join(data_path, "df_test.xlsx")

lambda_star_json = os.path.join(results_path, "lambda_star.json")

checkpoint_path = os.path.join(results_path, "test_checkpoint.pkl")
results_csv = os.path.join(results_path, "test_results.csv")
shap_long_parquet = os.path.join(results_path, "test_shap_long.parquet")


def load_lambda_star():
    with open(lambda_star_json) as f:
        cfg = json.load(f)
    return float(cfg["lambda_star"])


def load_data():
    df_train = pd.read_excel(train_path).drop(columns=["participant_id"])
    df_test_full = pd.read_excel(test_path)
    test_ids = df_test_full["participant_id"].reset_index(drop=True)
    df_test = df_test_full.drop(columns=["participant_id"]).reset_index(drop=True)

    age_col = df_train.columns.get_loc("age")
    X_pool = df_train.iloc[:, :age_col]
    y_pool = df_train["age"].reset_index(drop=True)
    X_test = df_test.iloc[:, :age_col]
    y_test = df_test["age"].reset_index(drop=True)
    """
    scaler = MinMaxScaler().fit(X_pool)
    X_pool_s = pd.DataFrame(scaler.transform(X_pool), columns=X_pool.columns).reset_index(drop=True)
    X_test_s = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns).reset_index(drop=True)
    """
    X_pool_s = X_pool.reset_index(drop=True);
    X_test_s = X_test.reset_index(drop=True);

    return X_pool_s, y_pool, X_test_s, y_test, test_ids, X_pool.columns.tolist()


def process_subject(model, x_i, y_i, X_pool, y_pool, lambda_star, seed_base):
    """Grid-search delta for one test subject, return the best result."""
    per_delta = {}
    for delta in DELTA_GRID:
        bg = extract_background(delta, y_i, X_pool, y_pool)
        if len(bg) == 0:
            per_delta[delta] = None
            continue
        res = compute_subject_delta(
            model, x_i, y_i, bg,
            K=K_BOOTSTRAP, summary_size=SUMMARY_SIZE,
            seed=seed_base + int(delta),
        )
        per_delta[delta] = res

    valid = [(d, r) for d, r in per_delta.items() if r is not None and r["sanity_ok"]]
    if not valid:
        return None

    best_d, best_r = min(
        valid,
        key=lambda dr: dr[1]["calibration"] + lambda_star * dr[1]["instab"],
    )
    best_r = dict(best_r)
    best_r["best_delta"] = best_d
    best_r["loss"] = best_r["calibration"] + lambda_star * best_r["instab"]
    
    best_r["local_density"] = best_r["n_background"]
    return best_r

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
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    lambda_star = load_lambda_star()
    print(f"lambda_star = {lambda_star}")

    X_pool, y_pool, X_test, y_test, test_ids, feature_names = load_data()
    print(f"Pool: {len(X_pool)} | Test: {len(X_test)} | Features: {len(feature_names)}")
    print(f"Delta grid: {DELTA_GRID} | K bootstrap: {K_BOOTSTRAP}")

    model = load_custom_model(data_path)

    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "rb") as f:
            results = pickle.load(f)
        start_idx = len(results)
        print(f"[resume] starting at test subject {start_idx + 1}/{len(X_test)}")
    else:
        results = []
        start_idx = 0

    for i in range(start_idx, len(X_test)):
        x_i = X_test.iloc[i].values
        y_i = float(y_test.iloc[i])
        sid = test_ids.iloc[i]
        print(f"[test] subject {i + 1}/{len(X_test)} id={sid} age={y_i:.1f}")

        best = process_subject(
            model, x_i, y_i, X_pool, y_pool, lambda_star,
            seed_base=SEED + i * 100,
        )
        if best is None:
            print(f"  [skip] all deltas failed or sanity check failed")
            results.append({"id": sid, "y": y_i, "status": "skipped"})
        else:
            results.append({
                "id": sid,
                "y": y_i,
                "status": "ok",
                "f_x": best["f_x"],
                "bag": best["f_x"] - y_i,
                "phi_0": best["phi_0"],
                "calibration": best["calibration"],
                "instab": best["instab"],
                "loss": best["loss"],
                "best_delta": best["best_delta"],
                "local_density": best["local_density"],
                "residual": best["residual"],
                "shap_values": best["shap_values"],
            })
            print(
                f"  f_x={best['f_x']:.2f} phi_0={best['phi_0']:.2f} "
                f"BAG={best['f_x']-y_i:+.2f} delta*={best['best_delta']} "
                f"cal={best['calibration']:.3f} instab={best['instab']:.4f}"
            )

        with open(checkpoint_path, "wb") as f:
            pickle.dump(results, f)

    
    summary_rows = []
    shap_long_rows = []
    for r in results:
        if r.get("status") != "ok":
            summary_rows.append({
                "id": r["id"], "y": r["y"], "status": r.get("status", "skipped"),
            })
            continue
        row = {k: r[k] for k in r if k != "shap_values"}
        summary_rows.append(row)
        for feat_idx, val in enumerate(r["shap_values"]):
            shap_long_rows.append({
                "id": r["id"],
                "feature": feature_names[feat_idx],
                "shap_value": float(val),
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(results_csv, index=False)
    print(f"\nWrote {results_csv}")

    if shap_long_rows:
        shap_long_df = pd.DataFrame(shap_long_rows)
        shap_long_df.to_parquet(shap_long_parquet, index=False)
        print(f"Wrote {shap_long_parquet}")


if __name__ == "__main__":
    main()
