"""
Select lambda* via Pareto analysis on the validation set.

For each validation subject, for each delta in DELTA_GRID, run bootstrap SHAP
(this does NOT depend on lambda). Then, for each candidate lambda, compute
per-subject best-delta using the loss
        loss(delta) = |phi_0 - y_subject| + lambda * instab
and aggregate calibration and instability across validation -> one Pareto point.

Outputs
-------
    results/validation_intermediate.pkl : checkpoint (resumable)
    results/pareto_points.csv           : (lambda, cal_mean, instab_mean, n_used)
    results/lambda_star.json            : selected lambda* (knee heuristic)
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



SEED = 42
DELTA_GRID = [2, 3, 5, 7, 10, 15, 20]
LAMBDA_GRID = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0, 10.0]

base_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_path, "data")
results_path = os.path.join(base_path, "results")
os.makedirs(results_path, exist_ok=True)

background_path = os.path.join(data_path, "df_background.xlsx")
val_path = os.path.join(data_path, "df_val.xlsx")

checkpoint_path = os.path.join(results_path, "validation_intermediate.pkl")
pareto_csv = os.path.join(results_path, "pareto_points.csv")
lambda_star_json = os.path.join(results_path, "lambda_star.json")


def load_split():
    df_bg = pd.read_excel(background_path).drop(columns=["participant_id"])
    df_val_full = pd.read_excel(val_path)
    val_ids = df_val_full["participant_id"].reset_index(drop=True)
    df_val = df_val_full.drop(columns=["participant_id"]).reset_index(drop=True)

    age_col = df_bg.columns.get_loc("age")
    X_bg = df_bg.iloc[:, :age_col]
    y_bg = df_bg["age"].reset_index(drop=True)
    X_val = df_val.iloc[:, :age_col]
    y_val = df_val["age"].reset_index(drop=True)

    """ 
    scaler = MinMaxScaler().fit(X_bg)
    X_bg_s = pd.DataFrame(scaler.transform(X_bg), columns=X_bg.columns).reset_index(drop=True)
    X_val_s = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns).reset_index(drop=True)
    """
    X_bg_s = X_bg.reset_index(drop=True);
    X_val_s = X_val.reset_index(drop=True);

    return X_bg_s, y_bg, X_val_s, y_val, val_ids


def run_validation(model, X_bg, y_bg, X_val, y_val, val_ids):
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "rb") as f:
            per_subject = pickle.load(f)
        start_idx = len(per_subject)
        print(f"[resume] starting at validation subject {start_idx + 1}/{len(X_val)}")
    else:
        per_subject = []
        start_idx = 0

    for i in range(start_idx, len(X_val)):
        x_i = X_val.iloc[i].values
        y_i = float(y_val.iloc[i])
        sid = val_ids.iloc[i]
        print(f"[val] subject {i + 1}/{len(X_val)} id={sid} age={y_i:.1f}")

        per_delta = {}
        for delta in DELTA_GRID:
            bg = extract_background(delta, y_i, X_bg, y_bg)
            if len(bg) == 0:
                per_delta[delta] = None
                continue
            seed = SEED + i * 100 + int(delta)
            res = compute_subject_delta(
                model, x_i, y_i, bg,
                K=K_BOOTSTRAP, summary_size=SUMMARY_SIZE, seed=seed,
            )
            per_delta[delta] = res
            if res is not None and not res["sanity_ok"]:
                print(f"  [sanity] delta={delta} residual={res['residual']:.4e} FAILED")

        per_subject.append({
            "id": sid,
            "y": y_i,
            "per_delta": per_delta,
        })

        with open(checkpoint_path, "wb") as f:
            pickle.dump(per_subject, f)

        
    return per_subject


def pareto_from_results(per_subject):
    rows = []
    for lam in LAMBDA_GRID:
        cals = []
        instabs = []
        deltas = []
        skipped = 0
        for subj in per_subject:
            valid = [
                (d, r) for d, r in subj["per_delta"].items()
                if r is not None and r["sanity_ok"]
            ]
            if not valid:
                skipped += 1
                continue
            best_d, best_r = min(
                valid,
                key=lambda dr: dr[1]["calibration"] + lam * dr[1]["instab"],
            )
            cals.append(best_r["calibration"])
            instabs.append(best_r["instab"])
            deltas.append(best_d)
        rows.append({
            "lambda": lam,
            "calibration_mean": float(np.mean(cals)) if cals else np.nan,
            "instability_mean": float(np.mean(instabs)) if instabs else np.nan,
            "delta_median": float(np.median(deltas)) if deltas else np.nan,
            "n_used": len(cals),
            "n_skipped": skipped,
        })
    return pd.DataFrame(rows)


def pick_knee(pareto_df):
    """Normalize both axes to [0,1], pick lambda at min distance to the origin."""
    df = pareto_df.dropna(subset=["calibration_mean", "instability_mean"]).copy()
    if len(df) == 0:
        return None
    c = df["calibration_mean"]
    s = df["instability_mean"]
    df["cal_norm"] = (c - c.min()) / (c.max() - c.min() + 1e-12)
    df["inst_norm"] = (s - s.min()) / (s.max() - s.min() + 1e-12)
    df["dist_origin"] = np.sqrt(df["cal_norm"] ** 2 + df["inst_norm"] ** 2)
    return float(df.loc[df["dist_origin"].idxmin(), "lambda"])

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

def wrap_model(model):
    
    def model_predict(x):
        return model(x)

    def predict_fn(x):
        return model_predict(tf.convert_to_tensor(x)).numpy()

    return predict_fn

def main():
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    X_bg, y_bg, X_val, y_val, val_ids = load_split()
    print(f"Background pool: {len(X_bg)} | Validation: {len(X_val)}")
    print(f"Delta grid: {DELTA_GRID}")
    print(f"Lambda grid (Pareto): {LAMBDA_GRID}")
    print(f"K bootstrap: {K_BOOTSTRAP} | summary size: {SUMMARY_SIZE}")

    
    
    model = load_custom_model(data_path)
    predict_fn = wrap_model(model)
    
    per_subject = run_validation(predict_fn, X_bg, y_bg, X_val, y_val, val_ids)

    pareto_df = pareto_from_results(per_subject)
    pareto_df.to_csv(pareto_csv, index=False)
    print("\n--- Pareto points ---")
    print(pareto_df.to_string(index=False))

    lambda_star = pick_knee(pareto_df)
    print(f"\nSelected lambda* = {lambda_star} (knee heuristic)")
    with open(lambda_star_json, "w") as f:
        json.dump(
            {
                "lambda_star": lambda_star,
                "pareto": pareto_df.to_dict(orient="records"),
                "delta_grid": DELTA_GRID,
                "lambda_grid": LAMBDA_GRID,
                "seed": SEED,
                "K_bootstrap": K_BOOTSTRAP,
                "summary_size": SUMMARY_SIZE,
            },
            f,
            indent=2,
        )
    print(f"Wrote {lambda_star_json}")


if __name__ == "__main__":
    main()
