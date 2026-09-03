"""
Split df_train.xlsx into background pool (80%) + validation (20%).
Random proportional split with fixed seed. No stratification (by design:
the skewed age distribution of df_train is the phenomenon under study).

Outputs:
    data/df_background.xlsx  -> pool for SHAP background extraction
    data/df_val.xlsx         -> validation set for lambda selection (Pareto)
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
VAL_FRACTION = 0.20

base_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_path, "data")

train_path = os.path.join(data_path, "df_train.xlsx")
background_path = os.path.join(data_path, "df_background.xlsx")
val_path = os.path.join(data_path, "df_val.xlsx")


def describe_age(df, label):
    age = df["age"]
    print(
        f"{label:12s} n={len(df):5d} | "
        f"age min={age.min():5.1f} max={age.max():5.1f} "
        f"mean={age.mean():5.1f} std={age.std():5.1f}"
    )


def main():
    print(f"Loading {train_path}")
    df_train = pd.read_excel(train_path)
    print(f"Loaded {len(df_train)} subjects, {df_train.shape[1]} columns")

    if "age" not in df_train.columns:
        raise ValueError("Column 'age' not found in df_train.xlsx")

    df_bg, df_val = train_test_split(
        df_train,
        test_size=VAL_FRACTION,
        random_state=SEED,
        shuffle=True,
    )

    print("\n--- Split summary ---")
    describe_age(df_train, "original")
    describe_age(df_bg, "background")
    describe_age(df_val, "validation")

    print("\n--- Age distribution by decade (counts) ---")
    bins = list(range(0, 101, 10))
    labels = [f"{b}-{b+9}" for b in bins[:-1]]
    orig_counts = pd.cut(df_train["age"], bins=bins, labels=labels, right=False).value_counts().sort_index()
    bg_counts = pd.cut(df_bg["age"], bins=bins, labels=labels, right=False).value_counts().sort_index()
    val_counts = pd.cut(df_val["age"], bins=bins, labels=labels, right=False).value_counts().sort_index()
    dist = pd.DataFrame({
        "original": orig_counts,
        "background": bg_counts,
        "validation": val_counts,
    })
    print(dist.to_string())

    empty_in_val = dist.index[(dist["original"] > 0) & (dist["validation"] == 0)].tolist()
    if empty_in_val:
        print(
            f"\n[warn] Age bins present in original but empty in validation: {empty_in_val}. "
            "Validation does not cover these ages — interpret lambda selection accordingly."
        )

    df_bg.to_excel(background_path, index=False)
    df_val.to_excel(val_path, index=False)
    print(f"\nWrote {background_path}")
    print(f"Wrote {val_path}")


if __name__ == "__main__":
    np.random.seed(SEED)
    main()
