"""
Takes any hackathon CSV dataset (sequence and label columns), runs Stratified
Cross-Validation reporting MCC, ROC-AUC, and F1, trains the final model
(RandomForest + LightGBM soft-voting ensemble), and saves acp_model.joblib.
"""

import argparse
import warnings

import joblib
import numpy as np
import pandas as pd
from baseline import (
    DEFAULT_MODEL_PATH,
    clean_sequence,
    extract_esm2_embeddings,
    extract_physicochemical,
)
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings(
    "ignore", message="X does not have valid feature names"
)


def build_feature_matrix(sequences: list[str]) -> np.ndarray:
    print(
        f"Extracting biophysical descriptors for {len(sequences)} peptides..."
    )
    bio_df = pd.DataFrame([extract_physicochemical(s) for s in sequences])

    print("Extracting ESM-2 protein language model embeddings...")
    esm_embs = extract_esm2_embeddings(sequences)

    return np.hstack([bio_df.values, esm_embs])


def make_ensemble(n_train_samples: int) -> VotingClassifier:
    min_child_samples = max(1, min(20, n_train_samples // 5))
    return VotingClassifier(
        estimators=[
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=300, random_state=42, n_jobs=-1
                ),
            ),
            (
                "lgbm",
                LGBMClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    min_child_samples=min_child_samples,
                    random_state=42,
                    verbose=-1,
                ),
            ),
        ],
        voting="soft",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train Anticancer Peptide Classifier"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to CSV with sequence & label columns",
    )
    parser.add_argument(
        "--seq-col",
        type=str,
        default="sequence",
        help="Name of sequence column",
    )
    parser.add_argument(
        "--label-col",
        type=str,
        default="label",
        help="Name of binary target column (0/1)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Output model filename",
    )
    args = parser.parse_args()

    df = (
        pd.read_csv(args.data)
        .dropna(subset=[args.seq_col, args.label_col])
        .copy()
    )
    df[args.seq_col] = df[args.seq_col].astype(str).map(clean_sequence)
    df = df[df[args.seq_col].str.len() > 0].reset_index(drop=True)

    if df.empty:
        raise ValueError(
            "No valid canonical amino acid sequences found in the dataset."
        )

    sequences = df[args.seq_col].tolist()
    y = df[args.label_col].astype(int).values

    unique_classes, class_counts = np.unique(y, return_counts=True)
    if len(unique_classes) != 2:
        found = unique_classes.tolist()
        raise ValueError(
            f"Expected binary labels (2 classes), found: {found}"
        )

    X = build_feature_matrix(sequences)

    min_class_count = int(class_counts.min())
    n_splits = min(5, min_class_count)

    if n_splits >= 2:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        mccs, aucs, f1s = [], [], []

        for train_idx, val_idx in skf.split(X, y):
            model = make_ensemble(len(train_idx))
            model.fit(X[train_idx], y[train_idx])
            probs = model.predict_proba(X[val_idx])[:, 1]
            preds = (probs >= 0.5).astype(int)

            mccs.append(matthews_corrcoef(y[val_idx], preds))
            aucs.append(roc_auc_score(y[val_idx], probs))
            f1s.append(f1_score(y[val_idx], preds, zero_division=0))

        print(f"\n--- {n_splits}-Fold Cross-Validation Results ---")
        print(f"ROC-AUC : {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
        print(f"MCC     : {np.mean(mccs):.4f} ± {np.std(mccs):.4f}")
        print(f"F1-Score: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    else:
        print(
            f"\nSkipping CV (minimum class count {min_class_count} < 2); "
            "fitting on full dataset."
        )

    # Fit on 100% of data for final submission & demo
    final_model = make_ensemble(len(X))
    final_model.fit(X, y)
    joblib.dump(final_model, args.out)
    print(f"\nSaved full-dataset ensemble model to {args.out}")


if __name__ == "__main__":
    main()

