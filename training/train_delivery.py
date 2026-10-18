"""Train the delivery scorer and run the §7 ablation.

    python training/train_delivery.py --source synthetic --export
    python training/train_delivery.py --source sessions --data exported.json --export

Methodology notes that matter for the paper:

* Splits are GROUPED by speaker/session. A random row split leaks: two answers
  from the same person share accent, camera angle, room acoustics and baseline
  pitch, so a row-split model scores brilliantly and generalises to nobody.
  This is the single most common way multimodal delivery papers overstate.
* The baseline is a ridge regression on the same features, not a constant. If
  the trees can't beat a linear model, "we used gradient boosting" is not a
  finding.
* Reported alongside R^2 is the mean-prediction baseline, because R^2 on a
  narrow label distribution flatters everything.
* The ablation compares feature SUBSETS with the model class and split held
  fixed, so a difference is attributable to modality rather than tuning.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import (  # noqa: E402
    ABLATIONS, LABEL_NAMES, SCHEMA_VERSION, Dataset, load_from_sessions, load_synthetic,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"


def grouped_split(ds: Dataset, test_frac: float = 0.25, seed: int = 0):
    """Speaker-disjoint train/test split."""
    rng = np.random.default_rng(seed)
    unique = np.unique(ds.groups)
    rng.shuffle(unique)
    n_test = max(1, int(len(unique) * test_frac))
    test_groups = set(unique[:n_test].tolist())
    test_mask = np.array([g in test_groups for g in ds.groups])
    return ~test_mask, test_mask


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    resid = y_true - y_pred
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "r2": round(1 - ss_res / ss_tot, 4) if ss_tot > 0 else 0.0,
        "mae": round(float(np.mean(np.abs(resid))), 4),
        "rmse": round(float(np.sqrt(np.mean(resid**2))), 4),
    }


def fit_ridge(Xtr, ytr, Xte, alpha: float = 1.0):
    """Closed-form ridge -- the honest linear baseline, no sklearn needed."""
    mu, sigma = Xtr.mean(0), Xtr.std(0) + 1e-9
    Z = (Xtr - mu) / sigma
    Z = np.hstack([Z, np.ones((Z.shape[0], 1))])
    A = Z.T @ Z + alpha * np.eye(Z.shape[1])
    # Un-penalise the intercept by removing alpha from its diagonal entry.
    # Zeroing the entry outright (the obvious-looking mistake) destroys the
    # intercept's normal equation and the whole fit collapses -- it shows up as
    # a large negative R^2 that is suspiciously identical across feature sets.
    A[-1, -1] -= alpha
    w = np.linalg.solve(A, Z.T @ ytr)
    Zte = np.hstack([(Xte - mu) / sigma, np.ones((Xte.shape[0], 1))])
    return np.clip(Zte @ w, 0, 1)


def fit_lightgbm(Xtr, ytr, Xte, feature_names, seed: int = 0):
    import lightgbm as lgb

    train_set = lgb.Dataset(Xtr, label=ytr, feature_name=list(feature_names))
    params = {
        "objective": "regression",
        "metric": "l2",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "verbose": -1,
        "seed": seed,
    }
    booster = lgb.train(params, train_set, num_boost_round=300)
    return np.clip(booster.predict(Xte), 0, 1), booster


def run_ablation(ds: Dataset, seed: int = 0) -> dict:
    tr, te = grouped_split(ds, seed=seed)
    print(f"\ntrain rows {tr.sum()}  test rows {te.sum()}  "
          f"(speaker-disjoint, {len(np.unique(ds.groups))} groups)\n")

    results: dict[str, dict] = {}
    for name, feats in ABLATIONS.items():
        sub = ds.subset(feats)
        Xtr, Xte = sub.X[tr], sub.X[te]
        per_label = {}
        for label in LABEL_NAMES:
            ytr, yte = ds.y[label][tr], ds.y[label][te]
            baseline = np.full_like(yte, ytr.mean())
            ridge = fit_ridge(Xtr, ytr, Xte)
            try:
                gbm, _ = fit_lightgbm(Xtr, ytr, Xte, feats, seed)
            except ImportError:
                gbm = ridge
            per_label[label] = {
                "mean_baseline": metrics(yte, baseline),
                "ridge": metrics(yte, ridge),
                "lightgbm": metrics(yte, gbm),
            }
        results[name] = {
            "n_features": len(feats),
            "per_label": per_label,
            "mean_r2_lightgbm": round(
                float(np.mean([per_label[l]["lightgbm"]["r2"] for l in LABEL_NAMES])), 4
            ),
            "mean_r2_ridge": round(
                float(np.mean([per_label[l]["ridge"]["r2"] for l in LABEL_NAMES])), 4
            ),
        }
    return results


def print_table(results: dict) -> None:
    print("=" * 78)
    print("ABLATION -- mean R^2 across the four delivery sub-scores (held-out speakers)")
    print("=" * 78)
    print(f"{'feature set':<20}{'n':>4}{'ridge':>10}{'lightgbm':>11}{'delta':>9}")
    print("-" * 78)
    for name, r in results.items():
        delta = r["mean_r2_lightgbm"] - r["mean_r2_ridge"]
        print(f"{name:<20}{r['n_features']:>4}{r['mean_r2_ridge']:>10.3f}"
              f"{r['mean_r2_lightgbm']:>11.3f}{delta:>+9.3f}")
    print("-" * 78)

    fused = results["fused"]["mean_r2_lightgbm"]
    text = results["text_meta_only"]["mean_r2_lightgbm"]
    audio = results["audio_only"]["mean_r2_lightgbm"]
    av = results["audio_plus_vision"]["mean_r2_lightgbm"]
    print(f"\nfused - text_meta_only    : {fused - text:+.3f}   (does listening help at all?)")
    print(f"fused - audio_only        : {fused - audio:+.3f}   (does VISION add over audio?)")
    print(f"fused - audio_plus_vision : {fused - av:+.3f}   (do text meta-features add?)")
    print(
        "\nThe second line is the paper's real question. Audio prosody beating\n"
        "text meta-features is unsurprising. Whether GAZE AND POSTURE add\n"
        "anything on top of prosody is the contested claim, and it is the one\n"
        "an honest ablation is built to be able to answer 'no' to.\n"
    )

    print("Per-label detail (lightgbm R^2):")
    print(f"{'feature set':<20}" + "".join(f"{l:>13}" for l in LABEL_NAMES))
    for name, r in results.items():
        row = "".join(f"{r['per_label'][l]['lightgbm']['r2']:>13.3f}" for l in LABEL_NAMES)
        print(f"{name:<20}{row}")


def export(ds: Dataset, seed: int = 0) -> None:
    """Fit final per-sub-score boosters on ALL data and write serving artefacts."""
    try:
        import lightgbm as lgb  # noqa: F401
    except ImportError:
        sys.exit("lightgbm is required to export. pip install -r backend/requirements-scoring.txt")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    base = MODELS_DIR / "delivery_scorer.txt"

    for label in LABEL_NAMES:
        _, booster = fit_lightgbm(ds.X, ds.y[label], ds.X[:1], ds.feature_names, seed)
        booster.save_model(str(base.with_suffix(f".{label}.txt")))

    # The runtime loads `delivery_scorer.txt` as the existence check and reads
    # the sidecar metadata for the schema guard.
    base.write_text("# see delivery_scorer.<subscore>.txt\n", encoding="utf-8")
    base.with_suffix(".meta.json").write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "feature_order": ds.feature_names,
                "labels": LABEL_NAMES,
                "n_train_rows": len(ds),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nexported to {MODELS_DIR}")
    print("restart the backend; it will log 'Delivery scorer: lightgbm'")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["synthetic", "sessions"], default="synthetic")
    p.add_argument("--data", type=Path, help="JSON export, required for --source sessions")
    p.add_argument("--n", type=int, default=4000, help="synthetic sample count")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--export", action="store_true", help="write serving artefacts")
    p.add_argument("--out", type=Path, help="write ablation results as JSON")
    args = p.parse_args()

    if args.source == "synthetic":
        print(f"loading synthetic dataset (n={args.n})")
        print("  NOTE: synthetic data validates the pipeline, it is not a result.")
        ds = load_synthetic(args.n, args.seed)
    else:
        if not args.data:
            sys.exit("--data is required with --source sessions")
        print(f"loading session export: {args.data}")
        ds = load_from_sessions(args.data)
    print(f"  {len(ds)} rows, {ds.X.shape[1]} features, schema v{SCHEMA_VERSION}")

    results = run_ablation(ds, args.seed)
    print_table(results)

    if args.out:
        args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    if args.export:
        export(ds, args.seed)


if __name__ == "__main__":
    main()
