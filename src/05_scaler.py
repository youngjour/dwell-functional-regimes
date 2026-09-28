"""Step 05 - train-only scaler for the emission table.

- data/processed/scaler.json: train-only mean/std for continuous magnitude features
  (presence_log, stay_vol_log). Compositional features (mix, dwell shares) are
  NOT standardized here (CLR transform is applied at the model-input stage).

Run: python src/05_scaler.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

C.set_seed()

EM = C.PROCESSED_DIR / "emission"
DUR = ["000", "030", "060", "120", "180", "240"]
MIX_COLS = [f"mix_{s}{b}" for s in ["M", "F"]
            for b in ["00", "10", "20", "30", "40", "50", "60", "70"]]
SCALE_FEATS = ["presence_log", "stay_vol_log"]


def lf_em():
    return pl.scan_parquet(str(EM / "year_month=*" / "*.parquet"))


def build_scaler():
    train = lf_em().filter(pl.col("split") == "train")
    stats = train.select(
        *[pl.col(f).mean().alias(f"{f}__mean") for f in SCALE_FEATS],
        *[pl.col(f).std().alias(f"{f}__std") for f in SCALE_FEATS],
        *[pl.col(f).count().alias(f"{f}__n") for f in SCALE_FEATS],
    ).collect()
    s = stats.row(0, named=True)
    scaler = {
        "method": "z-score (train-only fit)",
        "train_months": "2025.01-2026.03",
        "features": {f: {"mean": float(s[f"{f}__mean"]),
                         "std": float(s[f"{f}__std"]),
                         "n_train": int(s[f"{f}__n"])} for f in SCALE_FEATS},
        "not_standardized": {
            "mix(16)": "compositional, sums to 1; logit/CLR deferred to model stage",
            "dwell_share(6)": "compositional, sums to 1; deferred",
            "short_share/long_share": "derived from dwell_share; deferred",
        },
    }
    (C.PROCESSED_DIR / "scaler.json").write_text(
        json.dumps(scaler, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote scaler.json")
    return scaler


def main():
    build_scaler()


if __name__ == "__main__":
    main()
