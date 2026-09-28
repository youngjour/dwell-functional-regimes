"""Step 06 - build HMM model-input matrices from emission table (READ-ONLY).

Transforms:
  - mix(16) and dwell_share(6): each its own composition -> multiplicative
    zero-replacement (delta) -> CLR. short_share/long_share excluded (derived).
  - presence_log, stay_vol_log: z-score with scaler.json (train-only stats).
  - dwell missing (dwell_present=False): dwell CLR block + stay_vol_log_z imputed
    with TRAIN mean, plus `dwell_missing` indicator (1). mix missing
    (presence_demo_masked, <1%): mix CLR imputed with train mean + `mix_missing`.

Sequences = per (cell_id, date) daily cycle of up to 19 time_bins, time-ordered.
For tractable K-sweep we SUBSAMPLE day-sequences (seeded); imputation/scaler stats
are train-only so no leakage. Outputs -> results/models/input_<hash>/.

Run: python src/06_build_input_ksweep.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

SEED = C.SEED
DELTA = 1e-3                 # multiplicative zero-replacement detection limit
N_TRAIN_SEQ = 30_000        # sampled (cell,date) day-sequences for fitting
N_HOLD_SEQ = 12_000
TB_ORDER = ["00"] + [f"{h:02d}" for h in range(6, 24)]   # 19, chronological
TB_RANK = {b: i for i, b in enumerate(TB_ORDER)}

MIX = [f"mix_{s}{b}" for s in ["M", "F"]
       for b in ["00", "10", "20", "30", "40", "50", "60", "70"]]   # 16
DWELL = [f"dwell_share_{c}" for c in ["000", "030", "060", "120", "180", "240"]]  # 6

CLR_MIX = [f"clr_mix_{m.split('_', 1)[1]}" for m in MIX]
CLR_DWELL = [f"clr_dwell_{c}" for c in ["000", "030", "060", "120", "180", "240"]]
FEATURE_COLS = (["presence_log_z", "stay_vol_log_z"] + CLR_MIX + CLR_DWELL
                + ["dwell_missing", "mix_missing"])   # 2+16+6+2 = 26


def mult_replace_clr(P: np.ndarray, delta: float) -> np.ndarray:
    """Multiplicative zero-replacement then CLR. P: (n,D) rows sum~1, may have 0."""
    P = P.astype(np.float64).copy()
    Z = (P == 0).sum(axis=1, keepdims=True)               # zeros per row
    nonzero = P > 0
    P = np.where(P == 0, delta, P * (1.0 - Z * delta))     # replace + shrink
    P = P / P.sum(axis=1, keepdims=True)                   # re-close to 1
    L = np.log(P)
    return L - L.mean(axis=1, keepdims=True)               # CLR


def sample_keys(emission_dir: Path) -> pl.DataFrame:
    keys = (pl.scan_parquet(str(emission_dir / "year_month=*" / "*.parquet"))
            .select(["cell_id", "date", "split"]).unique()
            .collect(engine="streaming"))
    parts = []
    for split, n in [("train", N_TRAIN_SEQ), ("holdout", N_HOLD_SEQ)]:
        sub = keys.filter(pl.col("split") == split)
        n = min(n, sub.height)
        parts.append(sub.sample(n=n, seed=SEED))
    return pl.concat(parts)


def main():
    cfg = {"seed": SEED, "delta": DELTA, "n_train_seq": N_TRAIN_SEQ,
           "n_hold_seq": N_HOLD_SEQ, "features": FEATURE_COLS}
    h = hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]
    out = C.MODELS_DIR / f"input_{h}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"input hash={h} -> {out}")

    scaler = json.loads((C.PROCESSED_DIR / "scaler.json").read_text("utf-8"))
    pm, ps = (scaler["features"]["presence_log"]["mean"],
              scaler["features"]["presence_log"]["std"])
    sm, ss = (scaler["features"]["stay_vol_log"]["mean"],
              scaler["features"]["stay_vol_log"]["std"])

    em_dir = C.PROCESSED_DIR / "emission"
    keys = sample_keys(em_dir)
    print(f"sampled sequences: train={keys.filter(pl.col('split')=='train').height:,}"
          f" holdout={keys.filter(pl.col('split')=='holdout').height:,}")

    cols = (["cell_id", "date", "time_bin", "split", "presence_log",
             "stay_vol_log", "presence_demo_masked", "dwell_present",
             "short_share", "long_share"] + MIX + DWELL)
    df = (pl.scan_parquet(str(em_dir / "year_month=*" / "*.parquet"))
          .select(cols)
          .join(keys.lazy(), on=["cell_id", "date", "split"], how="inner")
          .with_columns(pl.col("time_bin").replace_strict(TB_RANK).alias("tb_rank"))
          .sort(["split", "cell_id", "date", "tb_rank"])
          .collect(engine="streaming"))
    print(f"sampled rows: {df.height:,}")

    # ---- numpy blocks ----
    mix = df.select(MIX).to_numpy()
    dwell = df.select(DWELL).to_numpy()
    presence_log = df["presence_log"].to_numpy().astype(np.float64)
    stay_vol_log = df["stay_vol_log"].to_numpy().astype(np.float64)
    is_train = (df["split"] == "train").to_numpy()

    mix_missing = (df["presence_demo_masked"].to_numpy()
                   | np.isnan(mix).any(axis=1))
    dwell_missing = (~df["dwell_present"].to_numpy()
                     | np.isnan(dwell).any(axis=1))

    # CLR on valid rows; placeholder zeros elsewhere
    clr_mix = np.zeros((df.height, len(MIX)))
    clr_mix[~mix_missing] = mult_replace_clr(mix[~mix_missing], DELTA)
    clr_dwell = np.zeros((df.height, len(DWELL)))
    clr_dwell[~dwell_missing] = mult_replace_clr(dwell[~dwell_missing], DELTA)

    # TRAIN means for imputation (train-only, no leakage)
    tr_mix_ok = is_train & ~mix_missing
    tr_dwell_ok = is_train & ~dwell_missing
    mix_mean = clr_mix[tr_mix_ok].mean(axis=0)
    dwell_mean = clr_dwell[tr_dwell_ok].mean(axis=0)
    clr_mix[mix_missing] = mix_mean
    clr_dwell[dwell_missing] = dwell_mean

    presence_z = (presence_log - pm) / ps                  # presence never missing
    stay_z = (stay_vol_log - sm) / ss
    stay_z[dwell_missing] = 0.0                             # train-mean (z=0)

    X = np.column_stack([
        presence_z, stay_z, clr_mix, clr_dwell,
        dwell_missing.astype(np.float64), mix_missing.astype(np.float64),
    ]).astype(np.float32)
    assert X.shape[1] == len(FEATURE_COLS), (X.shape, len(FEATURE_COLS))

    # lengths per (cell_id,date) within each split (rows already sorted)
    for split in ["train", "holdout"]:
        m = (df["split"] == split).to_numpy()
        Xs = X[m]
        sub = df.filter(pl.col("split") == split)
        lengths = (sub.group_by(["cell_id", "date"], maintain_order=True)
                   .agg(pl.len().alias("n"))["n"].to_numpy().astype(np.int64))
        assert lengths.sum() == Xs.shape[0]
        np.save(out / f"{split}_X.npy", Xs)
        np.save(out / f"{split}_lengths.npy", lengths)
        # keep raw cols (same order) for interpretation
        sub.select(["cell_id", "date", "time_bin", "tb_rank", "presence_log",
                    "stay_vol_log", "short_share", "long_share",
                    "dwell_present", *MIX, *DWELL]).write_parquet(
            out / f"{split}_meta.parquet")
        print(f"{split}: X={Xs.shape} seqs={len(lengths)} "
              f"meanlen={lengths.mean():.1f}")

    meta = {
        **cfg, "input_hash": h,
        "clr_delta": DELTA, "scaler": {"presence_log": [pm, ps],
                                       "stay_vol_log": [sm, ss]},
        "mix_clr_train_mean": mix_mean.tolist(),
        "dwell_clr_train_mean": dwell_mean.tolist(),
        "mix_parts": MIX, "dwell_parts": DWELL,
        "missing_rates": {
            "mix_missing_all": float(mix_missing.mean()),
            "dwell_missing_all": float(dwell_missing.mean()),
            "mix_missing_train": float(mix_missing[is_train].mean()),
            "dwell_missing_train": float(dwell_missing[is_train].mean()),
        },
    }
    (out / "feature_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    (C.MODELS_DIR / "latest_input.txt").write_text(h, encoding="utf-8")
    print("missing rates:", json.dumps(meta["missing_rates"], indent=2))
    print("DONE", h)


if __name__ == "__main__":
    main()
