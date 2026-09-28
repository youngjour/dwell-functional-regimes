"""Step 08 - model-input v2 for the K=7 ladder (z-scored CLR, full-19 diurnal seqs).

Same transform recipe as src/06 (CLR delta=1e-3, train-only scaler, missing->train
mean + indicator) PLUS train-only z-score on clr_mix_*/clr_dwell_*. Restricts to
COMPLETE diurnal cycles (exactly 19 time_bins) so the HSMM/Viterbi batch is a clean
(N, 19, D) tensor shared by all ladder models. holdout kept split-able by month.

Outputs -> results/models/input_v2_<hash>/ :
  {split}_X3d.npy (N,19,D) float32, {split}_meta.parquet (N*19 rows, ordered),
  feature_meta.json. holdout meta carries year_month for 2026.04 vs 2026.05 split.

Run: python src/08_build_input_ladder.py [N_SEQ]
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
DELTA = 1e-3
TB_ORDER = ["00"] + [f"{h:02d}" for h in range(6, 24)]   # 19
TB_RANK = {b: i for i, b in enumerate(TB_ORDER)}
T = len(TB_ORDER)
N_SEQ_DEFAULT = 120_000

MIX = [f"mix_{s}{b}" for s in ["M", "F"]
       for b in ["00", "10", "20", "30", "40", "50", "60", "70"]]
DWELL = [f"dwell_share_{c}" for c in ["000", "030", "060", "120", "180", "240"]]
CLR_MIX = [f"clr_mix_{m.split('_', 1)[1]}_z" for m in MIX]
CLR_DWELL = [f"clr_dwell_{c}_z" for c in ["000", "030", "060", "120", "180", "240"]]
FEATURE_COLS = (["presence_log_z", "stay_vol_log_z"] + CLR_MIX + CLR_DWELL
                + ["dwell_missing", "mix_missing"])   # 26


def mult_replace_clr(P, delta):
    P = P.astype(np.float64).copy()
    Z = (P == 0).sum(axis=1, keepdims=True)
    P = np.where(P == 0, delta, P * (1.0 - Z * delta))
    P = P / P.sum(axis=1, keepdims=True)
    L = np.log(P)
    return L - L.mean(axis=1, keepdims=True)


def main():
    n_seq = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() \
        else N_SEQ_DEFAULT
    cfg = {"seed": SEED, "delta": DELTA, "n_seq": n_seq, "T": T,
           "features": FEATURE_COLS, "full19_only": True, "clr_zscore": True}
    h = hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]
    out = C.MODELS_DIR / f"input_v2_{h}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"v2 hash={h} N_SEQ={n_seq} -> {out}", flush=True)

    scaler = json.loads((C.PROCESSED_DIR / "scaler.json").read_text("utf-8"))
    pm, ps = scaler["features"]["presence_log"]["mean"], scaler["features"]["presence_log"]["std"]
    sm, ss = scaler["features"]["stay_vol_log"]["mean"], scaler["features"]["stay_vol_log"]["std"]

    em = pl.scan_parquet(str(C.PROCESSED_DIR / "emission" / "year_month=*" / "*.parquet"))
    # complete-cycle (cell,date): exactly 19 present bins
    full = (em.group_by(["cell_id", "date", "split"]).agg(pl.len().alias("nb"))
            .filter(pl.col("nb") == T).collect(engine="streaming"))
    parts = []
    for split, frac_n in [("train", n_seq), ("holdout", max(1, n_seq // 4))]:
        sub = full.filter(pl.col("split") == split)
        k = min(frac_n, sub.height)
        parts.append(sub.sample(n=k, seed=SEED))
        print(f"{split}: full19 avail={sub.height:,} sampled={k:,}", flush=True)
    keys = pl.concat(parts).select(["cell_id", "date", "split"])

    cols = (["cell_id", "date", "time_bin", "split", "presence_log", "stay_vol_log",
             "presence_demo_masked", "dwell_present", "short_share", "long_share"]
            + MIX + DWELL)
    df = (em.select(cols)
          .join(keys.lazy(), on=["cell_id", "date", "split"], how="inner")
          .with_columns(
              pl.col("time_bin").replace_strict(TB_RANK).alias("tb_rank"),
              pl.col("date").dt.strftime("%Y%m").alias("year_month"))
          .sort(["split", "cell_id", "date", "tb_rank"])
          .collect(engine="streaming"))
    print(f"rows={df.height:,}", flush=True)

    mix = df.select(MIX).to_numpy()
    dwell = df.select(DWELL).to_numpy()
    presence_log = df["presence_log"].to_numpy().astype(np.float64)
    stay_vol_log = df["stay_vol_log"].to_numpy().astype(np.float64)
    is_train = (df["split"] == "train").to_numpy()
    mix_missing = df["presence_demo_masked"].to_numpy() | np.isnan(mix).any(axis=1)
    dwell_missing = (~df["dwell_present"].to_numpy()) | np.isnan(dwell).any(axis=1)

    clr_mix = np.zeros((df.height, len(MIX)))
    clr_mix[~mix_missing] = mult_replace_clr(mix[~mix_missing], DELTA)
    clr_dwell = np.zeros((df.height, len(DWELL)))
    clr_dwell[~dwell_missing] = mult_replace_clr(dwell[~dwell_missing], DELTA)

    tr_mix_ok = is_train & ~mix_missing
    tr_dwell_ok = is_train & ~dwell_missing
    mix_mean = clr_mix[tr_mix_ok].mean(axis=0)
    dwell_mean = clr_dwell[tr_dwell_ok].mean(axis=0)
    clr_mix[mix_missing] = mix_mean
    clr_dwell[dwell_missing] = dwell_mean

    # train-only z-score for CLR blocks (computed over imputed train rows)
    mix_z_mean = clr_mix[is_train].mean(axis=0); mix_z_std = clr_mix[is_train].std(axis=0)
    dw_z_mean = clr_dwell[is_train].mean(axis=0); dw_z_std = clr_dwell[is_train].std(axis=0)
    clr_mix_z = (clr_mix - mix_z_mean) / np.where(mix_z_std > 0, mix_z_std, 1)
    clr_dwell_z = (clr_dwell - dw_z_mean) / np.where(dw_z_std > 0, dw_z_std, 1)

    presence_z = (presence_log - pm) / ps
    stay_z = (stay_vol_log - sm) / ss
    stay_z[dwell_missing] = 0.0

    X = np.column_stack([presence_z, stay_z, clr_mix_z, clr_dwell_z,
                         dwell_missing.astype(float), mix_missing.astype(float)]
                        ).astype(np.float32)
    assert X.shape[1] == len(FEATURE_COLS)

    for split in ["train", "holdout"]:
        m = (df["split"] == split).to_numpy()
        Xs = X[m]
        n = Xs.shape[0] // T
        assert Xs.shape[0] == n * T, (Xs.shape, T)
        X3 = Xs.reshape(n, T, X.shape[1])
        np.save(out / f"{split}_X3d.npy", X3.astype(np.float32))
        df.filter(pl.col("split") == split).select(
            ["cell_id", "date", "time_bin", "tb_rank", "year_month",
             "presence_log", "stay_vol_log", "short_share", "long_share",
             "dwell_present", *MIX, *DWELL]).write_parquet(out / f"{split}_meta.parquet")
        print(f"{split}: X3d={X3.shape}", flush=True)

    meta = {**cfg, "input_hash": h, "scaler": {"presence_log": [pm, ps],
            "stay_vol_log": [sm, ss]},
            "clr_mix_train_mean": mix_mean.tolist(), "clr_dwell_train_mean": dwell_mean.tolist(),
            "clr_mix_z": [mix_z_mean.tolist(), mix_z_std.tolist()],
            "clr_dwell_z": [dw_z_mean.tolist(), dw_z_std.tolist()],
            "mix_parts": MIX, "dwell_parts": DWELL,
            "missing_rates": {"mix_all": float(mix_missing.mean()),
                              "dwell_all": float(dwell_missing.mean())}}
    (out / "feature_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                           encoding="utf-8")
    (C.MODELS_DIR / "latest_input_v2.txt").write_text(h, encoding="utf-8")
    print("DONE", h)


if __name__ == "__main__":
    main()
