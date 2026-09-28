"""Step 11 - decode the WHOLE universe with the adopted HSMM K=7 (no refit).

Applies the EXACT training transform (params from input_v2 feature_meta) to every
full-19 (cell,date) diurnal cycle of the universe (6,357 cells), runs HSMM Viterbi,
and aggregates per-cell regime summaries + diurnal profile + admin discordance.

Headline aggregates use the TRAIN period (2025.01-2026.03) to avoid the 2026.05 stay
export break; holdout (2026.04-05) kept for validation only.

Structural (non-functional) states := emission dwell_missing mean >= 0.5 (data-driven;
HSMM yields ONE such state s3, unlike HMM's two).

Outputs: data/processed/cell_regime_summary.parquet, results/stats/regime_diurnal.json,
results/stats/regime_admin_discordance.json.

Run: python src/11_decode_regimes.py
"""
from __future__ import annotations

import importlib.util
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

_e = importlib.util.spec_from_file_location("edh", str(Path(__file__).parent / "_edhsmm.py"))
edh = importlib.util.module_from_spec(_e); _e.loader.exec_module(edh)

LADDER = C.MODELS_DIR / "ladder_1455c0d960"
INPUT_V2 = C.MODELS_DIR / "input_v2_1455c0d960"
TB_ORDER = ["00"] + [f"{h:02d}" for h in range(6, 24)]
TB_RANK = {b: i for i, b in enumerate(TB_ORDER)}
T = len(TB_ORDER)
DAY_BINS = [TB_RANK[b] for b in ["10", "11", "12", "13", "14", "15", "16"]]
NIGHT_BINS = [TB_RANK[b] for b in ["00", "22", "23"]]
DELTA = 1e-3
TRAIN_MAX = "202603"
K = 7

MIX = [f"mix_{s}{b}" for s in ["M", "F"]
       for b in ["00", "10", "20", "30", "40", "50", "60", "70"]]
DWELL = [f"dwell_share_{c}" for c in ["000", "030", "060", "120", "180", "240"]]

# semantic regime labels (from HSMM_emission.csv)
REGIME_NAME = {0: "low_density", 1: "activity_A", 2: "dense_mixed",
               3: "structural_missing", 4: "activity_B", 5: "residential",
               6: "mid_density"}


def mult_replace_clr(P, delta):
    P = P.astype(np.float64).copy()
    Z = (P == 0).sum(1, keepdims=True)
    P = np.where(P == 0, delta, P * (1.0 - Z * delta))
    P = P / P.sum(1, keepdims=True)
    Lg = np.log(P)
    return Lg - Lg.mean(1, keepdims=True)


def load_model():
    hd = pickle.load(open(LADDER / "HSMM.pkl", "rb"))
    m = edh.EDHSMM(K)
    m.means_ = hd["means_"]; m.vars_ = hd["vars_"]; m.transmat_ = hd["transmat_"]
    m.startprob_ = hd["startprob_"]; m.dur_pmf_ = hd["dur_pmf_"]
    m.Dmax = hd["dur_pmf_"].shape[1]
    m.dur_surv_ = np.cumsum(hd["dur_pmf_"][:, ::-1], 1)[:, ::-1]
    m.dur_mean_ = hd["dur_mean_"]
    return m


def build_X(df, meta):
    """Apply the SAVED training transform to a month frame -> (N,T,26)."""
    mix = df.select(MIX).to_numpy(); dwell = df.select(DWELL).to_numpy()
    plog = df["presence_log"].to_numpy().astype(float)
    slog = df["stay_vol_log"].to_numpy().astype(float)
    mix_missing = df["presence_demo_masked"].to_numpy() | np.isnan(mix).any(1)
    dwell_missing = (~df["dwell_present"].to_numpy()) | np.isnan(dwell).any(1)
    clr_mix = np.zeros((df.height, len(MIX)))
    clr_mix[~mix_missing] = mult_replace_clr(mix[~mix_missing], DELTA)
    clr_dwell = np.zeros((df.height, len(DWELL)))
    clr_dwell[~dwell_missing] = mult_replace_clr(dwell[~dwell_missing], DELTA)
    clr_mix[mix_missing] = np.array(meta["clr_mix_train_mean"])
    clr_dwell[dwell_missing] = np.array(meta["clr_dwell_train_mean"])
    mzc, mzs = np.array(meta["clr_mix_z"][0]), np.array(meta["clr_mix_z"][1])
    dzc, dzs = np.array(meta["clr_dwell_z"][0]), np.array(meta["clr_dwell_z"][1])
    clr_mix_z = (clr_mix - mzc) / np.where(mzs > 0, mzs, 1)
    clr_dwell_z = (clr_dwell - dzc) / np.where(dzs > 0, dzs, 1)
    pm, ps = meta["scaler"]["presence_log"]; sm, ss = meta["scaler"]["stay_vol_log"]
    pz = (plog - pm) / ps; sz = (slog - sm) / ss; sz[dwell_missing] = 0.0
    X = np.column_stack([pz, sz, clr_mix_z, clr_dwell_z,
                         dwell_missing.astype(float), mix_missing.astype(float)]
                        ).astype(np.float32)
    return X.reshape(-1, T, X.shape[1])


def main():
    meta = json.loads((INPUT_V2 / "feature_meta.json").read_text("utf-8"))
    model = load_model()
    struct = [k for k in range(K) if model.means_[k, 24] >= 0.5]
    func = [k for k in range(K) if k not in struct]
    print(f"structural states={struct} functional={func}", flush=True)

    cm = pl.read_parquet(C.PROCESSED_DIR / "cell_master.parquet")
    uni = cm.filter(pl.col("in_universe"))
    cells = uni["cell_id"].to_list()
    cidx = {c: i for i, c in enumerate(cells)}
    n = len(cells)

    # accumulators (train)
    cnt_all = np.zeros((n, K)); cnt_day = np.zeros((n, K)); cnt_night = np.zeros((n, K))
    cnt_all_ho = np.zeros((n, K))
    tb_reg = np.zeros((T, K))            # diurnal P(regime|time_bin), train
    tb_reg_ho = np.zeros((T, K))

    cols = (["cell_id", "date", "time_bin", "split", "presence_log", "stay_vol_log",
             "presence_demo_masked", "dwell_present"] + MIX + DWELL)
    uni_set = set(cells)
    for ym in C.MONTHS:
        em = pl.scan_parquet(str(C.PROCESSED_DIR / "emission" / f"year_month={ym}" / "*.parquet"))
        em = em.filter(pl.col("cell_id").is_in(list(uni_set)))
        full = (em.group_by(["cell_id", "date"]).agg(pl.len().alias("nb"))
                .filter(pl.col("nb") == T))
        df = (em.join(full.select(["cell_id", "date"]), on=["cell_id", "date"], how="inner")
              .select(cols)
              .with_columns(pl.col("time_bin").replace_strict(TB_RANK).alias("tbr"))
              .sort(["cell_id", "date", "tbr"]).collect(engine="streaming"))
        if df.height == 0:
            continue
        X3 = build_X(df, meta)
        lab = model.predict(X3)                      # (Nseq, T)
        keys = df.select(["cell_id", "date"]).unique(maintain_order=True)
        seq_cell = keys["cell_id"].to_list()
        ci = np.array([cidx[c] for c in seq_cell])
        is_train = ym <= TRAIN_MAX
        # accumulate
        for ti in range(T):
            reg = lab[:, ti]
            tgt_tb = tb_reg if is_train else tb_reg_ho
            np.add.at(tgt_tb, (ti, reg), 1)
            tgt = cnt_all if is_train else cnt_all_ho
            np.add.at(tgt, (ci, reg), 1)
            if is_train and ti in DAY_BINS:
                np.add.at(cnt_day, (ci, reg), 1)
            if is_train and ti in NIGHT_BINS:
                np.add.at(cnt_night, (ci, reg), 1)
        print(f"{ym} [{'train' if is_train else 'holdout'}]: seqs={lab.shape[0]:,}", flush=True)

    # ---- per-cell summary (train) ----
    func_arr = np.array(func)
    func_cnt = cnt_all[:, func_arr]                 # (n, n_func)
    func_tot = func_cnt.sum(1, keepdims=True)
    func_share = np.where(func_tot > 0, func_cnt / np.where(func_tot > 0, func_tot, 1), 0)
    ent = -(np.where(func_share > 0, func_share * np.log(func_share + 1e-12), 0)).sum(1)
    ent_norm = ent / np.log(len(func))              # 0..1
    struct_ratio = cnt_all[:, struct].sum(1) / np.where(cnt_all.sum(1) > 0, cnt_all.sum(1), 1)

    def dom(cnt):
        d = np.full(cnt.shape[0], -1)
        m = cnt.sum(1) > 0
        d[m] = cnt[m].argmax(1)
        return d

    dom_all = dom(cnt_all); dom_day = dom(cnt_day); dom_night = dom(cnt_night)
    dom_func = np.full(n, -1)
    mfunc = func_cnt.sum(1) > 0
    dom_func[mfunc] = func_arr[func_cnt[mfunc].argmax(1)]

    out = uni.select(["cell_id", "utm_x", "utm_y", "admin_dong"]).with_columns(
        pl.Series("n_celldays_train", cnt_all.sum(1).astype(int)),
        pl.Series("dominant_all", [REGIME_NAME.get(int(x), "none") for x in dom_all]),
        pl.Series("dominant_day", [REGIME_NAME.get(int(x), "none") for x in dom_day]),
        pl.Series("dominant_night", [REGIME_NAME.get(int(x), "none") for x in dom_night]),
        pl.Series("dominant_functional", [REGIME_NAME.get(int(x), "none") for x in dom_func]),
        pl.Series("func_entropy", ent_norm),
        pl.Series("struct_bin_ratio", struct_ratio),
        pl.Series("dominant_all_ho", [REGIME_NAME.get(int(x), "none") for x in dom(cnt_all_ho)]),
    )
    for j, k in enumerate(func):
        out = out.with_columns(pl.Series(f"share_{REGIME_NAME[k]}", func_share[:, j]))
    out.write_parquet(C.PROCESSED_DIR / "cell_regime_summary.parquet")
    print(f"wrote cell_regime_summary.parquet ({out.height} cells)", flush=True)

    # ---- diurnal ----
    diur = {"time_bins": TB_ORDER, "func_states": func, "struct_states": struct,
            "regime_names": REGIME_NAME,
            "P_regime_given_tb_train": (tb_reg / tb_reg.sum(1, keepdims=True)).tolist(),
            "P_regime_given_tb_holdout": (tb_reg_ho / np.where(tb_reg_ho.sum(1, keepdims=True) > 0,
                                          tb_reg_ho.sum(1, keepdims=True), 1)).tolist()}
    (C.STATS_DIR / "regime_diurnal.json").write_text(json.dumps(diur, ensure_ascii=False), encoding="utf-8")

    # ---- admin discordance ----
    sm = out.with_columns(pl.col("admin_dong").str.slice(0, 5).alias("gu_code"))
    rows = []
    for r in sm.group_by("admin_dong").agg(
        pl.len().alias("n_cells"),
        pl.col("dominant_functional").n_unique().alias("n_distinct_dom"),
        pl.col("dominant_functional").mode().first().alias("modal_dom"),
        pl.col("func_entropy").mean().alias("mean_entropy"),
    ).iter_rows(named=True):
        rows.append(r)
    admin_df = pl.DataFrame(rows)
    # share of admin-dongs hosting >=2 distinct dominant functional regimes
    multi = admin_df.filter(pl.col("n_distinct_dom") >= 2).height
    # purity: within-dong fraction of cells in the modal dominant regime
    pur = (out.group_by(["admin_dong", "dominant_functional"]).len()
           .group_by("admin_dong").agg(pl.col("len").max().alias("mx"), pl.col("len").sum().alias("tot")))
    pur = pur.with_columns((pl.col("mx") / pl.col("tot")).alias("purity"))
    disc = {
        "n_admin_dong": admin_df.height,
        "n_dong_multi_regime": int(multi),
        "pct_dong_multi_regime": round(100 * multi / admin_df.height, 1),
        "mean_distinct_dom_per_dong": round(float(admin_df["n_distinct_dom"].mean()), 2),
        "mean_within_dong_purity": round(float(pur["purity"].mean()), 3),
        "median_within_dong_purity": round(float(pur["purity"].median()), 3),
    }
    admin_df.write_parquet(C.STATS_DIR / "regime_admin_table.parquet")
    (C.STATS_DIR / "regime_admin_discordance.json").write_text(
        json.dumps(disc, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(disc, indent=2, ensure_ascii=False))
    print("DONE")


if __name__ == "__main__":
    main()
