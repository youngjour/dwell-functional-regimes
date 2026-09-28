"""Step 20 - presence-only HSMM (Beyond Presence). Same model/K/cells/seq/seed as the
dwell-aware HSMM; ONLY the feature block changes (drop dwell).

presence-only features (18d): [presence_log_z, clr_mix_z(16), mix_missing].
(dwell-aware was 26d: + stay_vol_log_z, clr_dwell(6), dwell_missing.)

- Fit on the SAME input_v2 50k train sequences (rebuilt from train_meta: presence_log
  + raw mix), 5 reinit, HSMM K=7 -> save model.
- Predict on those 50k seqs -> labels_PO; load dwell-aware labels from the ladder npz
  (same sequences, in-sample) for a fair label-agreement comparison.
- Decode the WHOLE universe (full-19 cell-days, train headline) -> per-cell PO regime
  summary + diurnal + per-cell 7-state share vector.

Outputs: results/models/hsmm_presence_only.pkl, data/processed/cell_regime_presence_only.parquet,
results/stats/presence_only_{meta,diurnal}.json, results/models/po_compare_labels.npz.

Run (bg): python src/20_presence_only.py
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

SEED = C.SEED
K = 7
DELTA = 1e-3
TB_ORDER = ["00"] + [f"{h:02d}" for h in range(6, 24)]
TB_RANK = {b: i for i, b in enumerate(TB_ORDER)}
T = len(TB_ORDER)
TRAIN_MAX = "202603"
DAY = [TB_RANK[b] for b in ["10", "11", "12", "13", "14", "15", "16"]]
NIGHT = [TB_RANK[b] for b in ["00", "22", "23"]]
MIX = [f"mix_{s}{b}" for s in ["M", "F"]
       for b in ["00", "10", "20", "30", "40", "50", "60", "70"]]
INPUT_V2 = C.MODELS_DIR / "input_v2_1455c0d960"
LADDER = C.MODELS_DIR / "ladder_1455c0d960"


def mult_replace_clr(P, delta):
    P = P.astype(np.float64).copy()
    Z = (P == 0).sum(1, keepdims=True)
    P = np.where(P == 0, delta, P * (1.0 - Z * delta))
    P = P / P.sum(1, keepdims=True)
    Lg = np.log(P)
    return Lg - Lg.mean(1, keepdims=True)


def build_po(presence_log, mix, mix_missing, meta):
    pm, ps = meta["scaler_presence"]
    clr = np.zeros((len(presence_log), len(MIX)))
    clr[~mix_missing] = mult_replace_clr(mix[~mix_missing], DELTA)
    clr[mix_missing] = np.array(meta["clr_mix_train_mean"])
    mzc, mzs = np.array(meta["clr_mix_z"][0]), np.array(meta["clr_mix_z"][1])
    clr_z = (clr - mzc) / np.where(mzs > 0, mzs, 1)
    pz = (presence_log - pm) / ps
    return np.column_stack([pz, clr_z, mix_missing.astype(float)]).astype(np.float32)


def main():
    scaler = json.loads((C.PROCESSED_DIR / "scaler.json").read_text("utf-8"))
    pm, ps = scaler["features"]["presence_log"]["mean"], scaler["features"]["presence_log"]["std"]

    # ---- training sample = input_v2 train_meta (same 50k seqs, ordered) ----
    tm = pl.read_parquet(INPUT_V2 / "train_meta.parquet")
    plog = tm["presence_log"].to_numpy().astype(float)
    mix = tm.select(MIX).to_numpy()
    mix_missing = np.isnan(mix).any(1)
    # transform params (train-only)
    clr = np.zeros((len(plog), len(MIX)))
    clr[~mix_missing] = mult_replace_clr(mix[~mix_missing], DELTA)
    mix_mean = clr[~mix_missing].mean(0)
    clr[mix_missing] = mix_mean
    mzc, mzs = clr.mean(0), clr.std(0)
    meta = {"scaler_presence": [pm, ps], "clr_mix_train_mean": mix_mean.tolist(),
            "clr_mix_z": [mzc.tolist(), mzs.tolist()], "features_dim": 18, "seed": SEED}
    (C.STATS_DIR / "presence_only_meta.json").write_text(json.dumps(meta), encoding="utf-8")

    Xtr = build_po(plog, mix, mix_missing, meta)
    n = Xtr.shape[0] // T
    Xtr3 = Xtr.reshape(n, T, Xtr.shape[1])
    print(f"PO train {Xtr3.shape}", flush=True)

    # ---- fit HSMM K=7, 5 reinit ----
    fits = []
    for r in range(5):
        m = edh.EDHSMM(K, n_iter=25, random_state=SEED + 31 * r).fit(Xtr3)
        fits.append((m.train_ll_, m))
        print(f"  PO reinit{r} ll={m.train_ll_:.0f}", flush=True)
    fits.sort(key=lambda t: t[0], reverse=True)
    model = fits[0][1]
    pickle.dump({"means_": model.means_, "vars_": model.vars_,
                 "transmat_": model.transmat_, "startprob_": model.startprob_,
                 "dur_pmf_": model.dur_pmf_, "dur_mean_": model.dur_mean_},
                open(C.MODELS_DIR / "hsmm_presence_only.pkl", "wb"))

    # ---- label agreement on the SAME 50k seqs ----
    lab_po = model.predict(Xtr3)                                   # (n,19)
    lab_dw = np.load(LADDER / "viterbi_labels.npz")["HSMM"]        # (50000,19) dwell-aware
    cell_ids = tm.unique(subset=["cell_id", "date"], maintain_order=True)["cell_id"].to_list()
    np.savez_compressed(C.MODELS_DIR / "po_compare_labels.npz",
                        po=lab_po, dwell=lab_dw)
    print(f"compare labels: PO {lab_po.shape} dwell {lab_dw.shape}", flush=True)

    # ---- full-universe decode (train headline) ----
    cm = pl.read_parquet(C.PROCESSED_DIR / "cell_master.parquet").filter(pl.col("in_universe"))
    cells = cm["cell_id"].to_list(); cidx = {c: i for i, c in enumerate(cells)}
    nc = len(cells)
    cnt_all = np.zeros((nc, K)); cnt_day = np.zeros((nc, K)); cnt_night = np.zeros((nc, K))
    tb_reg = np.zeros((T, K))
    uni = set(cells)
    cols = ["cell_id", "date", "time_bin", "split", "presence_log",
            "presence_demo_masked"] + MIX
    for ym in C.MONTHS:
        if ym > TRAIN_MAX:
            continue
        em = pl.scan_parquet(str(C.PROCESSED_DIR / "emission" / f"year_month={ym}" / "*.parquet"))
        em = em.filter(pl.col("cell_id").is_in(list(uni)))
        full = (em.group_by(["cell_id", "date"]).agg(pl.len().alias("nb"))
                .filter(pl.col("nb") == T))
        df = (em.join(full.select(["cell_id", "date"]), on=["cell_id", "date"], how="inner")
              .select(cols).with_columns(pl.col("time_bin").replace_strict(TB_RANK).alias("tbr"))
              .sort(["cell_id", "date", "tbr"]).collect(engine="streaming"))
        if df.height == 0:
            continue
        mm = df["presence_demo_masked"].to_numpy() | np.isnan(df.select(MIX).to_numpy()).any(1)
        X = build_po(df["presence_log"].to_numpy().astype(float), df.select(MIX).to_numpy(),
                     mm, meta).reshape(-1, T, 18)
        lab = model.predict(X)
        keys = df.select(["cell_id", "date"]).unique(maintain_order=True)
        ci = np.array([cidx[c] for c in keys["cell_id"].to_list()])
        for ti in range(T):
            np.add.at(tb_reg, (ti, lab[:, ti]), 1)
            np.add.at(cnt_all, (ci, lab[:, ti]), 1)
            if ti in DAY:
                np.add.at(cnt_day, (ci, lab[:, ti]), 1)
            if ti in NIGHT:
                np.add.at(cnt_night, (ci, lab[:, ti]), 1)
        print(f"  decoded {ym}: {lab.shape[0]} seqs", flush=True)

    # ---- PO state -> residential/activity via diurnal (night vs day activation) ----
    Pd = tb_reg / tb_reg.sum(1, keepdims=True)            # P(state|tb)
    night_act = Pd[NIGHT].mean(0); day_act = Pd[DAY].mean(0)
    residentialness = night_act - day_act                # >0 night-leaning (residential)
    res_states = [k for k in range(K) if residentialness[k] > 0.01]
    act_states = [k for k in range(K) if residentialness[k] < -0.01]
    meta["res_states"] = res_states; meta["act_states"] = act_states
    meta["state_residentialness"] = residentialness.tolist()

    share = cnt_all / np.where(cnt_all.sum(1, keepdims=True) > 0, cnt_all.sum(1, keepdims=True), 1)
    dom_all = cnt_all.argmax(1); dom_day = cnt_day.argmax(1); dom_night = cnt_night.argmax(1)
    po_res_share = share[:, res_states].sum(1) if res_states else np.zeros(nc)
    po_act_share = share[:, act_states].sum(1) if act_states else np.zeros(nc)
    out = cm.select(["cell_id", "utm_x", "utm_y", "admin_dong"]).with_columns(
        pl.Series("po_dominant", dom_all.astype(int)),
        pl.Series("po_dominant_day", dom_day.astype(int)),
        pl.Series("po_dominant_night", dom_night.astype(int)),
        pl.Series("po_res_share", po_res_share),
        pl.Series("po_act_share", po_act_share),
        *[pl.Series(f"po_share_s{k}", share[:, k]) for k in range(K)])
    out.write_parquet(C.PROCESSED_DIR / "cell_regime_presence_only.parquet")
    (C.STATS_DIR / "presence_only_diurnal.json").write_text(json.dumps({
        "P_state_given_tb": Pd.tolist(), "res_states": res_states, "act_states": act_states,
        "residentialness": residentialness.tolist(),
        "state_mean_presence": [float(np.expm1(model.means_[k, 0] * ps + pm)) for k in range(K)]
    }, ensure_ascii=False), encoding="utf-8")
    (C.STATS_DIR / "presence_only_meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    print(f"res_states={res_states} act_states={act_states}", flush=True)
    print("wrote cell_regime_presence_only.parquet\nDONE")


if __name__ == "__main__":
    main()
