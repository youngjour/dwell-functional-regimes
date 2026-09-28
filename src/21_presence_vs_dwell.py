"""Step 21 - presence-only vs dwell-aware comparison + external-validation delta.

Reads po_compare_labels.npz, cell_regime_presence_only.parquet, cell_regime_summary,
transit_validation, cell_plan/place maps. Computes label agreement (ARI/NMI),
dwell-split characterisation, day/night switch, and the DECISIVE external-validation
R^2 delta (transit commute-asymmetry ~ regime composition / residential axis).

Outputs: results/stats/presence_vs_dwell_stats.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

LADDER = C.MODELS_DIR / "ladder_1455c0d960"
INPUT_V2 = C.MODELS_DIR / "input_v2_1455c0d960"
FUNC_REG = ["low_density", "activity_A", "dense_mixed", "activity_B", "residential",
            "mid_density", "structural_missing"]
SHARE_DW = [f"share_{r}" for r in ["low_density", "activity_A", "dense_mixed",
            "activity_B", "residential", "mid_density"]]


def hungarian_crosstab(a, b, ka, kb):
    M = np.zeros((ka, kb), int)
    np.add.at(M, (a, b), 1)
    from scipy.optimize import linear_sum_assignment
    ri, ci = linear_sum_assignment(-M)
    return M, dict(zip(ri.tolist(), ci.tolist()))


def lin(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    lr = stats.linregress(x[m], y[m])
    return {"r": round(lr.rvalue, 3), "R2": round(lr.rvalue**2, 3), "p": float(lr.pvalue),
            "n": int(m.sum())}


def multi_R2(X, y):
    m = np.isfinite(y) & np.isfinite(X).all(1)
    X1 = np.column_stack([np.ones(m.sum()), X[m]])
    beta, *_ = np.linalg.lstsq(X1, y[m], rcond=None)
    yhat = X1 @ beta
    ss_res = ((y[m] - yhat) ** 2).sum(); ss_tot = ((y[m] - y[m].mean()) ** 2).sum()
    R2 = 1 - ss_res / ss_tot
    k = X.shape[1]; nn = m.sum()
    adj = 1 - (1 - R2) * (nn - 1) / (nn - k - 1)
    return {"R2": round(float(R2), 3), "adj_R2": round(float(adj), 3), "n": int(nn), "k": k}


def catchment_mean(gix, giy, lookup, k):
    out = []
    for a, b in zip(gix, giy):
        vals = [lookup[(int(a)+da, int(b)+db)] for da in range(-k, k+1)
                for db in range(-k, k+1) if (int(a)+da, int(b)+db) in lookup]
        out.append(np.mean(vals, axis=0) if vals else np.full(next(iter(lookup.values())).shape, np.nan))
    return np.array(out)


def main():
    stats_out = {}
    # ---- label agreement ----
    lab = np.load(LADDER.parent / "po_compare_labels.npz")
    po = lab["po"].ravel(); dw = lab["dwell"].ravel()
    stats_out["label_agreement_cellbin"] = {
        "ARI": round(adjusted_rand_score(dw, po), 3),
        "NMI": round(normalized_mutual_info_score(dw, po), 3), "n": int(len(po))}
    M, mapping = hungarian_crosstab(po, dw, po.max()+1, dw.max()+1)
    stats_out["crosstab_po_x_dwell"] = M.tolist()

    # ---- per-cell dominant agreement ----
    dwc = pl.read_parquet(C.PROCESSED_DIR / "cell_regime_summary.parquet")
    poc = pl.read_parquet(C.PROCESSED_DIR / "cell_regime_presence_only.parquet")
    j = dwc.select(["cell_id", "dominant_functional", "dominant_day", "dominant_night",
                    *SHARE_DW, "struct_bin_ratio"]).join(poc, on="cell_id")
    # encode dwell dominant to int
    reg2i = {r: i for i, r in enumerate(FUNC_REG)}
    dwd = np.array([reg2i.get(x, 6) for x in j["dominant_functional"].to_list()])
    pod = j["po_dominant"].to_numpy()
    stats_out["per_cell_dominant_agreement"] = {
        "ARI": round(adjusted_rand_score(dwd, pod), 3),
        "NMI": round(normalized_mutual_info_score(dwd, pod), 3), "n": j.height}

    # ---- day/night switch ----
    dw_switch = (dwc.filter(pl.col("dominant_day") != pl.col("dominant_night")).height
                 / dwc.height)
    po_switch = (poc.filter(pl.col("po_dominant_day") != pl.col("po_dominant_night")).height
                 / poc.height)
    stats_out["daynight_switch"] = {"dwell_aware_pct": round(100*dw_switch, 1),
                                    "presence_only_pct": round(100*po_switch, 1)}

    # ---- dwell-split: PO state that maps to multiple dwell regimes, by long_share ----
    # for cells, group by po_dominant; within each, distribution of dwell dominant + mean long_share
    dwc2 = dwc.with_columns((pl.col("share_activity_A")+pl.col("share_activity_B")).alias("act"))
    jj = j.join(dwc2.select(["cell_id"]), on="cell_id")
    # use long_share proxy: residential share within po-state groups
    split_rows = []
    for s in range(int(pod.max())+1):
        sub = j.filter(pl.col("po_dominant") == s)
        if sub.height < 20:
            continue
        ddist = sub.group_by("dominant_functional").len().sort("len", descending=True)
        n_dw = ddist.height
        split_rows.append({"po_state": s, "n_cells": sub.height, "n_dwell_regimes": n_dw,
                           "top_dwell": ddist["dominant_functional"][0],
                           "mean_share_residential": round(float(sub["share_residential"].mean()), 3),
                           "mean_share_activityB": round(float(sub["share_activity_B"].mean()), 3)})
    stats_out["dwell_split_by_po_state"] = split_rows

    # ---- external validation delta ----
    tv = pl.read_parquet(C.PROCESSED_DIR / "transit_validation.parquet")
    rob = pl.read_parquet(C.PROCESSED_DIR / "regime_robustness.parquet").select(
        ["cell_id", "grid_ix", "grid_iy"])
    dwc3 = dwc.join(rob, on="cell_id").with_columns(
        (pl.col("share_activity_A")+pl.col("share_activity_B")).alias("act"))
    poc3 = poc.join(rob, on="cell_id")
    # lookups: dwell residential & activity share; PO res & act share; full comps
    dwL_res = {(a, b): np.array([r]) for a, b, r in zip(dwc3["grid_ix"], dwc3["grid_iy"], dwc3["share_residential"])}
    dwL_act = {(a, b): np.array([r]) for a, b, r in zip(dwc3["grid_ix"], dwc3["grid_iy"], dwc3["act"])}
    poL_res = {(a, b): np.array([r]) for a, b, r in zip(poc3["grid_ix"], poc3["grid_iy"], poc3["po_res_share"])}
    poL_act = {(a, b): np.array([r]) for a, b, r in zip(poc3["grid_ix"], poc3["grid_iy"], poc3["po_act_share"])}
    poshare = [f"po_share_s{k}" for k in range(7)]
    dwL_full = {(a, b): np.array(v) for a, b, v in
                zip(dwc3["grid_ix"], dwc3["grid_iy"],
                    dwc3.select(SHARE_DW+["struct_bin_ratio"]).to_numpy())}
    poL_full = {(a, b): np.array(v) for a, b, v in
                zip(poc3["grid_ix"], poc3["grid_iy"], poc3.select(poshare).to_numpy())}
    gix = tv["grid_ix"].to_numpy(); giy = tv["grid_iy"].to_numpy(); rs = tv["res_score"].to_numpy()
    ext = {}
    for k in [1, 2]:
        dwr = catchment_mean(gix, giy, dwL_res, k)[:, 0]
        dwa = catchment_mean(gix, giy, dwL_act, k)[:, 0]
        por = catchment_mean(gix, giy, poL_res, k)[:, 0]
        poa = catchment_mean(gix, giy, poL_act, k)[:, 0]
        dwf = catchment_mean(gix, giy, dwL_full, k)
        pof = catchment_mean(gix, giy, poL_full, k)
        ext[f"k{k}"] = {
            "dwell_res_axis": lin(dwr, rs), "po_res_axis": lin(por, rs),
            "dwell_act_axis": lin(dwa, rs), "po_act_axis": lin(poa, rs),
            "dwell_full_comp": multi_R2(dwf, rs), "po_full_comp": multi_R2(pof, rs)}
    stats_out["external_validation_delta"] = ext

    (C.STATS_DIR / "presence_vs_dwell_stats.json").write_text(
        json.dumps(stats_out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("DONE")


if __name__ == "__main__":
    main()
