"""Step 16 - external validation: transit ridership commute-asymmetry vs HSMM regime.

Independent AFC (transit-card) sensor vs telecom-derived regimes.
- Residential commute-asymmetry score per station/stop (operating window):
    res_score = ((AM_board+PM_alight) - (AM_alight+PM_board)) / total   in [-1,1]
  AM=7-9h, PM=18-20h. >0 = residential (leave home AM / return PM); <0 = activity.
- Catchment = station/stop cell + k-ring (k=0,1,2) universe cells; regime
  residential/activity share = mean over catchment cells (cell_regime_summary).
- Alignment: regress res_score ~ catchment residential share -> R^2, slope, p.
- Typology: classify stations (residential/activity/none) x catchment dominant
  regime group -> chi^2 + Cramer's V.
Headline = TRAIN months (2025.01-2026.03). Bus reinforces coverage at cell level.

Outputs: data/processed/transit_validation.parquet, results/stats/validation_stats.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

TRAIN = [m for m in C.MONTHS if m <= "202603"]
AM = [7, 8, 9]
PM = [18, 19, 20]


def res_score(df, idcols):
    """df: long (id..,hour,board,alight) summed over train. -> res_score per id."""
    g = df.group_by(idcols).agg(
        pl.col("board").filter(pl.col("hour").is_in(AM)).sum().alias("amb"),
        pl.col("alight").filter(pl.col("hour").is_in(AM)).sum().alias("ama"),
        pl.col("board").filter(pl.col("hour").is_in(PM)).sum().alias("pmb"),
        pl.col("alight").filter(pl.col("hour").is_in(PM)).sum().alias("pma"),
    )
    tot = (pl.col("amb") + pl.col("ama") + pl.col("pmb") + pl.col("pma"))
    return g.with_columns(
        pl.when(tot > 0)
        .then(((pl.col("amb") + pl.col("pma")) - (pl.col("ama") + pl.col("pmb"))) / tot)
        .otherwise(None).alias("res_score"),
        tot.alias("flow")).filter(pl.col("res_score").is_not_null())


def regime_lookup(reg):
    ix = np.round((reg["utm_x"].to_numpy() - 936375.0) / 250).astype(int)
    iy = np.round((reg["utm_y"].to_numpy() - 1937125.0) / 250).astype(int)
    resid = reg["share_residential"].to_numpy()
    act = reg["activity_share"].to_numpy() if "activity_share" in reg.columns else \
        (reg["share_activity_A"].to_numpy() + reg["share_activity_B"].to_numpy())
    dom = reg["dominant_functional"].to_list()
    L = {}
    for a, b, r, c, d in zip(ix, iy, resid, act, dom):
        L[(int(a), int(b))] = (r, c, d)
    return L


def catchment(gix, giy, L, k):
    """mean residential & activity share over k-ring universe cells; modal dom."""
    res, act, doms = [], [], []
    for a, b in zip(gix, giy):
        rr, aa, dd = [], [], []
        for da in range(-k, k + 1):
            for db in range(-k, k + 1):
                v = L.get((int(a) + da, int(b) + db))
                if v:
                    rr.append(v[0]); aa.append(v[1]); dd.append(v[2])
        res.append(np.mean(rr) if rr else np.nan)
        act.append(np.mean(aa) if aa else np.nan)
        doms.append(max(sorted(set(dd)), key=dd.count) if dd else None)   # ties -> first by name
    return np.array(res), np.array(act), doms


def cramers_v(ct):
    chi2, p, _, _ = stats.chi2_contingency(ct)
    n = ct.sum()
    return chi2, p, np.sqrt(chi2 / (n * (min(ct.shape) - 1)))


def regress(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    lr = stats.linregress(x[m], y[m])
    return {"n": int(m.sum()), "r": round(lr.rvalue, 3), "R2": round(lr.rvalue**2, 3),
            "slope": round(lr.slope, 3), "p": float(lr.pvalue),
            "pearson_res_vs_resid": round(lr.rvalue, 3)}


def main():
    reg = pl.read_parquet(C.PROCESSED_DIR / "cell_regime_summary.parquet").with_columns(
        (pl.col("share_activity_A") + pl.col("share_activity_B")).alias("activity_share"))
    L = regime_lookup(reg)
    stats_out = {}

    # ---------- SUBWAY ----------
    smap = pl.read_parquet(C.PROCESSED_DIR / "station_cell_map.parquet").filter(
        pl.col("cell_id").is_not_null())
    srid = pl.scan_parquet(str(C.PROCESSED_DIR / "subway_ridership" / "year_month=*" / "*.parquet")) \
        .filter(pl.col("use_mm").is_in(TRAIN)).collect()
    ssc = res_score(srid, ["station_name"])
    sj = ssc.join(smap.select(["station_name", "cell_id", "grid_ix", "grid_iy"]),
                  on="station_name", how="inner")
    gix = sj["grid_ix"].to_numpy(); giy = sj["grid_iy"].to_numpy()
    sub_res = {}
    for k in [0, 1, 2]:
        cr, ca, cd = catchment(gix, giy, L, k)
        sj = sj.with_columns(pl.Series(f"cat_resid_k{k}", cr),
                             pl.Series(f"cat_act_k{k}", ca))
        if k == 1:
            sj = sj.with_columns(pl.Series("cat_dom_k1", cd, dtype=pl.Utf8))
        rr = regress(sj[f"cat_resid_k{k}"].to_numpy(), sj["res_score"].to_numpy())
        ra = regress(sj[f"cat_act_k{k}"].to_numpy(), sj["res_score"].to_numpy())
        sub_res[f"k{k}"] = {"vs_residential_share": rr, "vs_activity_share": ra}
    stats_out["subway_alignment"] = sub_res
    stats_out["subway_n_stations"] = sj.height

    # typology (subway, k=1)
    sj = sj.with_columns(
        pl.when(pl.col("res_score") > 0.1).then(pl.lit("resid_commute"))
        .when(pl.col("res_score") < -0.1).then(pl.lit("activity_commute"))
        .otherwise(pl.lit("no_pattern")).alias("station_type"),
        pl.when(pl.col("cat_dom_k1") == "residential").then(pl.lit("residential"))
        .when(pl.col("cat_dom_k1").is_in(["activity_A", "activity_B"])).then(pl.lit("activity"))
        .otherwise(pl.lit("other")).alias("cell_group"))
    ct = sj.group_by(["station_type", "cell_group"]).len().pivot(
        values="len", index="station_type", on="cell_group").fill_null(0)
    cats_col = [c for c in ct.columns if c != "station_type"]
    M = ct.select(cats_col).to_numpy()
    chi2, p, v = cramers_v(M)
    stats_out["subway_typology"] = {"crosstab_rows": ct["station_type"].to_list(),
                                    "crosstab_cols": cats_col, "crosstab": M.tolist(),
                                    "chi2": round(chi2, 1), "p": float(p),
                                    "cramers_v": round(v, 3)}
    sj.write_parquet(C.PROCESSED_DIR / "transit_validation.parquet")

    # ---------- BUS (cell-level reinforcement) ----------
    bmap = pl.read_parquet(C.PROCESSED_DIR / "stop_cell_map.parquet").filter(
        pl.col("cell_id").is_not_null())
    brid = pl.scan_parquet(str(C.PROCESSED_DIR / "bus_ridership_bystop" / "year_month=*" / "*.parquet")) \
        .filter(pl.col("use_ym").is_in(TRAIN)).collect()
    bsc = res_score(brid, ["stop_id"]).join(
        bmap.select(["stop_id", "cell_id", "grid_ix", "grid_iy"]), on="stop_id", how="inner")
    # weighted mean res_score per cell (weight by flow)
    bcell = bsc.group_by(["cell_id", "grid_ix", "grid_iy"]).agg(
        ((pl.col("res_score") * pl.col("flow")).sum() / pl.col("flow").sum()).alias("res_score"),
        pl.col("flow").sum().alias("flow"), pl.len().alias("n_stops"))
    bgi = bcell["grid_ix"].to_numpy(); bgy = bcell["grid_iy"].to_numpy()
    cr, ca, _ = catchment(bgi, bgy, L, 0)
    bcell = bcell.with_columns(pl.Series("cat_resid_k0", cr), pl.Series("cat_act_k0", ca))
    br = regress(bcell["cat_resid_k0"].to_numpy(), bcell["res_score"].to_numpy())
    ba = regress(bcell["cat_act_k0"].to_numpy(), bcell["res_score"].to_numpy())
    stats_out["bus_alignment_cell"] = {"vs_residential_share": br,
                                       "vs_activity_share": ba, "n_cells": bcell.height}

    (C.STATS_DIR / "validation_stats.json").write_text(
        json.dumps(stats_out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("DONE")


if __name__ == "__main__":
    main()
