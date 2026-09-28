"""Step 17 - behavioral regime vs the Seoul living-zone plan (living-zone boundaries,
ZON100, + designated-center system, ZON500).

Plan CRS determined empirically = EPSG:5174 (Bessel central-belt): of 49 station<->
same-named center matches, 5174 puts 30 inside (median 0 m) vs 5181's 9 (median 136 m);
IoU(plan,grid) 0.957 vs 0.943. dbf read as CP949 (.cpg mislabels UTF-8). -> EPSG:5179.

(A) living-zone boundary discordance vs admin-dong (distinct dominant / purity /
    between-unit variance).
(B) designated-center concordance: activity-dominance per tier + bidirectional mismatch
    (planned-not-active, active-not-planned). A center is activity-dominant when
    > 50% of its cells are activity-core cells (dominant_functional in activity_A/B).
    The earlier rule (mean activity_share > Seoul mean) is kept as `above_city_mean`.
Headline = TRAIN regimes (cell_regime_summary).

Tier / unit names are kept in Korean as in the source data:
  living zones  권역생활권 (wide-area, ZON121), 지역생활권 (district, ZON125)
  centers       도심 (downtown, ZON510), 광역중심 (regional, ZON520),
                지역중심 (district, ZON530), 지구중심 (local, ZON540)

Outputs: data/processed/cell_plan_map.parquet, results/stats/plan_discordance_stats.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

PLAN_CRS = 5174
PLAN_DIR = C.PROCESSED_DIR / "plan"   # extracted by src/00_extract_spatial.py
ACT = ["activity_A", "activity_B"]
SAENGGWON = {"ZON121": "권역생활권", "ZON125": "지역생활권"}
JUNGSIM = {"ZON510": "도심", "ZON520": "광역중심", "ZON530": "지역중심", "ZON540": "지구중심"}
TIER_ORDER = ["도심", "광역중심", "지역중심", "지구중심"]
ACT_DOM_FRAC = 0.5   # center activity-dominant if frac of activity-core cells > this


def load_plan(name):
    import geopandas as gpd
    g = gpd.read_file(PLAN_DIR / name / f"UPIS_SHP_{name}.shp", encoding="cp949")
    return g.set_crs(PLAN_CRS, allow_override=True).to_crs(5179)


def cells_gdf(reg):
    import geopandas as gpd
    from shapely.geometry import Point
    pts = [Point(x, y) for x, y in zip(reg["utm_x"], reg["utm_y"])]
    pdf = reg.to_pandas()
    return gpd.GeoDataFrame(pdf, geometry=pts, crs=5179)


def discordance(df, group):
    d = df.filter((pl.col(group).is_not_null()) & (pl.col("dominant_functional") != "none"))
    per = d.group_by(group).agg(
        pl.col("dominant_functional").n_unique().alias("nd"),
        pl.len().alias("tot"),
        pl.col("dominant_functional").value_counts().struct.field("count").max().alias("mx"))
    return {"n_units": per.height,
            "mean_distinct_dom": round(float(per["nd"].mean()), 3),
            "mean_purity": round(float((per["mx"] / per["tot"]).mean()), 3),
            "median_purity": round(float((per["mx"] / per["tot"]).median()), 3),
            "pct_multi": round(100 * per.filter(pl.col("nd") >= 2).height / per.height, 1)}


def between_var(df, group, cols):
    d = df.filter(pl.col(group).is_not_null())
    tot_b = tot_t = 0.0
    gm = {r[group]: r for r in d.group_by(group).agg(
        [pl.col(c).mean().alias(c) for c in cols] + [pl.len().alias("ng")]).iter_rows(named=True)}
    for c in cols:
        x = d[c].to_numpy(); xbar = x.mean(); sst = np.sum((x - xbar) ** 2)
        ssb = sum(gm[k]["ng"] * (gm[k][c] - xbar) ** 2 for k in gm)
        tot_b += ssb; tot_t += sst
    return round(float(tot_b / tot_t), 4)


def main():
    import geopandas as gpd
    reg = pl.read_parquet(C.PROCESSED_DIR / "cell_regime_summary.parquet").with_columns(
        (pl.col("share_activity_A") + pl.col("share_activity_B")).alias("activity_share"))
    cg = cells_gdf(reg)

    sg = load_plan("ZON100")     # living zones
    js = load_plan("ZON500")     # designated centers
    sg["sg_tier"] = sg["CODE"].map(SAENGGWON)
    js["js_tier"] = js["CODE"].map(JUNGSIM)

    # spatial join cells -> plan units (point in polygon)
    j_local = gpd.sjoin(cg, sg[sg["sg_tier"] == "지역생활권"][["LABEL", "geometry"]],
                        predicate="within", how="left").rename(columns={"LABEL": "sg_local"})
    j_local = j_local[~j_local.index.duplicated(keep="first")]
    j_reg = gpd.sjoin(cg, sg[sg["sg_tier"] == "권역생활권"][["LABEL", "geometry"]],
                      predicate="within", how="left").rename(columns={"LABEL": "sg_region"})
    j_reg = j_reg[~j_reg.index.duplicated(keep="first")]
    jc = gpd.sjoin(cg, js[["LABEL", "js_tier", "geometry"]], predicate="within", how="left")
    jc = jc[~jc.index.duplicated(keep="first")]

    import pandas as pd
    def tol(s):
        return [None if pd.isna(v) else str(v) for v in s.tolist()]
    cellmap = reg.select(["cell_id", "admin_dong", "dominant_functional",
                          "activity_share", "share_low_density", "share_activity_A",
                          "share_dense_mixed", "share_activity_B", "share_residential",
                          "share_mid_density"]).with_columns(
        pl.Series("sg_local", tol(j_local["sg_local"]), dtype=pl.Utf8),
        pl.Series("sg_region", tol(j_reg["sg_region"]), dtype=pl.Utf8),
        pl.Series("jungsim", tol(jc["LABEL"]), dtype=pl.Utf8),
        pl.Series("jungsim_tier", tol(jc["js_tier"]), dtype=pl.Utf8))
    cellmap.write_parquet(C.PROCESSED_DIR / "cell_plan_map.parquet")

    # ---- (A) living zones vs admin-dong discordance ----
    comp_cols = [f"share_{r}" for r in
                 ["low_density", "activity_A", "dense_mixed", "activity_B",
                  "residential", "mid_density"]]
    rows = []
    for label, grp in [("행정동(admin)", "admin_dong"),
                       ("지역생활권", "sg_local"), ("권역생활권", "sg_region")]:
        d = discordance(cellmap, grp)
        d["between_var"] = between_var(cellmap, grp, comp_cols)
        d["unit"] = label
        rows.append(d)
    stats = {"plan_crs": PLAN_CRS, "saenggwon_discordance": rows}

    # ---- (B) designated-center concordance ----
    seoul_mean_act = float(reg["activity_share"].mean())
    jdf = cellmap.filter(pl.col("jungsim").is_not_null())
    per_js = jdf.group_by(["jungsim", "jungsim_tier"]).agg(
        pl.col("activity_share").mean().alias("mean_act"),
        pl.len().alias("n_cells"),
        (pl.col("dominant_functional").is_in(ACT).mean()).alias("frac_act_dom"))
    per_js = per_js.with_columns(
        (pl.col("frac_act_dom") > ACT_DOM_FRAC).alias("is_active"),        # activity-dominant
        (pl.col("mean_act") > seoul_mean_act).alias("above_city_mean"))    # old rule (auxiliary)
    tier_summary = []
    for t in TIER_ORDER:
        sub = per_js.filter(pl.col("jungsim_tier") == t)
        if sub.height == 0:
            continue
        tier_summary.append({
            "tier": t, "n_centers": sub.height,
            "n_active": int(sub["is_active"].sum()),
            "pct_active": round(100 * sub["is_active"].sum() / sub.height, 1),
            "mean_activity_share": round(float(sub["mean_act"].mean()), 3),
            "mean_frac_act_dom": round(float(sub["frac_act_dom"].mean()), 3),
            "n_above_city_mean": int(sub["above_city_mean"].sum()),
            "pct_above_city_mean": round(100 * sub["above_city_mean"].sum() / sub.height, 1)})
    planned_not_active = per_js.filter(~pl.col("is_active")).sort(
        ["frac_act_dom", "mean_act"]).select(
        ["jungsim", "jungsim_tier", "n_cells", "mean_act", "frac_act_dom", "above_city_mean"])
    stats["activity_dominant_rule"] = (
        f"center activity-dominant if fraction of its cells (centroid within polygon) with "
        f"dominant_functional in {ACT} > {ACT_DOM_FRAC}")
    stats["seoul_mean_activity_share"] = round(seoul_mean_act, 4)
    stats["jungsim_concordance_by_tier"] = tier_summary
    stats["planned_not_active"] = planned_not_active.to_dicts()
    tier_rank = {t: i for i, t in enumerate(TIER_ORDER)}
    stats["centers"] = sorted(per_js.select(
        ["jungsim", "jungsim_tier", "n_cells", "mean_act", "frac_act_dom", "is_active",
         "above_city_mean"]).to_dicts(),
        key=lambda r: (tier_rank[r["jungsim_tier"]], r["jungsim"]))

    # active-not-planned: activity-dominant cells outside any designated center
    act_cells = cellmap.filter(pl.col("dominant_functional").is_in(ACT))
    act_out = act_cells.filter(pl.col("jungsim").is_null())
    stats["active_not_planned"] = {
        "n_activity_dom_cells": act_cells.height,
        "n_outside_any_center": act_out.height,
        "pct_outside": round(100 * act_out.height / act_cells.height, 1),
        "top_admin_dong_outside": act_out.group_by("admin_dong").len()
            .sort("len", descending=True).head(12).to_dicts()}
    (C.STATS_DIR / "plan_discordance_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in stats.items()
                      if k in ("saenggwon_discordance", "jungsim_concordance_by_tier",
                               "active_not_planned")}, ensure_ascii=False, indent=2)[:1500])
    print("DONE")


if __name__ == "__main__":
    main()
