r"""Step 22 - manuscript tables. No model/decode rerun; reads existing outputs only.

LaTeX bodies (booktabs \toprule/\midrule/\botrule; no caption/label):
  table2_centers.tex     Table 2   <- results/stats/plan_discordance_stats.json (step 17)
  tableS4_threshold.tex  Table S4  <- results/stats/conclusions_robustness_stats.json (step 18)
  tableS5_clusters.tex   Table S5  <- data/processed/emergent_activity_clusters.parquet
                                      (step 18) + cell_regime_summary + 121-place polygons
                                      via 19_place_validation.load_places (hotspot =
                                      polygon containing the cluster centroid, as step 19)
  tableS5_clusters.csv   record of Table S5 with admin-dong codes + adjacent centers

Supporting CSV tables:
  T2_model_ladder.csv            Table S1 (model ladder)   <- ladder_summary.json
  T3_regime_characteristics.csv  Table 1 (regime profile)  <- HSMM_emission.csv + regime_diurnal.json
  T4_center_concordance.csv      tier summary behind Table 2
  T5_threshold_sensitivity.csv   short form of Table S4
  T6_sangkwon_gaps.csv           commercial districts outside any designated center

Outputs: results/tables/.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

TBL = C.TABLES_DIR
LAD = C.MODELS_DIR / "ladder_1455c0d960"
TB_ORDER = ["00"] + [f"{h:02d}" for h in range(6, 24)]
TIER_ORDER = ["도심", "광역중심", "지역중심", "지구중심"]
TIER_EN = {"도심": "Downtown", "광역중심": "Regional center",
           "지역중심": "District center", "지구중심": "Local center"}
TIER_SHORT = {"도심": "Downtown", "광역중심": "Regional",
              "지역중심": "District", "지구중심": "Local"}
THR_ROWS = [("abs_0.3", r"Activity share $\geq$ 0.3"),
            ("abs_0.4", r"Activity share $\geq$ 0.4"),
            ("abs_0.5", r"Activity share $\geq$ 0.5"),
            ("abs_0.6", r"Activity share $\geq$ 0.6"),
            ("seoul_mean", r"Activity share $\geq$ Seoul mean (0.23)"),
            ("top_quartile", "Top quartile"),
            ("top_decile", "Top decile"),
            ("dominant_actAB", "Dominant activity regime (main)")]
GU_EN = {"11110": "Jongno", "11140": "Jung", "11170": "Yongsan", "11200": "Seongdong",
         "11215": "Gwangjin", "11230": "Dongdaemun", "11260": "Jungnang", "11290": "Seongbuk",
         "11305": "Gangbuk", "11320": "Dobong", "11350": "Nowon", "11380": "Eunpyeong",
         "11410": "Seodaemun", "11440": "Mapo", "11470": "Yangcheon", "11500": "Gangseo",
         "11530": "Guro", "11545": "Geumcheon", "11560": "Yeongdeungpo", "11590": "Dongjak",
         "11620": "Gwanak", "11650": "Seocho", "11680": "Gangnam", "11710": "Songpa",
         "11740": "Gangdong"}
PLACE_EN = {"홍대입구역(2호선)": "Hongik University Station", "뚝섬역": "Ttukseom Station",
            "압구정로데오거리": "Apgujeong Rodeo Street", "김포공항": "Gimpo Airport",
            "이태원 관광특구": "Itaewon Special Tourist Zone", "충정로역": "Chungjeongno Station",
            "잠실 관광특구": "Jamsil Special Tourist Zone"}


def num(n):
    """Integer with LaTeX thousands separator: 1844 -> 1{,}844."""
    return f"{int(n):,}".replace(",", "{,}")


def pct0(v):
    return f"{v:.0f}"


def tabular(colspec, header, rows):
    out = [rf"\begin{{tabular}}{{{colspec}}}", r"\toprule",
           " & ".join(header) + r" \\", r"\midrule"]
    out += [" & ".join(r) + r" \\" for r in rows]
    out += [r"\botrule", r"\end{tabular}", ""]
    return "\n".join(out)


def table2(pd_stats):
    centers = pl.DataFrame(pd_stats["centers"])
    rows, check = [], []
    for t in TIER_ORDER:
        sub = centers.filter(pl.col("jungsim_tier") == t)
        n, na = sub.height, int(sub["is_active"].sum())
        ma = float(sub["mean_act"].mean())
        rows.append([TIER_EN[t], str(n), str(na), pct0(100 * na / n), f"{ma:.2f}"])
        check.append({"tier": t, "n_centers": n, "n_active": na,
                      "pct_active": round(100 * na / n, 1), "mean_activity_share": ma})
    # consistency with the step-24 tier summary
    for c, s in zip(check, pd_stats["jungsim_concordance_by_tier"]):
        assert (c["n_centers"], c["n_active"]) == (s["n_centers"], s["n_active"]), (c, s)
    hdr = ["Center tier", "Centers", "Activity-dominant centers",
           "Activity-dominant (\\%)", "Mean activity share"]
    return tabular("lrrrr", hdr, rows), check


def tableS4(cr_stats):
    ts = cr_stats["threshold_sensitivity"]
    by = {}
    for r in ts:
        key = "seoul_mean" if r["def"].startswith("seoul_mean") else r["def"]
        by[key] = r
    assert set(by) == {k for k, _ in THR_ROWS}, sorted(by)
    rows = []
    for key, label in THR_ROWS:
        r = by[key]
        rows.append([label, num(r["n_core_cells"]), f"{r['active_not_planned_pct']:.1f}"]
                    + [pct0(r["tier_active_pct"][t]) for t in TIER_ORDER])
    hdr = ["Activity-core definition", "Core cells", "Emergent (\\%)"] + \
          [TIER_SHORT[t] + (" (\\%)" if t == "지구중심" else "") for t in TIER_ORDER]
    return tabular("lrrrrrr", hdr, rows), [by[k] for k, _ in THR_ROWS]


def load_place_module():
    spec = importlib.util.spec_from_file_location(
        "place26", Path(__file__).resolve().parent / "19_place_validation.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def tableS5():
    from shapely.geometry import Point
    cl = pl.read_parquet(C.PROCESSED_DIR / "emergent_activity_clusters.parquet").sort("cluster_id")
    reg = pl.read_parquet(C.PROCESSED_DIR / "cell_regime_summary.parquet").with_columns(
        (pl.col("share_activity_A") + pl.col("share_activity_B")).alias("activity_share"))
    act = dict(zip(reg["cell_id"].to_list(), reg["activity_share"].to_list()))
    places, meta = load_place_module().load_places()
    nm_of = dict(zip(meta["AREA_CD"].to_list(), meta["AREA_NM"].to_list()))
    cat_of = dict(zip(meta["AREA_CD"].to_list(), meta["CATEGORY"].to_list()))
    rows, rec = [], []
    for r in cl.iter_rows(named=True):
        ma = float(np.mean([act[c] for c in r["cell_ids"]]))
        hit = places[places.contains(Point(r["cx"], r["cy"]))]
        nm = cat = None
        if len(hit):                      # same rule as step 26: first containing polygon
            cd = hit.iloc[0]["AREA_CD"]; nm = nm_of.get(cd); cat = cat_of.get(cd)
        gu = r["dominant_admin_dong"][:5]
        hs = PLACE_EN.get(nm, nm) if nm else "--"
        if nm and nm not in PLACE_EN:
            print(f"WARNING: no English name for place {nm}", flush=True)
        rows.append([str(r["cluster_id"]), str(r["n_cells"]), f"{ma:.2f}", GU_EN[gu], hs])
        rec.append({"cluster_id": r["cluster_id"], "n_cells": r["n_cells"],
                    "mean_activity_share": round(ma, 4),
                    "dominant_admin_dong": r["dominant_admin_dong"], "gu_code": gu,
                    "gu": GU_EN[gu], "hotspot_ko": nm, "hotspot_category": cat, "hotspot": hs,
                    "n_hotspot_polygons_containing": len(hit),
                    "touches_center": r["touches_center"],
                    "adjacent_centers": "; ".join(
                        f"{n} [{t}] {v}" for n, t, v in
                        zip(r["adj_centers"], r["adj_center_tiers"], r["adj_pair_counts"]))})
    hdr = ["Rank", "Cells", "Mean activity share", "Gu", "Hotspot"]
    return tabular("rrrll", hdr, rows), pl.DataFrame(rec)


def csv_tables():
    # T2 ladder metrics
    res = json.loads((LAD / "ladder_summary.json").read_text("utf-8"))
    t2 = pl.DataFrame([{"model": r["model"], "K": r["K"], "BIC": r.get("bic"),
                        "holdout_perplexity": r.get("ho_perplexity"),
                        "reinit_ARI": r.get("reinit_ari"), "coherence": r.get("coherence"),
                        "dur_KL": r.get("dur_kl")} for r in res])
    t2.write_csv(TBL / "T2_model_ladder.csv")

    # T3 regime characteristics + diurnal
    em = pl.read_csv(LAD / "HSMM_emission.csv")
    di = json.loads((C.STATS_DIR / "regime_diurnal.json").read_text("utf-8"))
    P = np.array(di["P_regime_given_tb_train"]); names = {int(k): v for k, v in di["regime_names"].items()}
    night = [TB_ORDER.index(b) for b in ["00", "22", "23"]]; day = [TB_ORDER.index(b) for b in ["10","11","12","13","14","15","16"]]
    rows = []
    for r in em.iter_rows(named=True):
        k = r["state"]
        rows.append({**{kk: r[kk] for kk in ["state", "mean_total_pop", "stay_vol",
                                             "short000", "long240", "dwell_missing", "top_demo"]},
                     "name": names.get(k, str(k)),
                     "night_activation": round(float(P[night, k].mean()), 3),
                     "day_activation": round(float(P[day, k].mean()), 3)})
    t3 = pl.DataFrame(rows).select(["state", "name", "mean_total_pop", "long240", "short000",
                                    "dwell_missing", "night_activation", "day_activation", "top_demo"])
    t3.write_csv(TBL / "T3_regime_characteristics.csv")

    # T4 center concordance
    pd = json.loads((C.STATS_DIR / "plan_discordance_stats.json").read_text("utf-8"))
    t4 = pl.DataFrame(pd["jungsim_concordance_by_tier"])
    t4.write_csv(TBL / "T4_center_concordance.csv")

    # T5 active-not-planned threshold sensitivity
    cr = json.loads((C.STATS_DIR / "conclusions_robustness_stats.json").read_text("utf-8"))
    t5 = pl.DataFrame([{"definition": r["def"], "n_core_cells": r["n_core_cells"],
                        "active_not_planned_pct": r["active_not_planned_pct"],
                        "도심_active%": r["tier_active_pct"].get("도심"),
                        "지구중심_active%": r["tier_active_pct"].get("지구중심")}
                       for r in cr["threshold_sensitivity"]])
    t5.write_csv(TBL / "T5_threshold_sensitivity.csv")

    # T6 commercial districts (발달상권) outside designated centers
    sg = json.loads((C.STATS_DIR / "place_validation_stats.json").read_text("utf-8"))
    t6 = pl.DataFrame(sg["sangkwon_planning_gaps"]["gap_list"])
    t6.write_csv(TBL / "T6_sangkwon_gaps.csv")


def main():
    TBL.mkdir(parents=True, exist_ok=True)
    csv_tables()
    pd_stats = json.loads((C.STATS_DIR / "plan_discordance_stats.json").read_text("utf-8"))
    cr_stats = json.loads((C.STATS_DIR / "conclusions_robustness_stats.json").read_text("utf-8"))

    t2, chk2 = table2(pd_stats)
    (TBL / "table2_centers.tex").write_text(t2, encoding="utf-8")
    tS4, chk4 = tableS4(cr_stats)
    (TBL / "tableS4_threshold.tex").write_text(tS4, encoding="utf-8")
    tS5, rec = tableS5()
    (TBL / "tableS5_clusters.tex").write_text(tS5, encoding="utf-8")
    rec.write_csv(TBL / "tableS5_clusters.csv")

    anp = pd_stats["active_not_planned"]
    em = cr_stats["emergent"]
    n_hot = int((rec["hotspot"] != "--").sum())
    summary = {
        "table2": chk2,
        "activity_core_cells": anp["n_activity_dom_cells"],
        "outside_centers": anp["n_outside_any_center"], "pct_outside": anp["pct_outside"],
        "n_clusters": em["n_clusters_ge5"],
        "n_clusters_touching_center": em["n_clusters_touching_center"],
        "n_clusters_in_hotspot": n_hot,
        "pct_clusters_in_hotspot": round(100 * n_hot / rec.height, 1)}
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print("DONE")


if __name__ == "__main__":
    main()
