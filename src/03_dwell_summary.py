"""Step 03 - dwell-duration summary of the stay-population table (Supplementary figure
`dwell_distribution`).

On the sample months (default 2025.01, 2025.04, 2025.10, 2026.05):
  - A_overall_share : population-weighted share of each STAY_MNUT_CD dwell interval
  - start_tz_by_dur : population per (dwell interval x start-time bin), 6 x 19

This is the dwell part (A) of the original exploratory EDA step; the other EDA panels
were not used in the manuscript and are omitted.

Output: results/stats/dwell_summary.json

Run: python src/03_dwell_summary.py
     python src/03_dwell_summary.py 202501 202504
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

C.set_seed()

SAMPLE_MONTHS = ["202501", "202504", "202510", "202605"]
DUR_ORDER = ["000", "030", "060", "120", "180", "240"]
TZ_ORDER = ["00"] + [f"{h:02d}" for h in range(6, 24)]


def stay_lf(months):
    parts = []
    for ym in months:
        d = C.STAY_DIR / f"year_month={ym}"
        if d.exists():
            parts.append(str(d / "*.parquet"))
    return pl.scan_parquet(parts)


def main():
    months = [a for a in sys.argv[1:] if a.isdigit()] or SAMPLE_MONTHS
    lf = stay_lf(months)
    # overall pop-weighted share per stay_dur
    overall = (lf.group_by("stay_dur")
               .agg(pl.col("total").sum().alias("pop"))
               .collect())
    o = {r["stay_dur"]: r["pop"] for r in overall.iter_rows(named=True)}
    tot = sum(o.values())
    shares = {k: o.get(k, 0) / tot for k in DUR_ORDER}
    # population by stay_dur x start_tz (6 x 19)
    bytz = (lf.group_by(["start_tz", "stay_dur"])
            .agg(pl.col("total").sum().alias("pop")).collect())
    M = [[0.0] * len(TZ_ORDER) for _ in DUR_ORDER]
    for r in bytz.iter_rows(named=True):
        if r["start_tz"] in TZ_ORDER and r["stay_dur"] in DUR_ORDER:
            M[DUR_ORDER.index(r["stay_dur"])][TZ_ORDER.index(r["start_tz"])] = float(r["pop"])
    out = {"_months": months, "dur_order": DUR_ORDER, "tz_order": TZ_ORDER,
           "A_overall_share": shares, "start_tz_by_dur": M}
    (C.STATS_DIR / "dwell_summary.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote results/stats/dwell_summary.json")


if __name__ == "__main__":
    main()
