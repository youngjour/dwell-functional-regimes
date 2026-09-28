"""Step 04 - build emission feature table (cell x date x 19 bin).

Presence (living, 19-bin downsample) + dwell (stay, start-bin shares) joined on
(cell_id, date, time_bin), restricted to the active-intersection universe.
Also writes a presence-only table over all active_50 living cells.

Per-month streaming aggregation -> data/processed/emission/year_month=YYYYMM/part.parquet
and data/processed/emission_presence_only/year_month=YYYYMM/part.parquet.

Run: python src/04_build_emission.py
     python src/04_build_emission.py 202501
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

C.set_seed()

TRAIN_MAX = "202603"           # train: 2025.01 .. 2026.03 ; holdout: 2026.04-05
DUR_CODES = ["000", "030", "060", "120", "180", "240"]

# living 14-band -> 8-band (10yr) harmonization
BAND8 = {
    "00": ["0_9"], "10": ["10_14", "15_19"], "20": ["20_24", "25_29"],
    "30": ["30_34", "35_39"], "40": ["40_44", "45_49"], "50": ["50_54", "55_59"],
    "60": ["60_64", "65_69"], "70": ["70_over"],
}
MIX_COLS = [f"mix_{s}{b}" for s in ["M", "F"] for b in BAND8]   # 16
BAND_TMP = [f"b_{s}{b}" for s in ["M", "F"] for b in BAND8]


def time_bin_expr(hour_col: str = "hour") -> pl.Expr:
    return (pl.when(pl.col(hour_col) <= 5).then(pl.lit("00"))
            .otherwise(pl.col(hour_col).cast(pl.Int32).cast(pl.Utf8).str.zfill(2))
            .alias("time_bin"))


def presence_month(ym: str, keep_cells: list[str]) -> pl.DataFrame:
    lf = pl.scan_parquet(str(C.LIVING_DIR / f"year_month={ym}" / "*.parquet"))
    lf = lf.filter(pl.col("grid_id").is_in(keep_cells))
    # harmonize 14->8 (fill_null(0): masked demo treated as 0 for mix)
    harm = []
    for s in ["M", "F"]:
        for b8, comps in BAND8.items():
            harm.append(
                pl.sum_horizontal([pl.col(f"{s}_{c}").fill_null(0) for c in comps])
                .alias(f"b_{s}{b8}"))
    lf = lf.with_columns(time_bin_expr(), *harm)
    agg = (lf.group_by(["grid_id", "date", "time_bin"]).agg(
        pl.col("total_pop").mean().alias("mean_total_pop"),
        pl.col("total_pop").count().alias("presence_n_hours"),
        *[pl.col(t).mean().alias(t) for t in BAND_TMP],
    ).collect(engine="streaming"))

    band_sum = pl.sum_horizontal([pl.col(t) for t in BAND_TMP])
    agg = agg.with_columns(band_sum.alias("_bsum"))
    out = agg.with_columns(
        pl.col("mean_total_pop").log1p().alias("presence_log"),
        (pl.col("_bsum") <= 0).alias("presence_demo_masked"),
        *[pl.when(pl.col("_bsum") > 0)
          .then(pl.col(t) / pl.col("_bsum")).otherwise(None).alias(m)
          for t, m in zip(BAND_TMP, MIX_COLS)],
    ).drop(["_bsum", *BAND_TMP]).rename({"grid_id": "cell_id"})
    # drop cell-bins with no presence signal at all
    out = out.filter(pl.col("mean_total_pop").is_not_null())
    return out


def dwell_month(ym: str, keep_cells: list[str]) -> pl.DataFrame:
    lf = pl.scan_parquet(str(C.STAY_DIR / f"year_month={ym}" / "*.parquet"))
    lf = lf.filter(pl.col("cell_id").is_in(keep_cells))
    agg = (lf.group_by(["cell_id", "date", "start_tz"]).agg(
        pl.col("total").sum().alias("denom"),
        pl.col("total").count().alias("dwell_n_codes"),
        *[pl.col("total").filter(pl.col("stay_dur") == c).sum().alias(f"v_{c}")
          for c in DUR_CODES],
    ).collect(engine="streaming"))
    # denom = sum of non-null totals over the (<=6) stay_dur rows.
    # polars sum(all-null)=0, so denom==0 means the whole (cell,date,bin) is
    # fully masked -> no usable dwell signal: shares & stay_vol_log -> null.
    out = agg.with_columns(
        *[pl.when(pl.col("denom") > 0)
          .then(pl.col(f"v_{c}") / pl.col("denom")).otherwise(None)
          .alias(f"dwell_share_{c}") for c in DUR_CODES],
        pl.when(pl.col("denom") > 0).then(pl.col("denom").log1p())
        .otherwise(None).alias("stay_vol_log"),
        (pl.col("dwell_n_codes") <= 1).alias("dwell_heavy_masked"),
    )
    out = out.with_columns(
        pl.col("dwell_share_000").alias("short_share"),
        pl.col("dwell_share_240").alias("long_share"),
    ).rename({"start_tz": "time_bin"}).drop([f"v_{c}" for c in DUR_CODES] + ["denom"])
    return out


def build_month(ym: str, universe: list[str], active50: list[str]):
    split = "train" if ym <= TRAIN_MAX else "holdout"

    # presence over all active_50 cells (superset) -> reused for both tables
    pres_all = presence_month(ym, active50).with_columns(pl.lit(split).alias("split"))

    # presence-only table (all active_50 living cells)
    po_dir = C.PROCESSED_DIR / "emission_presence_only" / f"year_month={ym}"
    po_dir.mkdir(parents=True, exist_ok=True)
    pres_all.write_parquet(po_dir / "part.parquet", compression="zstd")

    # main emission: presence restricted to universe + dwell join
    uni_set = set(universe)
    pres = pres_all.filter(pl.col("cell_id").is_in(list(uni_set)))
    dwell = dwell_month(ym, universe)

    dwell_cols = ([f"dwell_share_{c}" for c in DUR_CODES]
                  + ["short_share", "long_share", "stay_vol_log",
                     "dwell_n_codes", "dwell_heavy_masked"])
    em = pres.join(dwell, on=["cell_id", "date", "time_bin"], how="left")
    em = em.with_columns(pl.col("stay_vol_log").is_not_null().alias("dwell_present"))

    em_dir = C.PROCESSED_DIR / "emission" / f"year_month={ym}"
    em_dir.mkdir(parents=True, exist_ok=True)
    # column order
    order = (["cell_id", "date", "time_bin", "split", "presence_log",
              "mean_total_pop", "presence_n_hours", "presence_demo_masked"]
             + MIX_COLS + dwell_cols + ["dwell_present"])
    em = em.select([c for c in order if c in em.columns])
    em.write_parquet(em_dir / "part.parquet", compression="zstd")

    n_dwell = int(em["dwell_present"].sum())
    print(f"{ym} [{split}]: emission rows={em.height:,} cells={em['cell_id'].n_unique():,}"
          f" dwell_present={100*n_dwell/em.height:.1f}% | presence_only rows={pres_all.height:,}",
          flush=True)


def main():
    cm = pl.read_parquet(C.PROCESSED_DIR / "cell_master.parquet")
    universe = cm.filter(pl.col("in_universe"))["cell_id"].to_list()
    active50 = cm.filter(pl.col("active_50"))["cell_id"].to_list()
    print(f"universe cells={len(universe):,} active50 cells={len(active50):,}")
    months = [a for a in sys.argv[1:] if a.isdigit()] or C.MONTHS
    for ym in months:
        build_month(ym, universe, active50)
    print("DONE")


if __name__ == "__main__":
    main()
