"""Step 02 - cell universe -> data/processed/cell_master.parquet.

Universe = (living grid_id ∩ stay cell_id, full 17 months) with coords+admin
attached, plus full-period mean total_pop and active flags at 25/50/100.

Active threshold: per-cell mean total_pop over all (date,hour)
non-null living records >= 50. Universe selection uses the FULL period (allowed;
not a leakage source — only the scaler is train-only).

Run: python src/02_cell_master.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

C.set_seed()

ACTIVE_THR = 50  # persons (mean total_pop per cell-hour record)


def main():
    living = pl.scan_parquet(str(C.LIVING_DIR / "year_month=*" / "*.parquet"))
    stay = pl.scan_parquet(str(C.STAY_DIR / "year_month=*" / "*.parquet"))

    # per-cell mean total_pop (null excluded) + obs count, full period
    living_cell = (living.group_by("grid_id").agg(
        pl.col("total_pop").mean().alias("mean_total_pop"),
        pl.col("total_pop").count().alias("n_nonnull_obs"),
        pl.len().alias("n_rows"),
    ).collect(engine="streaming"))
    print(f"living cells (any): {living_cell.height:,}")

    stay_cells = (stay.select("cell_id").unique().collect(engine="streaming")
                  ["cell_id"])
    stay_set = set(stay_cells.to_list())
    print(f"stay cells (any): {len(stay_set):,}")

    # admin_dong: full-period modal per cell (most frequent admin over all months)
    admin = (living.group_by(["grid_id", "admin_dong"]).agg(pl.len().alias("n"))
             .collect(engine="streaming")
             .sort(["grid_id", "n"], descending=[False, True])
             .group_by("grid_id", maintain_order=True)
             .agg(pl.col("admin_dong").first()))

    # coords from shapefile; fallback to stay UTM (modal) for cells absent in shp
    import geopandas as gpd
    g = gpd.read_file(C.GRID_SHP)
    coords = pl.DataFrame({
        "grid_id": g["CELL_ID"].astype(str).tolist(),
        "utm_x": g["CELL_X"].astype(float).tolist(),
        "utm_y": g["CELL_Y"].astype(float).tolist(),
    })
    stay_xy = (stay.group_by("cell_id").agg(
        pl.col("utm_x").first().alias("sx"), pl.col("utm_y").first().alias("sy"))
        .collect(engine="streaming").rename({"cell_id": "grid_id"}))

    master = (living_cell
              .join(admin, on="grid_id", how="left")
              .join(coords, on="grid_id", how="left")
              .join(stay_xy, on="grid_id", how="left")
              .with_columns(
                  pl.col("utm_x").fill_null(pl.col("sx")),
                  pl.col("utm_y").fill_null(pl.col("sy")),
              )
              .drop(["sx", "sy"])
              .with_columns(
                  pl.col("grid_id").is_in(list(stay_set)).alias("in_stay"),
                  (pl.col("mean_total_pop") >= 25).alias("active_25"),
                  (pl.col("mean_total_pop") >= 50).alias("active_50"),
                  (pl.col("mean_total_pop") >= 100).alias("active_100"),
              )
              .rename({"grid_id": "cell_id"}))

    # universe = intersection ∩ active_50
    master = master.with_columns(
        (pl.col("in_stay") & pl.col("active_50")).alias("in_universe"))

    # threshold counts (logging)
    counts = {
        "living_any": living_cell.height,
        "stay_any": len(stay_set),
        "intersection": int(master["in_stay"].sum()),
        "active_25_all": int(master["active_25"].sum()),
        "active_50_all": int(master["active_50"].sum()),
        "active_100_all": int(master["active_100"].sum()),
        "intersection_active_25": int((master["in_stay"] & master["active_25"]).sum()),
        "intersection_active_50": int((master["in_stay"] & master["active_50"]).sum()),
        "intersection_active_100": int((master["in_stay"] & master["active_100"]).sum()),
        "shapefile": len(g),
        "coords_missing": int(master["utm_x"].is_null().sum()),
        "admin_missing": int(master["admin_dong"].is_null().sum()),
        "UNIVERSE": int(master["in_universe"].sum()),
    }

    out = C.PROCESSED_DIR / "cell_master.parquet"
    master.write_parquet(out)
    (C.STATS_DIR / "cell_master_counts.json").write_text(
        json.dumps(counts, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote", out)
    print(json.dumps(counts, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
