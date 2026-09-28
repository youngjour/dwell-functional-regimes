"""Step 15 - geocode subway stations & bus stops -> 250m universe cells.

Bus: bus_ridership_bystop.stop_id <-> stop master (서울시 정류장마스터 정보) 정류장_ID
(ID-first, names ignored).
Subway: subway_ridership (line+station) <-> station master (서울시 역사마스터 정보,
호선+역명); station name normalised (strip parenthetical sub-names / spaces /
trailing '역'), then residual fuzzy. The ridership API has no station ID, so we match
on name (line as tiebreak), taking the centroid coordinate over matching master rows
(transfer stations ~co-located).

WGS84 (lat/lon) -> EPSG:5179, snapped to the 250m cell grid (centres offset 125m).
Outputs station/stop -> cell maps (+ grid_ix/iy for later k-ring catchments).

Run: python src/15_geocode_join.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

SRC = C.TRANSPORT_RAW_DIR
STEP = 250.0
X0 = 936375.0
Y0 = 1937125.0


def read_cp949(path: Path) -> pl.DataFrame:
    txt = path.read_bytes().decode("cp949", errors="replace")
    return pl.read_csv(txt.encode("utf-8"), infer_schema_length=0)


def reproject(lat, lon):
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", "EPSG:5179", always_xy=True)
    x, y = tr.transform(np.asarray(lon, float), np.asarray(lat, float))
    return np.asarray(x), np.asarray(y)


def norm_station(s: str) -> str:
    s = re.sub(r"\(.*?\)", "", s)          # drop parenthetical sub-name
    s = s.replace(" ", "").strip()
    if len(s) > 1 and s.endswith("역"):
        s = s[:-1]
    return s


def snap_cell(x, y, cellmap):
    ix = np.round((x - X0) / STEP).astype(int)
    iy = np.round((y - Y0) / STEP).astype(int)
    cid = np.array([cellmap.get((int(a), int(b))) for a, b in zip(ix, iy)], dtype=object)
    return cid, ix, iy


def main():
    reg = pl.read_parquet(C.PROCESSED_DIR / "cell_regime_summary.parquet")
    cix = np.round((reg["utm_x"].to_numpy() - X0) / STEP).astype(int)
    ciy = np.round((reg["utm_y"].to_numpy() - Y0) / STEP).astype(int)
    cellmap = {(int(a), int(b)): c for a, b, c in
               zip(cix, ciy, reg["cell_id"].to_list())}
    print(f"universe cells in grid map: {len(cellmap)}", flush=True)
    report = {}

    # ---------- BUS ----------
    bm = read_cp949(SRC / "서울시 정류장마스터 정보.csv").select(
        pl.col("정류장_ID").str.strip_chars().alias("stop_id"),
        pl.col("위도").cast(pl.Float64).alias("lat"),
        pl.col("경도").cast(pl.Float64).alias("lon"))
    stops = (pl.scan_parquet(str(C.PROCESSED_DIR / "bus_ridership_bystop" /
                                 "year_month=*" / "*.parquet"))
             .select("stop_id", "stop_name").unique().collect())
    bj = stops.join(bm, on="stop_id", how="left")
    have = bj.filter(pl.col("lat").is_not_null())
    bx, by = reproject(have["lat"].to_numpy(), have["lon"].to_numpy())
    cid, gix, giy = snap_cell(bx, by, cellmap)
    bus_map = have.with_columns(
        pl.Series("utm_x", bx), pl.Series("utm_y", by),
        pl.Series("cell_id", cid, dtype=pl.Utf8),
        pl.Series("grid_ix", gix), pl.Series("grid_iy", giy)).filter(
        pl.col("cell_id").is_not_null())
    bus_map.write_parquet(C.PROCESSED_DIR / "stop_cell_map.parquet")
    report["bus"] = {
        "stops_total": stops.height,
        "geocoded": int(have.height),
        "geocode_rate": round(100 * have.height / stops.height, 1),
        "in_universe_cell": int(bus_map.height),
        "universe_rate_of_geocoded": round(100 * bus_map.height / have.height, 1)}
    print("bus:", report["bus"], flush=True)

    # ---------- SUBWAY ----------
    sm = read_cp949(SRC / "서울시 역사마스터 정보.csv").select(
        pl.col("역사명").alias("st_name"), pl.col("호선").alias("line"),
        pl.col("위도").cast(pl.Float64).alias("lat"),
        pl.col("경도").cast(pl.Float64).alias("lon"))
    sm = sm.with_columns(pl.col("st_name").map_elements(norm_station,
                         return_dtype=pl.Utf8).alias("nkey"))
    # centroid coord per normalised name
    sm_c = sm.group_by("nkey").agg(pl.col("lat").mean(), pl.col("lon").mean(),
                                   pl.col("st_name").first())

    sub = (pl.scan_parquet(str(C.PROCESSED_DIR / "subway_ridership" /
                               "year_month=*" / "*.parquet"))
           .select("station_name").unique().collect())
    sub = sub.with_columns(pl.col("station_name").map_elements(norm_station,
                           return_dtype=pl.Utf8).alias("nkey"))
    sj = sub.join(sm_c, on="nkey", how="left")
    matched = sj.filter(pl.col("lat").is_not_null())
    unmatched = sj.filter(pl.col("lat").is_null())["station_name"].to_list()

    # fuzzy for residual (difflib on normalised keys)
    if unmatched:
        import difflib
        keys = sm_c["nkey"].to_list()
        fz = []
        for st in unmatched:
            nk = norm_station(st)
            cand = difflib.get_close_matches(nk, keys, n=1, cutoff=0.8)
            if cand:
                row = sm_c.filter(pl.col("nkey") == cand[0]).row(0, named=True)
                fz.append({"station_name": st, "nkey": cand[0],
                           "lat": row["lat"], "lon": row["lon"], "st_name": row["st_name"]})
        if fz:
            matched = pl.concat([matched, pl.DataFrame(fz).select(matched.columns)])
        unmatched = [u for u in unmatched
                     if u not in {f["station_name"] for f in fz}]

    sx, sy = reproject(matched["lat"].to_numpy(), matched["lon"].to_numpy())
    cid, gix, giy = snap_cell(sx, sy, cellmap)
    sub_map = matched.with_columns(
        pl.Series("utm_x", sx), pl.Series("utm_y", sy),
        pl.Series("cell_id", cid, dtype=pl.Utf8),
        pl.Series("grid_ix", gix), pl.Series("grid_iy", giy))
    sub_in = sub_map.filter(pl.col("cell_id").is_not_null())
    sub_map.write_parquet(C.PROCESSED_DIR / "station_cell_map.parquet")
    report["subway"] = {
        "stations_total": sub.height,
        "matched_master": int(matched.height),
        "match_rate": round(100 * matched.height / sub.height, 1),
        "in_universe_cell": int(sub_in.height),
        "universe_rate_of_matched": round(100 * sub_in.height / matched.height, 1),
        "unmatched_stations": unmatched}
    print("subway:", {k: v for k, v in report["subway"].items()
                      if k != "unmatched_stations"}, flush=True)
    print("unmatched stations:", unmatched, flush=True)

    (C.STATS_DIR / "geocode_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote station_cell_map.parquet, stop_cell_map.parquet, geocode_report.json")
    print("DONE")


if __name__ == "__main__":
    main()
