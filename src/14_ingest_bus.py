"""Step 14 - ingest bus time-of-day ridership (OA-12913, local CP949 CSVs).

Files: data/Seoul Transport/*버스노선별_정류장별_시간대별_승하차_인원_정보*.csv (17 monthly). 57 cols = 6 meta + 48 hour
(board/alight x 24h, labels 00,1..23) + 3 (transit-type code/name, reg date).
Each record = monthly aggregate board/alight at (route x stop x hour-of-day).

Outputs (per month hive):
  data/processed/bus_ridership/        route-level tidy
  data/processed/bus_ridership_bystop/ stop-level (summed over routes; station-area comparison)
schema(route): use_ym|route_no|route_name|stop_id|ars_no|stop_name|
               transit_type_code|transit_type_name|hour|board|alight
schema(stop):  use_ym|stop_id|stop_name|hour|board|alight

Run: python src/14_ingest_bus.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

SRC = C.TRANSPORT_RAW_DIR
OUT_R = C.PROCESSED_DIR / "bus_ridership"
OUT_S = C.PROCESSED_DIR / "bus_ridership_bystop"
ENC = "cp949"
BUS_GLOB = "*버스노선별_정류장별_시간대별_승하차_인원_정보*.csv"

META = ["use_ym", "route_no", "route_name", "stop_id", "ars_no", "stop_name"]
HOUR_LABELS = ["00"] + [str(h) for h in range(1, 24)]      # 00,1,2,...,23
HOUR_COLS = []
for hl in HOUR_LABELS:
    HOUR_COLS += [f"b_{hl}", f"a_{hl}"]                    # board,alight interleaved
TRAIL = ["transit_type_code", "transit_type_name", "reg_date"]
READ = META + HOUR_COLS + TRAIL                            # 57


def read_csv(path: Path) -> pl.DataFrame:
    text = path.read_bytes().decode(ENC, errors="replace")
    df = pl.read_csv(text.encode("utf-8"), has_header=False, skip_rows=1,
                     infer_schema_length=0, new_columns=READ, quote_char='"',
                     truncate_ragged_lines=True)
    assert df.width == len(READ), (df.width, len(READ), path.name)
    return df


def tidy(df: pl.DataFrame) -> pl.DataFrame:
    sid = [pl.col(c).str.strip_chars() for c in
           ["use_ym", "route_no", "route_name", "stop_id", "ars_no", "stop_name",
            "transit_type_code", "transit_type_name"]]
    recs = []
    for hl in HOUR_LABELS:
        recs.append(df.select(
            *sid,
            pl.lit(int(hl), dtype=pl.Int8).alias("hour"),
            pl.col(f"b_{hl}").str.strip_chars().cast(pl.Float64).alias("board"),
            pl.col(f"a_{hl}").str.strip_chars().cast(pl.Float64).alias("alight"),
        ))
    cols = (["use_ym", "route_no", "route_name", "stop_id", "ars_no", "stop_name",
             "transit_type_code", "transit_type_name", "hour", "board", "alight"])
    return pl.concat(recs).select(cols).sort(["stop_id", "route_no", "hour"])


def main():
    OUT_R.mkdir(parents=True, exist_ok=True)
    OUT_S.mkdir(parents=True, exist_ok=True)
    files = sorted(SRC.glob(BUS_GLOB))   # skip the station/stop master CSVs in SRC
    log = []
    for f in files:
        t0 = time.time()
        df = read_csv(f)
        td = tidy(df)
        ym = td["use_ym"][0]
        dr = OUT_R / f"year_month={ym}"; dr.mkdir(parents=True, exist_ok=True)
        td.write_parquet(dr / "part.parquet", compression="zstd")
        # stop-level: sum over routes
        st = (td.group_by(["use_ym", "stop_id", "stop_name", "hour"])
              .agg(pl.col("board").sum(), pl.col("alight").sum())
              .sort(["stop_id", "hour"]))
        ds = OUT_S / f"year_month={ym}"; ds.mkdir(parents=True, exist_ok=True)
        st.write_parquet(ds / "part.parquet", compression="zstd")
        info = {"file": f.name, "use_ym": ym, "src_rows": df.height,
                "route_rows": td.height, "stop_rows": st.height,
                "n_routes": td["route_no"].n_unique(), "n_stops": td["stop_id"].n_unique(),
                "sec": round(time.time() - t0, 1)}
        log.append(info)
        print(f"{ym}: src={df.height:,} route_rows={td.height:,} stop_rows={st.height:,} "
              f"routes={info['n_routes']} stops={info['n_stops']} ({info['sec']}s)", flush=True)
    (C.LOGS_DIR / "bus_ingest_log.json").write_text(
        json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    print("DONE")


if __name__ == "__main__":
    main()
