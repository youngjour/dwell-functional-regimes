"""Step 13 - ingest subway time-of-day ridership (OA-12252, CardSubwayTime).

Service requires USE_MM (month). Each record = monthly aggregate board/alight per
(line x station x hour-of-day). No station_code in this API (station name only).
Hours are reported on the operating day order HR_4..HR_23, HR_0..HR_3 -> mapped to
int hour 0..23 (1-3am usually ~0 = no service).

Output -> data/processed/subway_ridership/year_month=YYYYMM/part.parquet
schema: use_mm | line | station_name | hour | board | alight

The Seoul Open Data Plaza API key is read from the environment variable
SEOUL_OPENAPI_KEY (or a SEOUL_OPENAPI_KEY=... line in <repo>/.env, never committed).

Run: python src/13_ingest_subway.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

SERVICE = "CardSubwayTime"
PAGE = 1000
HOURS = list(range(4, 24)) + list(range(0, 4))   # operating-day order
OUT = C.PROCESSED_DIR / "subway_ridership"


def load_key():
    k = os.environ.get("SEOUL_OPENAPI_KEY")
    if k:
        return k.strip()
    p = C.ROOT / ".env"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("SEOUL_OPENAPI_KEY"):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("SEOUL_OPENAPI_KEY not found (environment variable or .env)")


def fetch_page(key, ym, start, end, retries=4):
    url = f"http://openapi.seoul.go.kr:8088/{key}/json/{SERVICE}/{start}/{end}/{ym}/"
    for a in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=40) as r:
                d = json.loads(r.read().decode("utf-8"))
            body = d.get(SERVICE, {})
            code = body.get("RESULT", {}).get("CODE")
            if code not in ("INFO-000", None):
                raise RuntimeError(f"API code {code}: {body.get('RESULT',{}).get('MESSAGE')}")
            return body.get("list_total_count", 0), body.get("row", [])
        except Exception as e:  # noqa: BLE001
            if a == retries - 1:
                raise
            time.sleep(1.5 * (a + 1))
    return 0, []


def fetch_month(key, ym):
    total, rows = fetch_page(key, ym, 1, PAGE)
    got = len(rows)
    while got < total:
        _, more = fetch_page(key, ym, got + 1, min(got + PAGE, total))
        if not more:
            break
        rows += more; got += len(more)
        time.sleep(0.2)
    return total, rows


def tidy(rows):
    if not rows:
        return pl.DataFrame()
    df = pl.DataFrame(rows)
    recs = []
    for h in HOURS:
        on = f"HR_{h}_GET_ON_NOPE"; off = f"HR_{h}_GET_OFF_NOPE"
        if on not in df.columns:
            continue
        recs.append(df.select(
            pl.col("USE_MM").alias("use_mm"),
            pl.col("SBWY_ROUT_LN_NM").str.strip_chars().alias("line"),
            pl.col("STTN").str.strip_chars().alias("station_name"),
            pl.lit(h, dtype=pl.Int8).alias("hour"),
            pl.col(on).cast(pl.Float64).alias("board"),
            pl.col(off).cast(pl.Float64).alias("alight"),
        ))
    out = pl.concat(recs)
    # some months (e.g. 202603) are returned duplicated by the API -> dedup.
    return out.unique(subset=["use_mm", "line", "station_name", "hour"],
                      keep="first").sort(["station_name", "line", "hour"])


def main():
    key = load_key()
    OUT.mkdir(parents=True, exist_ok=True)
    log = []
    for ym in C.MONTHS:
        t0 = time.time()
        total, rows = fetch_month(key, ym)
        df = tidy(rows)
        d = OUT / f"year_month={ym}"
        d.mkdir(parents=True, exist_ok=True)
        df.write_parquet(d / "part.parquet", compression="zstd")
        info = {"month": ym, "api_records": total, "fetched": len(rows),
                "tidy_rows": df.height, "stations": df["station_name"].n_unique() if df.height else 0,
                "lines": df["line"].n_unique() if df.height else 0,
                "sec": round(time.time() - t0, 1)}
        log.append(info)
        print(f"{ym}: api={total} tidy_rows={df.height:,} stations={info['stations']} "
              f"lines={info['lines']} ({info['sec']}s)", flush=True)
    (C.LOGS_DIR / "subway_ingest_log.json").write_text(
        json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    print("DONE")


if __name__ == "__main__":
    main()
