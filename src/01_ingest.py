"""Step 01 - stream raw zip -> per-product hive-partitioned parquet.

Two products kept SEPARATE (different grain). Per-day part files written under
  data/processed/<product>/year_month=YYYYMM/part-YYYYMMDD.parquet
which together form one queryable hive dataset (memory-safe, restartable).

polars cannot decode CP949 directly, so each zipped CSV member is read as bytes,
decoded cp949 -> re-encoded utf8, then parsed all-as-String (infer_schema=0) and
explicitly cast. Masking token '*' -> null (no 0-imputation). Rounding means
sub-group sums need not equal totals (NOT corrected, by design).

Run (background recommended):
  python src/01_ingest.py both
  python src/01_ingest.py living 202501
"""
from __future__ import annotations

import io
import json
import sys
import time
import zipfile
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

C.set_seed()

# --- read-time column names (positional) -----------------------------------
LIVING_READ = ["일자", "시간", "행정동코드", "250M격자", "생활인구합계"] + [
    f"d{i}" for i in range(28)
]
STAY_READ = (
    ["ETL_YMD", "ADMDONG_CD", "CELL_ID", "UTM250_X", "UTM250_Y",
     "ST_TIMEZN_CD", "STAY_MNUT_CD", "TOTAL"]
    + [f"d{i}" for i in range(16)]
)  # 24 real columns. NOTE: raw rows carry 0 OR 4 trailing empty fields
# (inconsistent across months: 2025.* & 2026.01-04 => 28 fields; 2026.05 => 24).
# We slice the first len(names) columns positionally, so either width is handled.

NULL_TOKENS = ["*", ""]


def _num(src: str, out: str) -> pl.Expr:
    v = pl.col(src).str.strip_chars()
    return (
        pl.when(v.is_in(NULL_TOKENS)).then(None).otherwise(v)
        .cast(pl.Float32).alias(out)
    )


def read_csv_member(zf: zipfile.ZipFile, member: str, names: list[str]) -> pl.DataFrame:
    """Decode CP949->utf8, parse all-as-String, keep first len(names) columns.

    Raw row width is inconsistent across months (trailing empty fields), so we
    do NOT bind new_columns at read time; instead slice the first len(names)
    real columns positionally and rename. Works for any width >= len(names).
    """
    raw = zf.read(member)
    text = raw.decode(C.RAW_ENCODING, errors="replace")
    df = pl.read_csv(
        text.encode("utf-8"),
        has_header=False,
        skip_rows=1,
        infer_schema_length=0,      # all columns -> String
        quote_char='"',
        truncate_ragged_lines=True,
        low_memory=True,
    )
    df = df.select(df.columns[: len(names)])
    df.columns = names
    return df


def transform_living(df: pl.DataFrame) -> pl.DataFrame:
    return df.select(
        pl.col("일자").str.strptime(pl.Date, "%Y%m%d").alias("date"),
        pl.col("시간").str.strip_chars().cast(pl.Int8).alias("hour"),
        pl.col("행정동코드").str.strip_chars().alias("admin_dong"),
        pl.col("250M격자").str.strip_chars().alias("grid_id"),
        _num("생활인구합계", "total_pop"),  # total can also be '*' masked
        *[_num(f"d{i}", out) for i, out in enumerate(C.LIVING_DEMO_COLS)],
    )


def transform_stay(df: pl.DataFrame) -> pl.DataFrame:
    return df.select(
        pl.col("ETL_YMD").str.strptime(pl.Date, "%Y%m%d").alias("date"),
        pl.col("ADMDONG_CD").str.strip_chars().alias("admin_dong"),
        pl.col("CELL_ID").str.strip_chars().alias("cell_id"),
        pl.col("UTM250_X").str.strip_chars().cast(pl.Float64).alias("utm_x"),
        pl.col("UTM250_Y").str.strip_chars().cast(pl.Float64).alias("utm_y"),
        pl.col("ST_TIMEZN_CD").str.strip_chars().alias("start_tz"),
        pl.col("STAY_MNUT_CD").str.strip_chars().alias("stay_dur"),
        _num("TOTAL", "total"),  # total can also be '*' masked
        *[_num(f"d{i}", out) for i, out in enumerate(C.STAY_DEMO_COLS)],
    )


PRODUCTS = {
    "living": dict(
        zip_tmpl=C.LIVING_ZIP, out_dir=C.LIVING_DIR, read_names=LIVING_READ,
        transform=transform_living, demo=C.LIVING_DEMO_COLS,
        grid_col="grid_id", total_col="total_pop",
    ),
    "stay": dict(
        zip_tmpl=C.STAY_ZIP, out_dir=C.STAY_DIR, read_names=STAY_READ,
        transform=transform_stay, demo=C.STAY_DEMO_COLS,
        grid_col="cell_id", total_col="total",
    ),
}


def ingest_product(prod: str, months: list[str], force: bool = False) -> dict:
    spec = PRODUCTS[prod]
    out_root: Path = spec["out_dir"]
    out_root.mkdir(parents=True, exist_ok=True)
    log_path = C.LOGS_DIR / f"ingest_{prod}.log"

    def plog(msg: str):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    stats = {}
    for ym in months:
        zp = C.RAW_DIR / spec["zip_tmpl"].format(ym=ym)
        part_dir = out_root / f"year_month={ym}"
        part_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        m_rows = 0
        m_grids: set[str] = set()
        m_null_cells = 0
        m_total_null = 0
        m_demo_cells = 0
        dmin, dmax = None, None
        per_day = {}

        with zipfile.ZipFile(zp) as zf:
            members = sorted(m for m in zf.namelist() if m.lower().endswith(".csv"))
            for member in members:
                day = member[-12:-4]  # YYYYMMDD
                out_path = part_dir / f"part-{day}.parquet"
                if out_path.exists() and not force:
                    # restart: read stats back cheaply
                    sdf = pl.read_parquet(out_path, columns=[spec["grid_col"]])
                    n = sdf.height
                    m_rows += n
                    m_grids |= set(sdf[spec["grid_col"]].unique().to_list())
                    per_day[day] = n
                    continue
                df = read_csv_member(zf, member, spec["read_names"])
                out = spec["transform"](df)
                out.write_parquet(out_path, compression="zstd")
                n = out.height
                m_rows += n
                per_day[day] = n
                m_grids |= set(out[spec["grid_col"]].unique().to_list())
                # null accounting on demographic columns
                nulls = out.select(
                    pl.sum_horizontal(pl.col(c).is_null().sum() for c in spec["demo"])
                ).item()
                m_null_cells += int(nulls)
                m_demo_cells += n * len(spec["demo"])
                m_total_null += int(out[spec["total_col"]].is_null().sum())
                d0, d1 = out["date"].min(), out["date"].max()
                dmin = d0 if dmin is None else min(dmin, d0)
                dmax = d1 if dmax is None else max(dmax, d1)
                del df, out

        elapsed = time.time() - t0
        stats[ym] = dict(
            rows=m_rows, n_days=len(per_day), uniq_grid=len(m_grids),
            date_min=str(dmin), date_max=str(dmax),
            demo_null_pct=round(100 * m_null_cells / max(m_demo_cells, 1), 2)
            if m_demo_cells else None,
            total_null=m_total_null,
            rows_per_day_min=min(per_day.values()) if per_day else 0,
            rows_per_day_max=max(per_day.values()) if per_day else 0,
            sec=round(elapsed, 1),
        )
        plog(f"{prod} {ym}: rows={m_rows:,} days={len(per_day)} "
             f"grids={len(m_grids):,} range={dmin}..{dmax} "
             f"demo_null={stats[ym]['demo_null_pct']}% total_null={m_total_null} "
             f"({elapsed:.1f}s)")

    (C.LOGS_DIR / f"ingest_{prod}_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    return stats


def main():
    args = sys.argv[1:]
    prod = args[0] if args else "both"
    months = [a for a in args[1:] if a.isdigit()]
    force = "--force" in args
    if not months:
        months = C.MONTHS
    targets = ["living", "stay"] if prod == "both" else [prod]
    for p in targets:
        print(f"=== INGEST {p} : {len(months)} month(s) ===", flush=True)
        ingest_product(p, months, force=force)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
