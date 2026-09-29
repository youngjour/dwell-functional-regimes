"""Build the derived-data package (Zenodo) needed to rerun the post-model steps.

Collects the minimal set of files that lets steps 12 and 16-23 run WITHOUT the raw
mobile-network data (~20 GB) and WITHOUT refitting the models. The package is split
by content into several zips (each under 20 MB, so they upload reliably through a
browser); unzipping all of them into one folder gives the complete
`dfr_derived_data/` tree:

  dfr_derived_data_core.zip      everything except the monthly bus-stop ridership
                                 partitions, + README_data.md
  dfr_derived_data_bus_<g>.zip   processed/bus_ridership_bystop/year_month=... (BUS_GROUPS)

Also written: one file listing per zip (<zip name>_files.tsv) and SHA256SUMS.txt.
All outputs are written OUTSIDE the repository.

Sources (defaults follow the repository layout; override with the options):
  --data-dir    data tree after steps 00-15 (default: $DFR_DATA_DIR or <repo>/data)
  --models-dir  fitted models (default: $DFR_MODELS_DIR or <repo>/results/models)
  --stats-dir   JSON outputs of steps 02, 03, 11, 20 (default: <repo>/results/stats)
  --out-dir     where the zip is written (default: <repo>/../dwell-functional-regimes-zenodo)

Run: python scripts/package_derived_data.py
"""
from __future__ import annotations

import argparse
import hashlib
import os
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PKG = "dfr_derived_data"
LADDER = "ladder_1455c0d960"

PROCESSED_FILES = [
    "cell_master.parquet", "cell_regime_summary.parquet", "cell_regime_presence_only.parquet",
    "station_cell_map.parquet", "stop_cell_map.parquet"]
PROCESSED_DIRS = ["subway_ridership", "bus_ridership_bystop", "grid", "plan", "places"]
RAW_FILES = [("Seoul Plan", "서울시 주요 121장소 목록.xlsx")]
MODEL_FILES = ["ksweep_summary.json", "hsmm_presence_only.pkl", "po_compare_labels.npz",
               "latest_input.txt", "latest_input_v2.txt"]
MODEL_DIRS = ["k4", "k5", "k6", "k7", "k8", "k9", LADDER]
MODEL_INPUT_META = ["input_0f25c5fc0a/feature_meta.json",
                    f"input_v2_{LADDER.split('_')[1]}/feature_meta.json"]
BUS_DIR = "processed/bus_ridership_bystop/"
# monthly bus-stop partitions per zip (each partition is ~4.2 MB of zstd parquet)
BUS_GROUPS = {
    "bus_2025a": ["202501", "202502", "202503", "202504"],
    "bus_2025b": ["202505", "202506", "202507", "202508"],
    "bus_2025c": ["202509", "202510", "202511", "202512"],
    # 2026 split in two: five partitions in one zip would be ~20.9 MB (> 20 MB)
    "bus_2026a": ["202601", "202602", "202603"],
    "bus_2026b": ["202604", "202605"],
}
STATS_FILES = ["regime_diurnal.json", "regime_admin_discordance.json",
               "regime_admin_table.parquet", "presence_only_meta.json",
               "presence_only_diurnal.json", "cell_master_counts.json", "dwell_summary.json"]


def collect(data_dir: Path, models_dir: Path, stats_dir: Path):
    items = []                                   # (source path, path inside the zip)
    for f in PROCESSED_FILES:
        items.append((data_dir / "processed" / f, f"processed/{f}"))
    for d in PROCESSED_DIRS:
        for p in sorted((data_dir / "processed" / d).rglob("*")):
            if p.is_file():
                items.append((p, f"processed/{d}/{p.relative_to(data_dir / 'processed' / d).as_posix()}"))
    for sub, f in RAW_FILES:
        items.append((data_dir / sub / f, f"{sub}/{f}"))
    for f in MODEL_FILES:
        items.append((models_dir / f, f"models/{f}"))
    for d in MODEL_DIRS:
        for p in sorted((models_dir / d).rglob("*")):
            if p.is_file():
                items.append((p, f"models/{d}/{p.relative_to(models_dir / d).as_posix()}"))
    for f in MODEL_INPUT_META:
        items.append((models_dir / f, f"models/{f}"))
    for f in STATS_FILES:
        items.append((stats_dir / f, f"stats/{f}"))
    missing = [str(s) for s, _ in items if not s.exists()]
    if missing:
        raise SystemExit("missing source files:\n  " + "\n  ".join(missing))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path,
                    default=Path(os.environ.get("DFR_DATA_DIR", REPO / "data")))
    ap.add_argument("--models-dir", type=Path,
                    default=Path(os.environ.get("DFR_MODELS_DIR", REPO / "results" / "models")))
    ap.add_argument("--stats-dir", type=Path, default=REPO / "results" / "stats")
    ap.add_argument("--out-dir", type=Path, default=REPO.parent / "dwell-functional-regimes-zenodo")
    a = ap.parse_args()

    readme = Path(__file__).with_name("README_data.md")
    items = [(readme, "README_data.md")] + collect(a.data_dir, a.models_dir, a.stats_dir)
    parts = split(items)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    sums = []
    for part, members in parts.items():
        zpath = a.out_dir / f"{PKG}_{part}.zip"
        with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for src, arc in members:
                zf.write(src, f"{PKG}/{arc}")
        lines = [f"{s.stat().st_size}\t{PKG}/{arc}" for s, arc in members]
        (a.out_dir / f"{zpath.stem}_files.tsv").write_text(
            "bytes\tpath\n" + "\n".join(lines) + "\n", encoding="utf-8", newline="\n")
        sums.append(f"{sha256(zpath)}  {zpath.name}")
        unc = sum(s.stat().st_size for s, _ in members)
        print(f"{zpath.name}: {len(members)} files, uncompressed {unc / 1e6:.2f} MB, "
              f"zip {zpath.stat().st_size / 1e6:.2f} MB", flush=True)
    (a.out_dir / "SHA256SUMS.txt").write_text(    # LF line endings: `sha256sum -c` compatible
        "\n".join(sums) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(items)} files in {len(parts)} zips -> {a.out_dir}")


def split(items):
    """Assign every (source, path-in-package) to exactly one zip; bus partitions by month."""
    month_part = {ym: g for g, yms in BUS_GROUPS.items() for ym in yms}
    parts = {"core": []}
    parts.update({g: [] for g in BUS_GROUPS})
    for src, arc in items:
        if arc.startswith(BUS_DIR):
            ym = arc[len(BUS_DIR):].split("/")[0].removeprefix("year_month=")
            if ym not in month_part:
                raise SystemExit(f"bus partition {ym} is not assigned to a zip (BUS_GROUPS)")
            parts[month_part[ym]].append((src, arc))
        else:
            parts["core"].append((src, arc))
    return parts


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    main()
