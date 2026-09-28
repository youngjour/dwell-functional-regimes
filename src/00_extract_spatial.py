"""Step 00 - unzip the spatial source files into data/processed/ (file preparation only).

  Seoul Data/서울체류인구_250m격자정보_EPSG5179.zip -> processed/grid/match/match.shp
  Seoul Plan/UPIS_SHP_ZON100.zip                     -> processed/plan/ZON100/
  Seoul Plan/UPIS_SHP_ZON500.zip                     -> processed/plan/ZON500/
  Seoul Plan/서울시 주요 121장소 영역.zip             -> processed/places/서울시 주요 121장소 영역/

The 250 m grid shapefile is also distributed as 서울시_250m격자.zip (identical file,
same md5); either name is accepted.

Run: python src/00_extract_spatial.py
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

GRID_ZIPS = ["서울체류인구_250m격자정보_EPSG5179.zip", "서울시_250m격자.zip"]


def extract(zip_path: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    print(f"extracted {zip_path.name} -> {dest}", flush=True)


def main():
    grid = next((C.RAW_DIR / z for z in GRID_ZIPS if (C.RAW_DIR / z).exists()), None)
    if grid is None:
        sys.exit(f"grid zip not found in {C.RAW_DIR} (expected one of {GRID_ZIPS})")
    extract(grid, C.PROCESSED_DIR / "grid")                       # contains match/
    for zon in ["ZON100", "ZON500"]:
        extract(C.PLAN_RAW_DIR / f"UPIS_SHP_{zon}.zip", C.PROCESSED_DIR / "plan" / zon)
    extract(C.PLAN_RAW_DIR / "서울시 주요 121장소 영역.zip", C.PROCESSED_DIR / "places")
    print("DONE")


if __name__ == "__main__":
    main()
