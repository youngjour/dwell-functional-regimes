"""Project-wide configuration: paths, reproducibility, raw-data schema constants.

Paths can be redirected with environment variables so the raw / processed data and
the fitted models do not have to live inside the repository:

  DFR_DATA_DIR    root of the data tree (default: <repo>/data). Expected layout:
                    <DFR_DATA_DIR>/Seoul Data/       raw living / stay population zips
                    <DFR_DATA_DIR>/Seoul Plan/       planning + 121-place spatial data
                    <DFR_DATA_DIR>/Seoul Transport/  subway / bus ridership + stop masters
                    <DFR_DATA_DIR>/processed/        intermediate + derived tables
  DFR_MODELS_DIR  fitted models and model summaries (default: <repo>/results/models)
"""
from __future__ import annotations

import os
import random
from pathlib import Path

# --- Reproducibility -------------------------------------------------------
SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Fix all RNG seeds we might touch."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass


# --- Paths -----------------------------------------------------------------
# src/ -> project root
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("DFR_DATA_DIR", ROOT / "data")).resolve()
RAW_DIR = DATA_DIR / "Seoul Data"               # read-only
PLAN_RAW_DIR = DATA_DIR / "Seoul Plan"          # read-only
TRANSPORT_RAW_DIR = DATA_DIR / "Seoul Transport"  # read-only
PROCESSED_DIR = DATA_DIR / "processed"
LIVING_DIR = PROCESSED_DIR / "living_population"
STAY_DIR = PROCESSED_DIR / "stay_population"
GRID_SHP = PROCESSED_DIR / "grid" / "match" / "match.shp"  # EPSG:5179, key=CELL_ID
RESULTS_DIR = ROOT / "results"
MODELS_DIR = Path(os.environ.get("DFR_MODELS_DIR", RESULTS_DIR / "models")).resolve()
EDA_DIR = RESULTS_DIR / "eda"          # diagnostic plots written by the model-fitting steps
STATS_DIR = RESULTS_DIR / "stats"      # numeric outputs (JSON) of every analysis step
LOGS_DIR = RESULTS_DIR / "logs"        # ingest / download logs
FIG_DIR = RESULTS_DIR / "figures"      # manuscript figures (png + pdf)
TABLES_DIR = RESULTS_DIR / "tables"    # manuscript tables (tex + csv)
for _d in (EDA_DIR, STATS_DIR, LOGS_DIR, FIG_DIR, TABLES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Source file naming ----------------------------------------------------
MONTHS = [
    "202501", "202502", "202503", "202504", "202505", "202506",
    "202507", "202508", "202509", "202510", "202511", "202512",
    "202601", "202602", "202603", "202604", "202605",
]  # 17 months, 2025.01 ~ 2026.05

LIVING_ZIP = "250_LOCAL_RESD_{ym}.zip"
STAY_ZIP = "SEOUL_STYTIME_04_250M_OPEN_NATIVE_{ym}.zip"

# --- Encoding --------------------------------------------------------------
# Verified empirically: both products are CP949.
RAW_ENCODING = "cp949"

# --- Schema: living population ---------------------------------------------
# 33 columns = 5 meta + 28 (sex x age). Header verified from CSV.
LIVING_AGE_BANDS = [
    "0_9", "10_14", "15_19", "20_24", "25_29", "30_34", "35_39",
    "40_44", "45_49", "50_54", "55_59", "60_64", "65_69", "70_over",
]  # 14 bands, 5-year resolution
LIVING_SEX = ["M", "F"]  # male, female

# Output column order (after meta cols): M_0_9 ... M_70_over, F_0_9 ... F_70_over
LIVING_DEMO_COLS = [f"{s}_{b}" for s in LIVING_SEX for b in LIVING_AGE_BANDS]  # 28

# --- Schema: stay population -----------------------------------------------
# Header has 24 named cols; data rows carry 4 extra trailing empty fields (28).
STAY_AGE_BANDS = [
    "00_09", "10_19", "20_29", "30_39", "40_49", "50_59", "60_69", "70_over",
]  # 8 bands, 10-year resolution
STAY_SEX = ["M", "F"]  # male, female
STAY_DEMO_COLS = [f"{s}_{b}" for s in STAY_SEX for b in STAY_AGE_BANDS]  # 16

# STAY_MNUT_CD: 6 dwell-duration interval codes (NOT continuous minutes)
STAY_DUR_CODES = {
    "000": "0-29min",
    "030": "30-59min",
    "060": "60-119min",
    "120": "120-179min",
    "180": "180-239min",
    "240": "240min+",
}
# ST_TIMEZN_CD: 19 start-time bins (00 = 00:00~05:59 merged, 06..23 hourly)
STAY_START_TZ_CODES = ["00"] + [f"{h:02d}" for h in range(6, 24)]  # 19 codes

MASK_TOKEN = "*"  # masked when count <= 3 -> null in output
