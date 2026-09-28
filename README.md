# Urban functional regimes from presence and dwell time in mobile network data and their alignment with planned centers in Seoul

Code to reproduce the results of the manuscript "Urban functional regimes from presence
and dwell time in mobile network data and their alignment with planned centers in Seoul".

The pipeline builds a 250 m cell x day x time-bin emission table from Seoul's public
living-population (presence) and stay-population (dwell time) data, fits a dwell-aware
hidden semi-Markov model (HSMM, K = 7) and a model ladder, decodes functional regimes for
6,357 cells, validates them against transit-card ridership and 121 official places,
and compares them with the designated centers of the 2030 Seoul living-zone plan.

## Contents

```
src/        pipeline steps 00-23 (run in numerical order) + _edhsmm.py (HSMM implementation)
scripts/    package_derived_data.py + README_data.md (Zenodo derived-data package)
results/
  stats/    numeric outputs (JSON) of every step
  tables/   manuscript tables (.tex bodies, .csv)
  figures/  manuscript figures (.png + .pdf, 300 dpi)
```

`results/` holds the outputs of the verification run described below. JSON outputs of the
model steps that are not rerun in path (b) (`regime_diurnal.json`,
`regime_admin_discordance.json`, `regime_admin_table.parquet`, `presence_only_*.json`,
`cell_master_counts.json`) are the outputs of the original run.

## Setup

Python 3.13.

```bash
pip install -r requirements.txt
```

Paths are set in `src/config.py` and can be redirected with environment variables:

| variable | default | meaning |
|---|---|---|
| `DFR_DATA_DIR` | `./data` | data tree (raw + processed) |
| `DFR_MODELS_DIR` | `./results/models` | fitted models and model summaries |
| `SEOUL_OPENAPI_KEY` | none | Seoul Open Data Plaza API key (step 13 only); may also be put in a git-ignored `.env` file as `SEOUL_OPENAPI_KEY=...` |

## Data

All original data are public on the Seoul Open Data Plaza (<https://data.seoul.go.kr>)
under the **Korea Open Government License Type 1 (Attribution)**. Source: Seoul
Metropolitan Government.

| dataset | where to put it (`$DFR_DATA_DIR/...`) |
|---|---|
| OA-22784 Seoul living population, 250 m grid (17 monthly zips, 2025.01-2026.05) | `Seoul Data/250_LOCAL_RESD_YYYYMM.zip` |
| OA-22892 Seoul stay population, 250 m grid (17 monthly zips) | `Seoul Data/SEOUL_STYTIME_04_250M_OPEN_NATIVE_YYYYMM.zip` |
| 250 m grid shapefile (distributed with OA-22892) | `Seoul Data/서울체류인구_250m격자정보_EPSG5179.zip` (or `서울시_250m격자.zip`, identical) |
| OA-12252 subway ridership by station and hour | downloaded by step 13 through the Open API (`CardSubwayTime`); needs `SEOUL_OPENAPI_KEY` |
| OA-12913 bus ridership by route, stop and hour (17 monthly CSVs) | `Seoul Transport/YYYY년_버스노선별_정류장별_시간대별_승하차_인원_정보(MM월).csv` |
| Seoul station master / bus-stop master | `Seoul Transport/서울시 역사마스터 정보.csv`, `Seoul Transport/서울시 정류장마스터 정보.csv` |
| 2030 Seoul living-zone plan spatial data: living zones (ZON100), designated centers (ZON500) | `Seoul Plan/UPIS_SHP_ZON100.zip`, `Seoul Plan/UPIS_SHP_ZON500.zip` |
| Seoul 121 major places (polygons + list) | `Seoul Plan/서울시 주요 121장소 영역.zip`, `Seoul Plan/서울시 주요 121장소 목록.xlsx` |

File names are those of the portal downloads; the scripts expect them as listed.

### Derived data (Zenodo)

Zenodo record: DOI `TBD`. The package `dfr_derived_data.zip` (~78 MB) holds the decoded
regime summaries, station/stop-cell maps, ridership tables, spatial reference files, the
fitted models and model summaries needed to rerun the post-model steps (path b). It is
released under **CC BY 4.0**; the original data are attributed to the Seoul Metropolitan
Government (Seoul Open Data Plaza, KOGL Type 1). File-by-file description:
[`scripts/README_data.md`](scripts/README_data.md). The package is built with
`python scripts/package_derived_data.py`.

## Pipeline

Run the steps in order (`python src/NN_name.py`). Runtimes are wall-clock on a Windows
desktop; the heavy steps were not timed precisely.

| step | script | input | output | cost |
|---|---|---|---|---|
| 00 | `00_extract_spatial.py` | grid, ZON100/ZON500, 121-place zips | `processed/grid`, `processed/plan`, `processed/places` | seconds |
| 01 | `01_ingest.py` | living / stay zips (CP949 CSV) | `processed/living_population/`, `processed/stay_population/` (hive parquet, ~11 GB) | heavy (streams 17 months x 2 products) |
| 02 | `02_cell_master.py` | living/stay parquet, grid | `processed/cell_master.parquet`, `stats/cell_master_counts.json` | minutes |
| 03 | `03_dwell_summary.py` | stay parquet (4 sample months) | `stats/dwell_summary.json` | seconds |
| 04 | `04_build_emission.py` | living/stay parquet, cell_master | `processed/emission/`, `processed/emission_presence_only/` (~8.6 GB) | heavy |
| 05 | `05_scaler.py` | emission | `processed/scaler.json` | minutes |
| 06 | `06_build_input_ksweep.py` | emission, scaler | `models/input_<hash>/` | minutes |
| 07 | `07_hmm_ksweep.py` | model input | `models/k4..k9/`, `models/ksweep_summary.json` | **heavy** (6 K x 5 reinits) |
| 08 | `08_build_input_ladder.py` | emission, scaler | `models/input_v2_<hash>/` (N x 19 x 26 tensor) | minutes |
| 09 | `09_fit_ladder.py` | input v2 | `models/ladder_<hash>/` (HMM, sticky HMM, HSMM, LDA) | **heavy** (35+ fits; one HSMM fit ~2 min) |
| 10 | `10_finalize_ladder.py` | ladder models | `ladder_summary.json`, `viterbi_labels.npz` | tens of minutes |
| 11 | `11_decode_regimes.py` | emission, HSMM | `processed/cell_regime_summary.parquet`, `stats/regime_diurnal.json`, `stats/regime_admin_discordance.json` | **heavy** (Viterbi over the full universe) |
| 12 | `12_regime_robustness.py` | cell_regime_summary | `processed/regime_robustness.parquet`, `stats/regime_robustness_stats.json` | ~7 s |
| 13 | `13_ingest_subway.py` | Open API | `processed/subway_ridership/` | minutes (network) |
| 14 | `14_ingest_bus.py` | bus CSVs | `processed/bus_ridership/`, `processed/bus_ridership_bystop/` | minutes |
| 15 | `15_geocode_join.py` | ridership, station/stop masters, cell_regime_summary | `processed/station_cell_map.parquet`, `processed/stop_cell_map.parquet`, `stats/geocode_report.json` | ~5 s |
| 16 | `16_transit_validation.py` | ridership, maps, cell_regime_summary | `processed/transit_validation.parquet`, `stats/validation_stats.json` | ~6 s |
| 17 | `17_plan_discordance.py` | cell_regime_summary, ZON100/ZON500 | `processed/cell_plan_map.parquet`, `stats/plan_discordance_stats.json` | ~5 s |
| 18 | `18_conclusions_robustness.py` | cell_regime_summary, regime_robustness, cell_plan_map, transit_validation, cell_master | `processed/emergent_activity_clusters.parquet`, `stats/conclusions_robustness_stats.json` | ~8 s |
| 19 | `19_place_validation.py` | cell_regime_summary, cell_plan_map, clusters, 121 places | `processed/place_cell_map.parquet`, `stats/place_validation_stats.json` | ~5 s |
| 20 | `20_presence_only.py` | input v2, emission | `models/hsmm_presence_only.pkl`, `models/po_compare_labels.npz`, `processed/cell_regime_presence_only.parquet`, `stats/presence_only_*.json` | **heavy** (HSMM fit + full decode) |
| 21 | `21_presence_vs_dwell.py` | outputs of 11, 16, 20 | `stats/presence_vs_dwell_stats.json` | ~6 s |
| 22 | `22_paper_tables.py` | outputs of 09-11, 17-19 | `results/tables/*` | ~5 s |
| 23 | `23_figures.py` | outputs of 03, 07, 09-11, 16-21 | `results/figures/*` | ~16 s |

`processed/...` is under `$DFR_DATA_DIR`, `models/...` under `$DFR_MODELS_DIR`,
`stats/...` is `results/stats/`. The raw and processed data need roughly 20 GB of disk;
the ingest and emission steps stream one month at a time.

## Reproducing the results

**(a) Full run from the original data.** Place the raw files as listed above, set
`SEOUL_OPENAPI_KEY`, and run steps 00-23 in order. Steps 01, 04, 07, 09-11 and 20 are
computationally heavy.

**(b) Post-model steps from the Zenodo derived data.** No raw mobile-network data and no
model fitting:

```bash
unzip dfr_derived_data.zip -d /path/to            # -> /path/to/dfr_derived_data/
export DFR_DATA_DIR=/path/to/dfr_derived_data
export DFR_MODELS_DIR=/path/to/dfr_derived_data/models
cp /path/to/dfr_derived_data/stats/* results/stats/   # model-step JSON (already in the repo)
for s in 12 16 17 18 19 21 22 23; do python src/${s}_*.py; done
```

This regenerates every manuscript table and figure. The verification of both paths is
summarised below.

## Paper output ↔ script

| paper output | script | file |
|---|---|---|
| Fig. model ladder | `23_figures.py` (from steps 09-10) | `results/figures/model_ladder.{png,pdf}` |
| Fig. day vs night regimes | `23_figures.py` (from step 11) | `results/figures/regime_day_night.{png,pdf}` |
| Fig. regime diurnal profile | `23_figures.py` (from step 11) | `results/figures/regime_diurnal.{png,pdf}` |
| Fig. presence vs dwell | `23_figures.py` (from step 21) | `results/figures/presence_vs_dwell.{png,pdf}` |
| Fig. external validation | `23_figures.py` (from steps 16, 19) | `results/figures/external_validation.{png,pdf}` |
| Fig. activity vs plan | `23_figures.py` (from steps 11, 17 inputs) | `results/figures/activity_vs_plan.{png,pdf}` |
| Suppl. fig. regime map | `23_figures.py` (from step 11) | `results/figures/regime_map.{png,pdf}` |
| Suppl. fig. K selection | `23_figures.py` (from step 07) | `results/figures/k_selection.{png,pdf}` |
| Suppl. fig. dwell distribution | `23_figures.py` (from step 03) | `results/figures/dwell_distribution.{png,pdf}` |
| Suppl. fig. purity vs unit size | `23_figures.py` (same null as step 18) | `results/figures/purity_vs_size.{png,pdf}` |
| Table 1 regime characteristics | `22_paper_tables.py` (from steps 09, 11) | `results/tables/T3_regime_characteristics.csv` |
| Table 2 designated centers | `22_paper_tables.py` (from step 17) | `results/tables/table2_centers.tex` (+ `T4_center_concordance.csv`) |
| Table S1 model ladder | `22_paper_tables.py` (from steps 09-10) | `results/tables/T2_model_ladder.csv` |
| Table S2 K sweep | `07_hmm_ksweep.py` | `$DFR_MODELS_DIR/ksweep_summary.json` |
| Table S3 purity vs size-matched null | `18_conclusions_robustness.py` | `results/stats/conclusions_robustness_stats.json` → `area_aware_purity` |
| Table S4 threshold sensitivity | `22_paper_tables.py` (from step 18) | `results/tables/tableS4_threshold.tex` (+ `T5_threshold_sensitivity.csv`) |
| Table S5 emergent clusters | `22_paper_tables.py` (from steps 18, 19) | `results/tables/tableS5_clusters.tex` (+ `tableS5_clusters.csv`) |
| Day-night regime switch 52% (dwell-aware) vs 24% (presence-only) | `21_presence_vs_dwell.py` | `presence_vs_dwell_stats.json` → `daynight_switch` |
| Presence-only vs dwell-aware agreement, ARI 0.38 | `21_presence_vs_dwell.py` | `presence_vs_dwell_stats.json` → `per_cell_dominant_agreement.ARI` (0.379; cell-bin level `label_agreement_cellbin.ARI` 0.375) |
| Transit validation R², r (subway, bus) | `16_transit_validation.py` | `validation_stats.json` → `subway_alignment`, `bus_alignment_cell` |
| χ², Cramér's V (station type x catchment regime) | `16_transit_validation.py` | `validation_stats.json` → `subway_typology` (see note below) |
| Density-controlled partial r | `18_conclusions_robustness.py` | `conclusions_robustness_stats.json` → `density_partial_corr` |
| Presence-only vs dwell-aware external R² | `21_presence_vs_dwell.py` | `presence_vs_dwell_stats.json` → `external_validation_delta` |
| Activity share by official hotspot category | `19_place_validation.py` | `place_validation_stats.json` → `category_regime` |
| 99% of administrative dongs contain more than one dominant regime | `11_decode_regimes.py` | `regime_admin_discordance.json` → `pct_dong_multi_regime` |
| 19% between-dong share of regime-composition variance | `12_regime_robustness.py` | `regime_robustness_stats.json` → `variance_decomp_eta2._aggregate_between_share` |
| 1,470 activity-core cells, 783 (53.3%) outside designated centers | `17_plan_discordance.py` | `plan_discordance_stats.json` → `active_not_planned` |
| 28 emergent clusters, 21 touching a designated center | `18_conclusions_robustness.py` | `conclusions_robustness_stats.json` → `emergent` |
| 7 clusters (25%) with centroid in an official hotspot | `19_place_validation.py`, `22_paper_tables.py` | `place_validation_stats.json` → `emergent_clusters_in_place_pct`; `tableS5_clusters.csv` |

`*.json` files above are in `results/stats/` unless a path is given.

**Note on the station-typology χ² and Cramér's V.** In step 16 the modal catchment
regime of a station (`cat_dom_k1`) is taken with `max(set(...), key=list.count)`; when
two regimes tie, the winner depends on Python's per-process string-hash seed. The
regression statistics are unaffected, but χ² and Cramér's V vary between runs (χ² 55.6-77.2,
V 0.307-0.362 over 14 runs; the manuscript values 65.0 / 0.332 are one such run). Set
`PYTHONHASHSEED` to make a run repeatable.

## Verification

Without refitting any model, the post-model steps were rerun with this code and compared
with the original outputs: (a) steps 00, 03, 12, 15-19 and 21-23 on the original raw and
processed data, and (b) steps 12, 16-19 and 21-23 from the Zenodo package alone. Tables `table2_centers.tex`, `tableS4_threshold.tex`,
`tableS5_clusters.tex`/`.csv`, `T2`, `T3`, `T5` are byte-identical in both paths; JSON
statistics agree to a relative difference below 1e-12, except for the ordering of tied
entries in ranked lists, the hash-seed dependence noted above, and float32 summation
noise (~3e-5 relative) in `dwell_summary.json`. Nine of the ten figures are
pixel-identical to the manuscript figures; `external_validation` differs in 0.06% of
pixels because semi-transparent points are drawn in a different row order.
`T4_center_concordance.csv` uses the activity-dominance rule of the manuscript
(activity-core share > 0.5; Regional centers 6/7).

## Citation

Park, Y. Urban functional regimes from presence and dwell time in mobile network data and
their alignment with planned centers in Seoul. Manuscript submitted.

See also [`CITATION.cff`](CITATION.cff).

## License

Code: MIT (see [`LICENSE`](LICENSE)). Derived data (Zenodo): CC BY 4.0. Original data:
Seoul Metropolitan Government, Korea Open Government License Type 1.
