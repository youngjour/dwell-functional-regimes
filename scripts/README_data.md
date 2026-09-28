# Derived data: Urban functional regimes from presence and dwell time (Seoul)

Derived data for the manuscript *"Urban functional regimes from presence and dwell time
in mobile network data and their alignment with planned centers in Seoul"* (manuscript
submitted). Code: <https://github.com/youngjour/dwell-functional-regimes>.
Zenodo record: DOI [10.5281/zenodo.23006739](https://doi.org/10.5281/zenodo.23006739).

The package holds the minimum set of files needed to rerun the post-model steps of the
pipeline (steps 12 and 16-23: spatial robustness, transit validation, plan discordance,
robustness and emergent clusters, 121-place validation, presence-only vs dwell-aware
comparison, manuscript tables and figures) **without** the ~20 GB raw mobile-network
tables and **without** refitting the models.

## How to use

```bash
unzip dfr_derived_data.zip                       # -> dfr_derived_data/
export DFR_DATA_DIR=/path/to/dfr_derived_data
export DFR_MODELS_DIR=/path/to/dfr_derived_data/models
cp dfr_derived_data/stats/*  <repo>/results/stats/   # only if missing in the repo
cd <repo>
for s in 12 16 17 18 19 21 22 23; do python src/${s}_*.py; done
```

Steps 00-11, 13-15 and 20 need the raw data (see the repository README).

## Layout and files

All spatial coordinates are EPSG:5179 (Korea 2000 / Unified CS) unless stated.
Grid cells are the 250 m cells of the Seoul living-population grid; `cell_id` is the
grid's `CELL_ID`. The analysis universe is 6,357 cells. Regime names:
`residential`, `activity_A` (daytime activity), `activity_B` (all-day activity),
`dense_mixed`, `mid_density`, `low_density`, `structural_missing` (state dominated by
masked / missing dwell records).

### `processed/` (derived tables; created by the pipeline)

| file | produced by | content |
|---|---|---|
| `cell_master.parquet` | step 02 | 8,880 cells seen in both products. `cell_id`, `mean_total_pop` (full-period mean living population per cell-hour), `n_nonnull_obs`, `n_rows`, `admin_dong` (modal 8-digit administrative-dong code), `utm_x`/`utm_y` (cell centre), `in_stay`, `active_25/50/100` (mean_total_pop >= threshold), `in_universe` (analysis universe = both products AND active_50) |
| `cell_regime_summary.parquet` | step 11 | dwell-aware HSMM (K=7) regime summary per universe cell, training period 2025.01-2026.03. `n_celldays_train` (number of decoded training labels, i.e. cell-days x 19 time bins); `dominant_all` (modal state over all bins), `dominant_day` (10:00-16:00 bins), `dominant_night` (00:00-05:59, 22:00, 23:00 bins), `dominant_functional` (modal non-structural regime), `func_entropy` (normalised entropy of the functional-regime shares), `struct_bin_ratio` (share of bins in the structural state), `dominant_all_ho` (modal state in the 2026.04-05 holdout), `share_<regime>` (share of functional bins per regime) |
| `cell_regime_presence_only.parquet` | step 20 | same for the presence-only HSMM. `po_dominant`, `po_dominant_day`, `po_dominant_night` (state index 0-6), `po_res_share` / `po_act_share` (share of residential-type / activity-type states), `po_share_s0..s6` |
| `station_cell_map.parquet` | step 15 | subway station -> cell. `station_name` (ridership name), `nkey` (normalised name), `st_name` (station-master name), `lat`/`lon` (WGS84), `utm_x`/`utm_y`, `cell_id` (null if outside the universe), `grid_ix`/`grid_iy` (grid indices, 250 m) |
| `stop_cell_map.parquet` | step 15 | bus stop -> cell (universe cells only). `stop_id`, `stop_name`, `lat`/`lon`, `utm_x`/`utm_y`, `cell_id`, `grid_ix`/`grid_iy` |
| `subway_ridership/year_month=YYYYMM/part.parquet` | step 13 | monthly subway boardings/alightings by hour. `use_mm`, `line`, `station_name`, `hour` (0-23), `board`, `alight` |
| `bus_ridership_bystop/year_month=YYYYMM/part.parquet` | step 14 | monthly bus boardings/alightings by stop and hour (summed over routes). `use_ym`, `stop_id`, `stop_name`, `hour`, `board`, `alight` |
| `grid/match/match.*` | step 00 | 250 m grid shapefile (redistributed unchanged) |
| `plan/ZON100/`, `plan/ZON500/` | step 00 | 2030 Seoul living-zone plan: living zones (ZON100) and designated centers (ZON500), redistributed unchanged. No `.prj`; the CRS is EPSG:5174 (determined empirically), dbf encoding CP949 |
| `places/서울시 주요 121장소 영역/` | step 00 | polygons of the 121 major places of Seoul (EPSG:4326, UTF-8 dbf), redistributed unchanged |

### `Seoul Plan/서울시 주요 121장소 목록.xlsx`

List of the 121 major places (category, code `AREA_CD`, name `AREA_NM`), redistributed
unchanged.

### `models/` (fitted models and model summaries)

| file | produced by | content |
|---|---|---|
| `ksweep_summary.json` | step 07 | shared-state Gaussian HMM, K = 4..9: BIC, holdout log-likelihood / perplexity, reinit ARI (Table S2, Figure `k_selection`) |
| `k4/` ... `k9/` | step 07 | K-sweep models (`model.pkl`, hmmlearn) and inverse-transformed emission tables |
| `ladder_1455c0d960/ladder_summary.json` | steps 09-10 | four-model ladder at K=7 (HMM, sticky HMM, HSMM, LDA; plus HMM K=6): BIC, holdout perplexity (Apr / May split), reinit ARI, temporal coherence, duration KL (Table S1, Figure `model_ladder`) |
| `ladder_1455c0d960/HSMM.pkl`, `HSMM_dur_pmf.npy`, `HSMM_emission.csv` | step 09 | adopted dwell-aware HSMM (K=7): parameters, duration pmf, emission profile per state (Table 1) |
| `ladder_1455c0d960/HMM_*.pkl`, `Sticky_*.pkl`, `*_emission.csv` | step 09 | other ladder models |
| `ladder_1455c0d960/viterbi_labels.npz` | step 10 | Viterbi labels (50,000 training day-sequences x 19 time bins) per ladder model |
| `hsmm_presence_only.pkl` | step 20 | presence-only HSMM (K=7) |
| `po_compare_labels.npz` | step 20 | `po` and `dwell`: labels of the presence-only and dwell-aware HSMMs on the same 50,000 x 19 training sequences |
| `latest_input*.txt`, `input_*/feature_meta.json` | steps 06, 08 | model-input identifiers and feature-transform parameters (CLR, z-score) |

The pickles were written with the pinned versions in the repository's `requirements.txt`
(Python 3.13, hmmlearn 0.3.3, scikit-learn 1.6.1, numpy 2.2.5).

### `stats/` (JSON outputs of steps that are not rerun from this package)

| file | produced by | content |
|---|---|---|
| `regime_diurnal.json` | step 11 | P(regime given time bin), training and holdout (Figure `regime_diurnal`, Table 1) |
| `regime_admin_discordance.json`, `regime_admin_table.parquet` | step 11 | regimes within administrative dongs (share of dongs with more than one dominant regime) |
| `presence_only_meta.json`, `presence_only_diurnal.json` | step 20 | presence-only model: feature scaling, residential / activity state assignment, P(state given time bin) |
| `cell_master_counts.json` | step 02 | cell counts behind the analysis universe |
| `dwell_summary.json` | step 03 | population-weighted dwell-duration shares and start-time x dwell counts (Figure `dwell_distribution`) |

## Sources and licences

Original data, Seoul Open Data Plaza (<https://data.seoul.go.kr>), used under the
**Korea Open Government License (KOGL) Type 1: Attribution**:

- OA-22784 Seoul living population, 250 m grid (서울 생활인구, 250m 격자)
- OA-22892 Seoul stay population, 250 m grid (서울 체류인구, 250m 격자)
- OA-12252 Subway ridership by station and hour (지하철 호선별 역별 시간대별 승하차 인원)
- OA-12913 Bus ridership by route, stop and hour (버스노선별 정류장별 시간대별 승하차 인원)
- 2030 Seoul living-zone plan spatial data (생활권계획 공간자료: ZON100, ZON500)
- Seoul 121 major places (서울시 주요 121장소)
- Seoul station master and bus-stop master (서울시 역사마스터 정보, 서울시 정류장마스터 정보)

Source: Seoul Metropolitan Government, Seoul Open Data Plaza.

This derived-data package is released under **CC BY 4.0**
(<https://creativecommons.org/licenses/by/4.0/>). When reusing it, please cite the
manuscript and the Zenodo record (DOI 10.5281/zenodo.23006739), and attribute the original data to the Seoul
Metropolitan Government (Seoul Open Data Plaza, KOGL Type 1).
