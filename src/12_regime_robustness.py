"""Step 12 - spatial-structure & admin-discordance robustness for HSMM regimes.

Read-only on cell_regime_summary.parquet (no refit/redecode). Builds a 250m queen
grid weight, then:
  1. Moran's I (+999-perm p) for continuous regime indicators.
  2. Join-count for categorical dominant regime (same-regime adjacency vs null).
  3. Between-dong vs within-dong variance decomposition (eta^2 / ICC) of the
     6-vector functional composition.
  4. Discordance vs nulls: (a) marginal random relabel, (b) spatial toroidal shift.
  5. Per-cell label confidence margin (top1-top2 functional share) + sensitivity
     recompute excluding low-margin / structural-heavy cells.

Outputs: data/processed/regime_robustness.parquet, results/stats/regime_robustness_stats.json.
Run: python src/12_regime_robustness.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

SEED = C.SEED
STEP = 250.0
N_PERM = 999
N_NULL = 200
FUNC = ["low_density", "activity_A", "dense_mixed", "activity_B",
        "residential", "mid_density"]
SHARE_COLS = [f"share_{r}" for r in FUNC]


def build_queen(ix, iy):
    """Queen (8-neighbour) adjacency on a regular grid. Returns neighbor index lists
    and unordered edge arrays."""
    pos = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(ix, iy))}
    nbrs = [[] for _ in range(len(ix))]
    edges_i, edges_j = [], []
    offs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for i, (a, b) in enumerate(zip(ix, iy)):
        for da, db in offs:
            j = pos.get((int(a) + da, int(b) + db))
            if j is not None:
                nbrs[i].append(j)
                if i < j:
                    edges_i.append(i); edges_j.append(j)
    return nbrs, np.array(edges_i), np.array(edges_j)


def morans_i(x, Wn, has_nb, rng, n_perm=N_PERM):
    """Wn = row-standardised sparse weight; has_nb = bool mask of non-isolates."""
    x = x.astype(float); z = x - x.mean()
    den = np.sum(z * z)
    n = int(has_nb.sum()); S0 = n     # row-standardised non-isolate rows each sum to 1
    lag = Wn @ z
    I = (n / S0) * (np.sum(z * lag * has_nb) / den) if den > 0 else np.nan
    perm = np.empty(n_perm)
    for p in range(n_perm):
        zp = rng.permutation(z)
        perm[p] = (n / S0) * (np.sum(zp * (Wn @ zp) * has_nb) / den)
    pval = (np.sum(perm >= I) + 1) / (n_perm + 1)
    return float(I), float(-1.0 / (len(x) - 1)), float(pval), float(perm.mean()), float(perm.std())


def join_count(labels, ei, ej, rng, n_perm=N_PERM):
    same = labels[ei] == labels[ej]
    obs = same.mean()
    n_join = len(ei)
    perm = np.empty(n_perm)
    for p in range(n_perm):
        lp = rng.permutation(labels)
        perm[p] = (lp[ei] == lp[ej]).mean()
    z = (obs - perm.mean()) / (perm.std() + 1e-12)
    pval = (np.sum(perm >= obs) + 1) / (n_perm + 1)
    return float(obs), float(perm.mean()), float(z), float(pval)


def variance_decomp(df, cols, group="admin_dong"):
    """eta^2 per component + aggregate (SS_between/SS_total)."""
    res = {}
    tot_b = tot_t = 0.0
    g = df.group_by(group)
    grp_means = g.agg([pl.col(c).mean().alias(c) for c in cols] + [pl.len().alias("ng")])
    gm = {r[group]: r for r in grp_means.iter_rows(named=True)}
    for c in cols:
        x = df[c].to_numpy(); xbar = x.mean()
        sst = np.sum((x - xbar) ** 2)
        ssb = sum(gm[k]["ng"] * (gm[k][c] - xbar) ** 2 for k in gm)
        res[c] = round(float(ssb / sst), 4) if sst > 0 else None
        tot_b += ssb; tot_t += sst
    res["_aggregate_between_share"] = round(float(tot_b / tot_t), 4)
    return res


def discordance(dom, dong):
    """distinct dominant per dong (mean) + within-dong purity (mean)."""
    df = pl.DataFrame({"dom": dom, "dong": dong}).filter(pl.col("dom") != "none")
    per = df.group_by("dong").agg(
        pl.col("dom").n_unique().alias("nd"),
        pl.len().alias("tot"),
        pl.col("dom").value_counts().struct.field("count").max().alias("mx"))
    return (float(per["nd"].mean()),
            float((per["mx"] / per["tot"]).mean()),
            int(per.filter(pl.col("nd") >= 2).height), per.height)


def main():
    s = pl.read_parquet(C.PROCESSED_DIR / "cell_regime_summary.parquet")
    rng = np.random.default_rng(SEED)
    ix = np.round((s["utm_x"].to_numpy() - s["utm_x"].min()) / STEP).astype(int)
    iy = np.round((s["utm_y"].to_numpy() - s["utm_y"].min()) / STEP).astype(int)
    nbrs, ei, ej = build_queen(ix, iy)
    deg = np.array([len(n) for n in nbrs])
    print(f"cells={s.height} edges={len(ei)} mean_deg={deg.mean():.2f} "
          f"isolates={(deg==0).sum()}", flush=True)
    # row-standardised sparse weight for fast Moran
    import scipy.sparse as sp
    n = s.height
    A = sp.coo_matrix((np.ones(2 * len(ei)),
                       (np.concatenate([ei, ej]), np.concatenate([ej, ei]))),
                      shape=(n, n)).tocsr()
    rs = np.asarray(A.sum(1)).ravel()
    has_nb = rs > 0
    Wn = sp.diags(np.where(rs > 0, 1.0 / rs, 0.0)) @ A

    stats = {"n_cells": s.height, "n_edges": int(len(ei)),
             "mean_degree": float(deg.mean()), "n_isolates": int((deg == 0).sum())}

    # ---- per-cell margin (top1-top2 functional share) ----
    sh = s.select(SHARE_COLS).to_numpy()
    order = np.sort(sh, axis=1)
    margin = order[:, -1] - order[:, -2]
    s = s.with_columns(pl.Series("dom_margin", margin),
                       pl.Series("grid_ix", ix), pl.Series("grid_iy", iy))

    # ---- 1. Moran's I ----
    s = s.with_columns((pl.col("share_activity_A") + pl.col("share_activity_B"))
                       .alias("activity_share"))
    moran = {}
    for col in ["share_residential", "activity_share", "share_dense_mixed",
                "func_entropy", "dom_margin"]:
        I, EI, pv, pm, ps = morans_i(s[col].to_numpy(), Wn, has_nb,
                                     np.random.default_rng(SEED + 1))
        moran[col] = {"I": round(I, 4), "E_I": round(EI, 5), "p_perm": pv,
                      "perm_mean": round(pm, 4), "perm_std": round(ps, 4)}
        print(f"Moran {col}: I={I:.3f} p={pv:.4f}", flush=True)
    stats["morans_I"] = moran

    # ---- 2. join-count (dominant regime) ----
    dom = s["dominant_functional"].to_numpy()
    jc_obs, jc_null, jc_z, jc_p = join_count(dom, ei, ej,
                                             np.random.default_rng(SEED + 2))
    stats["join_count_dominant"] = {
        "same_regime_join_share_obs": round(jc_obs, 4),
        "null_mean": round(jc_null, 4), "z": round(jc_z, 1), "p_perm": jc_p}
    print(f"join-count same-share obs={jc_obs:.3f} null={jc_null:.3f} z={jc_z:.0f}",
          flush=True)

    # ---- 3. variance decomposition ----
    stats["variance_decomp_eta2"] = variance_decomp(s, SHARE_COLS)
    print("between-dong aggregate share:",
          stats["variance_decomp_eta2"]["_aggregate_between_share"], flush=True)

    # ---- 4. discordance observed + nulls ----
    dong = s["admin_dong"].to_numpy()
    obs_nd, obs_pur, obs_multi, n_dong = discordance(dom, dong)
    stats["discordance_observed"] = {
        "mean_distinct_dom": round(obs_nd, 3), "mean_purity": round(obs_pur, 3),
        "n_dong_multi": obs_multi, "n_dong": n_dong,
        "pct_multi": round(100 * obs_multi / n_dong, 1)}

    # (a) marginal relabel null
    nd_a, pur_a = [], []
    for _ in range(N_NULL):
        lp = rng.permutation(dom)
        nd, pur, _, _ = discordance(lp, dong)
        nd_a.append(nd); pur_a.append(pur)
    # (b) spatial toroidal shift null (preserves spatial autocorrelation)
    nx, ny = ix.max() + 1, iy.max() + 1
    grid = np.full((nx, ny), "", dtype=object)
    grid[ix, iy] = dom
    # moderate shifts (beyond a typical dong's ~4-5 cell extent) to decouple from
    # admin while preserving spatial autocorrelation; keep reps with >=30% overlap.
    nd_b, pur_b = [], []
    attempts = 0
    while len(pur_b) < 150 and attempts < 4000:
        attempts += 1
        dx = int(rng.integers(6, 30)) * rng.choice([-1, 1])
        dy = int(rng.integers(6, 30)) * rng.choice([-1, 1])
        rolled = np.roll(np.roll(grid, dx, 0), dy, 1)
        lab = rolled[ix, iy]
        m = lab != ""
        if m.sum() < 0.3 * len(lab):
            continue
        nd, pur, _, _ = discordance(np.array(lab)[m], dong[m])
        nd_b.append(nd); pur_b.append(pur)
    stats["discordance_null_marginal_relabel"] = {
        "mean_distinct_dom": round(float(np.mean(nd_a)), 3),
        "mean_purity": round(float(np.mean(pur_a)), 3),
        "purity_p975": round(float(np.quantile(pur_a, 0.975)), 3)}
    stats["discordance_null_spatial_shift"] = {
        "mean_distinct_dom": round(float(np.mean(nd_b)), 3),
        "mean_purity": round(float(np.mean(pur_b)), 3),
        "purity_p025": round(float(np.quantile(pur_b, 0.025)), 3),
        "purity_p975": round(float(np.quantile(pur_b, 0.975)), 3),
        "n_valid_reps": len(pur_b)}
    print(f"discordance purity obs={obs_pur:.3f} "
          f"marginal-null={np.mean(pur_a):.3f} spatial-null={np.mean(pur_b):.3f}",
          flush=True)

    # ---- 5. sensitivity: drop low-margin & structural-heavy ----
    sens = {}
    for thr in [0.0, 0.1, 0.2]:
        keep = (s["dom_margin"].to_numpy() >= thr) & \
               (s["struct_bin_ratio"].to_numpy() <= 0.5) & (dom != "none")
        nd, pur, multi, ndong = discordance(dom[keep], dong[keep])
        sens[f"margin_ge_{thr}_struct_le_0.5"] = {
            "n_cells": int(keep.sum()), "mean_distinct_dom": round(nd, 3),
            "mean_purity": round(pur, 3), "pct_multi": round(100 * multi / ndong, 1)}
    stats["sensitivity"] = sens
    stats["margin_distribution"] = {
        q: round(float(np.quantile(margin, q)), 3)
        for q in [0.1, 0.25, 0.5, 0.75, 0.9]}

    s.select(["cell_id", "utm_x", "utm_y", "admin_dong", "grid_ix", "grid_iy",
              "dominant_functional", "dom_margin", "func_entropy",
              "struct_bin_ratio", "activity_share", "share_residential"]
             ).write_parquet(C.PROCESSED_DIR / "regime_robustness.parquet")
    (C.STATS_DIR / "regime_robustness_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote regime_robustness.parquet + stats json\nDONE")


if __name__ == "__main__":
    main()
