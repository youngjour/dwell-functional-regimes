"""Step 18 - robustness of the conclusions. No model/decode rerun.

1. Area-aware purity: size-matched contiguous-patch null + spatial-shift null for
   admin-dong / district living zone (지역생활권) / wide-area living zone (권역생활권).
2. Activity-core threshold sensitivity (absolute / relative / categorical).
3. Emergent (active-not-planned) clusters + gu / admin-dong ranking.
   Each cluster gets cluster_id (rank by size desc), its cell_ids, and the designated
   centers it touches (8-neighbour cell pairs cluster-cell <-> center-cell).
4. Density-controlled partial correlation for the transit alignment.

Outputs: data/processed/emergent_activity_clusters.parquet,
results/stats/conclusions_robustness_stats.json.
"""
from __future__ import annotations

import json
import sys
from collections import deque
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

SEED = C.SEED
ACT = ["activity_A", "activity_B"]
OFFS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
N_PATCH = 40
N_SHIFT = 150


def build_adj(ix, iy):
    pos = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(ix, iy))}
    nbrs = [[] for _ in range(len(ix))]
    for i, (a, b) in enumerate(zip(ix, iy)):
        for da, db in OFFS:
            j = pos.get((int(a) + da, int(b) + db))
            if j is not None:
                nbrs[i].append(j)
    return nbrs, pos


def region_grow(seed, size, nbrs, rng):
    seen = {seed}; frontier = list(nbrs[seed])
    rng.shuffle(frontier)
    fr = deque(frontier)
    while len(seen) < size and fr:
        j = fr.popleft()
        if j in seen:
            continue
        seen.add(j)
        nb = [k for k in nbrs[j] if k not in seen]
        rng.shuffle(nb)
        fr.extend(nb)
    return seen


def purity_of(idx, dom, isf):
    sub = [dom[i] for i in idx if isf[i]]
    if not sub:
        return None
    vals, cnts = np.unique(sub, return_counts=True)
    return cnts.max() / len(sub)


def comp1_area_purity(reg, nbrs, dom, isf):
    rng = np.random.default_rng(SEED)
    n = len(dom)
    out = {}
    units = {"행정동": reg["admin_dong"].to_list(),
             "지역생활권": reg["sg_local"].to_list(),
             "권역생활권": reg["sg_region"].to_list()}
    # spatial-shift null setup
    ix = reg["grid_ix"].to_numpy(); iy = reg["grid_iy"].to_numpy()
    nx, ny = ix.max() + 1, iy.max() + 1
    grid = np.full((nx, ny), "", dtype=object); grid[ix, iy] = np.array(dom, dtype=object)
    curve = {"n_cells": [], "purity": [], "tier": []}
    for tier, labels in units.items():
        # observed per-unit purity + size
        by = {}
        for i, u in enumerate(labels):
            if u is None:
                continue
            by.setdefault(u, []).append(i)
        obs_pur, sizes, patch_null, patch_z = [], [], [], []
        for u, idx in by.items():
            p = purity_of(idx, dom, isf)
            if p is None:
                continue
            m = len(idx)
            obs_pur.append(p); sizes.append(m)
            curve["n_cells"].append(m); curve["purity"].append(p); curve["tier"].append(tier)
            # size-matched contiguous patches
            ps = []
            for _ in range(N_PATCH):
                s = int(rng.integers(0, n))
                patch = region_grow(s, m, nbrs, rng)
                pv = purity_of(patch, dom, isf)
                if pv is not None:
                    ps.append(pv)
            if ps:
                patch_null.append(np.mean(ps))
                patch_z.append((p - np.mean(ps)) / (np.std(ps) + 1e-9))
        # spatial-shift null (toroidal)
        shift_pur = []
        for _ in range(N_SHIFT):
            dx = int(rng.integers(6, 30)) * rng.choice([-1, 1])
            dy = int(rng.integers(6, 30)) * rng.choice([-1, 1])
            lab = np.roll(np.roll(grid, dx, 0), dy, 1)[ix, iy]
            m = lab != ""
            if m.sum() < 0.3 * n:
                continue
            d2 = pl.DataFrame({"u": [labels[i] for i in range(n) if m[i]],
                               "d": [lab[i] for i in range(n) if m[i]],
                               "f": [isf[i] for i in range(n) if m[i]]}).filter(
                pl.col("u").is_not_null() & pl.col("f"))
            per = d2.group_by("u").agg(
                pl.col("d").value_counts().struct.field("count").max().alias("mx"),
                pl.len().alias("t"))
            shift_pur.append(float((per["mx"] / per["t"]).mean()))
        out[tier] = {
            "n_units": len(obs_pur),
            "obs_mean_purity": round(float(np.mean(obs_pur)), 3),
            "mean_cells_per_unit": round(float(np.mean(sizes)), 1),
            "patch_null_mean_purity": round(float(np.mean(patch_null)), 3),
            "patch_z_mean": round(float(np.mean(patch_z)), 2),
            "obs_below_patch_null": bool(np.mean(obs_pur) < np.mean(patch_null)),
            "shift_null_mean_purity": round(float(np.mean(shift_pur)), 3) if shift_pur else None}
    return out, curve


def comp2_threshold(cm):
    seoul_mean = float(cm["activity_share"].mean())
    q75 = float(cm["activity_share"].quantile(0.75))
    q90 = float(cm["activity_share"].quantile(0.90))
    defs = {
        "abs_0.3": pl.col("activity_share") >= 0.3,
        "abs_0.4": pl.col("activity_share") >= 0.4,
        "abs_0.5": pl.col("activity_share") >= 0.5,
        "abs_0.6": pl.col("activity_share") >= 0.6,
        f"seoul_mean({seoul_mean:.2f})": pl.col("activity_share") >= seoul_mean,
        "top_quartile": pl.col("activity_share") >= q75,
        "top_decile": pl.col("activity_share") >= q90,
        "dominant_actAB": pl.col("dominant_functional").is_in(ACT)}
    rows = []
    for name, expr in defs.items():
        core = cm.with_columns(expr.alias("is_core"))
        ncore = core.filter(pl.col("is_core"))
        out = ncore.filter(pl.col("is_core") & pl.col("jungsim").is_null()).height
        anp = round(100 * out / ncore.height, 1) if ncore.height else None
        # center active% per tier: center active if mean activity_share of cells >= same abs thr,
        # else fraction core cells > 0.5
        thr = None
        if name.startswith("abs_"):
            thr = float(name.split("_")[1])
        elif name.startswith("seoul"):
            thr = seoul_mean
        per = core.filter(pl.col("jungsim").is_not_null()).group_by(["jungsim", "jungsim_tier"]).agg(
            pl.col("activity_share").mean().alias("ma"),
            pl.col("is_core").mean().alias("fc"))
        if thr is not None:
            per = per.with_columns((pl.col("ma") >= thr).alias("act"))
        else:
            per = per.with_columns((pl.col("fc") > 0.5).alias("act"))
        tieract = {}
        for t in ["도심", "광역중심", "지역중심", "지구중심"]:
            sub = per.filter(pl.col("jungsim_tier") == t)
            tieract[t] = round(100 * sub["act"].sum() / sub.height, 0) if sub.height else None
        rows.append({"def": name, "n_core_cells": ncore.height,
                     "active_not_planned_pct": anp, "tier_active_pct": tieract})
    return rows


def comp3_emergent(cm, reg, nbrs):
    act_out_mask = cm["dominant_functional"].is_in(ACT) & cm["jungsim"].is_null()
    em_idx = [i for i, v in enumerate(act_out_mask.to_list()) if v]
    em_set = set(em_idx)
    admin = reg["admin_dong"].to_list()
    gu = [a[:5] if a else None for a in admin]
    cm = cm.with_columns(pl.Series("gu", gu))
    em = cm.filter(act_out_mask)
    gu_rank = em.group_by("gu").len().sort("len", descending=True).head(15)
    dong_rank = em.group_by("admin_dong").len().sort("len", descending=True).head(15)
    # connected components among emergent cells
    seen = set(); comps = []
    for s in em_idx:
        if s in seen:
            continue
        comp = []; dq = deque([s]); seen.add(s)
        while dq:
            j = dq.popleft(); comp.append(j)
            for k in nbrs[j]:
                if k in em_set and k not in seen:
                    seen.add(k); dq.append(k)
        comps.append(comp)
    js_name = cm["jungsim"].to_list(); js_tier = cm["jungsim_tier"].to_list()
    cell_ids = reg["cell_id"].to_list()
    rows = []
    actsh = reg["activity_share"].to_numpy() if "activity_share" in reg.columns else \
        (reg["share_activity_A"] + reg["share_activity_B"]).to_numpy()
    ix = reg["grid_ix"].to_numpy(); iy = reg["grid_iy"].to_numpy()
    for c in comps:
        if len(c) < 5:
            continue
        dns = [admin[i] for i in c]
        modal = max(sorted(set(dns)), key=dns.count)    # ties -> smallest code (deterministic)
        # adjacency to designated centers: 8-neighbour (cluster cell, center cell) pairs
        pairs = {}
        for i in c:
            for k in nbrs[i]:
                if js_name[k] is not None:
                    key = (js_name[k], js_tier[k])
                    pairs[key] = pairs.get(key, 0) + 1
        adj = sorted(pairs.items(), key=lambda kv: (-kv[1], kv[0][0]))
        rows.append({"n_cells": len(c), "dominant_admin_dong": modal,
                     "gu": modal[:5], "mean_activity_share": round(float(np.mean([actsh[i] for i in c])), 3),
                     "cx": float(np.mean([ix[i] for i in c]) * 250 + 936375),
                     "cy": float(np.mean([iy[i] for i in c]) * 250 + 1937125),
                     "cell_ids": sorted(cell_ids[i] for i in c),
                     "adj_centers": [n for (n, _), _ in adj],
                     "adj_center_tiers": [t for (_, t), _ in adj],
                     "adj_pair_counts": [v for _, v in adj],
                     "n_adj_pairs": int(sum(pairs.values())),
                     "touches_center": bool(pairs)})
    rows.sort(key=lambda r: -r["n_cells"])     # stable: ties keep discovery order
    for rank, r in enumerate(rows, 1):
        r["cluster_id"] = rank
    return gu_rank, dong_rank, rows, em_idx


def partial_r(x, y, z):
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[m], y[m], z[m]
    rxy = np.corrcoef(x, y)[0, 1]; rxz = np.corrcoef(x, z)[0, 1]; ryz = np.corrcoef(y, z)[0, 1]
    pr = (rxy - rxz * ryz) / np.sqrt((1 - rxz**2) * (1 - ryz**2))
    n = m.sum(); t = pr * np.sqrt((n - 3) / (1 - pr**2))
    from scipy import stats
    p = 2 * stats.t.sf(abs(t), n - 3)
    return {"raw_r": round(float(rxy), 3), "partial_r": round(float(pr), 3),
            "p": float(p), "n": int(n)}


def comp4_density(reg):
    tv = pl.read_parquet(C.PROCESSED_DIR / "transit_validation.parquet")
    cmst = pl.read_parquet(C.PROCESSED_DIR / "cell_master.parquet").filter(pl.col("in_universe"))
    ix = np.round((cmst["utm_x"].to_numpy() - 936375.0) / 250).astype(int)
    iy = np.round((cmst["utm_y"].to_numpy() - 1937125.0) / 250).astype(int)
    dens = {(int(a), int(b)): float(v) for a, b, v in
            zip(ix, iy, cmst["mean_total_pop"].to_list())}
    out = {}
    for k in [1, 2]:
        gix = tv["grid_ix"].to_numpy(); giy = tv["grid_iy"].to_numpy()
        cd = []
        for a, b in zip(gix, giy):
            vals = [dens[(int(a) + da, int(b) + db)] for da in range(-k, k + 1)
                    for db in range(-k, k + 1) if (int(a) + da, int(b) + db) in dens]
            cd.append(np.log1p(np.mean(vals)) if vals else np.nan)
        cd = np.array(cd)
        rs = tv["res_score"].to_numpy()
        out[f"k{k}"] = {
            "res_vs_residential": partial_r(rs, tv[f"cat_resid_k{k}"].to_numpy(), cd),
            "res_vs_activity": partial_r(rs, tv[f"cat_act_k{k}"].to_numpy(), cd)}
    return out


def main():
    reg = pl.read_parquet(C.PROCESSED_DIR / "cell_regime_summary.parquet").with_columns(
        (pl.col("share_activity_A") + pl.col("share_activity_B")).alias("activity_share"))
    rob = pl.read_parquet(C.PROCESSED_DIR / "regime_robustness.parquet").select(
        ["cell_id", "grid_ix", "grid_iy"])
    plan = pl.read_parquet(C.PROCESSED_DIR / "cell_plan_map.parquet").select(
        ["cell_id", "sg_local", "sg_region", "jungsim", "jungsim_tier"])
    reg = reg.join(rob, on="cell_id").join(plan, on="cell_id")
    nbrs, pos = build_adj(reg["grid_ix"].to_numpy(), reg["grid_iy"].to_numpy())
    dom = reg["dominant_functional"].to_list()
    isf = [d not in ("structural_missing", "none") for d in dom]
    stats = {}

    print("comp1 area-purity...", flush=True)
    c1, curve = comp1_area_purity(reg, nbrs, dom, isf)
    stats["area_aware_purity"] = c1
    print("comp2 thresholds...", flush=True)
    stats["threshold_sensitivity"] = comp2_threshold(reg)
    print("comp3 emergent...", flush=True)
    gu_rank, dong_rank, clusters, em_idx = comp3_emergent(reg, reg, nbrs)
    n_touch = sum(1 for r in clusters if r["touches_center"])
    stats["emergent"] = {"n_emergent_cells": len(em_idx),
                         "gu_rank": gu_rank.to_dicts(), "dong_rank": dong_rank.to_dicts(),
                         "n_clusters_ge5": len(clusters),
                         "n_clusters_touching_center": n_touch,
                         "pct_clusters_touching_center": round(100 * n_touch / len(clusters), 1),
                         "clusters": [{k: v for k, v in r.items() if k != "cell_ids"}
                                      for r in clusters]}
    cols = ["cluster_id"] + [k for k in clusters[0] if k != "cluster_id"]
    pl.DataFrame([{k: r[k] for k in cols} for r in clusters]).write_parquet(
        C.PROCESSED_DIR / "emergent_activity_clusters.parquet")
    print("comp4 partial corr...", flush=True)
    stats["density_partial_corr"] = comp4_density(reg)

    (C.STATS_DIR / "conclusions_robustness_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print("DONE")


if __name__ == "__main__":
    main()
