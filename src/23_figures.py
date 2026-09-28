"""Step 23 - manuscript figures (Scientific Reports). No new analysis: re-renders the
outputs of the previous steps. png + pdf @300 dpi, English labels, colourblind-safe.

Main text                                    Supplementary
  model_ladder         <- ladder_summary.json   regime_map         <- cell_regime_summary
  regime_day_night     <- cell_regime_summary   k_selection        <- ksweep_summary.json
  regime_diurnal       <- regime_diurnal.json   dwell_distribution <- dwell_summary.json
  activity_vs_plan     <- cell_regime_summary   purity_vs_size     <- cell_regime_summary +
                          + ZON500 polygons                           regime_robustness +
  external_validation  <- transit_validation                          cell_plan_map
                          + place_validation_stats.json
  presence_vs_dwell    <- presence_vs_dwell_stats.json

Outputs: results/figures/<name>.{png,pdf}

Run: python src/23_figures.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

PUB = C.FIG_DIR
LAD = C.MODELS_DIR / "ladder_1455c0d960"
MM = 1 / 25.4
W1, W2 = 90 * MM, 180 * MM

# Main-text figures were rendered with MAIN_RC; the supplementary figures additionally
# with axes.unicode_minus=False (ASCII minus). Kept separate so the output is unchanged.
MAIN_RC = {
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9,
    "axes.labelsize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "axes.linewidth": 0.6, "lines.linewidth": 1.2,
    "savefig.dpi": 300, "figure.dpi": 150, "pdf.fonttype": 42, "ps.fonttype": 42}
SI_RC = {**MAIN_RC, "axes.unicode_minus": False}

# ---- shared regime palette (tab10 order) ----
REG_ORDER = ["residential", "activity_A", "activity_B", "dense_mixed",
             "mid_density", "low_density", "structural_missing"]
REG_COLOR = {"residential": "#1f77b4", "activity_A": "#ff7f0e", "activity_B": "#d62728",
             "dense_mixed": "#9467bd", "mid_density": "#2ca02c", "low_density": "#8c564b",
             "structural_missing": "#bbbbbb"}
REG_LABEL = {"residential": "Residential", "activity_A": "Activity A (daytime)",
             "activity_B": "Activity B (all-day)", "dense_mixed": "Dense-mixed",
             "mid_density": "Mid-density", "low_density": "Low-density",
             "structural_missing": "Structural"}
REG_LS = {"residential": "-", "activity_A": "--", "activity_B": "-.", "dense_mixed": ":",
          "mid_density": (0, (4, 1, 1, 1)), "low_density": (0, (5, 1)),
          "structural_missing": (0, (1, 1))}
REG_MK = {"residential": "o", "activity_A": "^", "activity_B": "s", "dense_mixed": "D",
          "mid_density": "v", "low_density": "P", "structural_missing": "x"}
TB = ["00"] + [f"{h:02d}" for h in range(6, 24)]
HHMM = [f"{int(t):02d}:00" for t in TB]

# designated-centre tiers (ZON510/520/530/540), darkest blue = Downtown
TIER = {"ZON510": ("Downtown", "#08519c"), "ZON520": ("Regional center", "#3182bd"),
        "ZON530": ("District center", "#6baed6"), "ZON540": ("Local center", "#bdd7e7")}

# Okabe-Ito colourblind-safe accents
OI_BLUE, OI_VERM, OI_GREEN = "#0072B2", "#D55E00", "#009E73"
OI_ORANGE, OI_SKY = "#E69F00", "#56B4E9"

DUR_ORDER = ["000", "030", "060", "120", "180", "240"]
DUR_LABEL = {"000": "0-29", "030": "30-59", "060": "60-119",
             "120": "120-179", "180": "180-239", "240": "≥240"}
TZ_ORDER = ["00"] + [f"{h:02d}" for h in range(6, 24)]
TZ_LABEL = ["00-05"] + [f"{h:02d}" for h in range(6, 24)]


def save(fig, name):
    fig.savefig(PUB / f"{name}.png", bbox_inches="tight")
    fig.savefig(PUB / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("saved", name, flush=True)


def reg_handles(counts=None):
    out = []
    for r in REG_ORDER:
        lab = REG_LABEL[r]
        if counts is not None:
            lab = f"{lab} (n={counts.get(r, 0):,})"
        out.append(mpatches.Patch(facecolor=REG_COLOR[r], label=lab))
    return out


def grid_regime():
    import geopandas as gpd
    g = gpd.read_file(C.GRID_SHP)[["CELL_ID", "geometry"]].rename(columns={"CELL_ID": "cell_id"})
    s = pl.read_parquet(C.PROCESSED_DIR / "cell_regime_summary.parquet").to_pandas()
    return g.merge(s, on="cell_id", how="inner")


def load_step(mod_name, fname):
    s = importlib.util.spec_from_file_location(mod_name, str(Path(__file__).parent / fname))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


# ================================================================ main text
def model_ladder():
    res = {r["model"]: r for r in json.loads((LAD / "ladder_summary.json").read_text("utf-8"))}
    order = ["HMM_K6", "HMM_K7", "Sticky_k1", "Sticky_k10", "Sticky_k50",
             "Sticky_k100", "HSMM", "LDA"]
    names = [m for m in order if m in res]
    short = {"HMM_K6": "HMM K6", "HMM_K7": "HMM K7", "Sticky_k1": "Sticky κ1",
             "Sticky_k10": "Sticky κ10", "Sticky_k50": "Sticky κ50",
             "Sticky_k100": "Sticky κ100", "HSMM": "HSMM", "LDA": "LDA"}
    metrics = [("ho_perplexity", "Holdout perplexity (↓)", "{:.0f}"),
               ("reinit_ari", "Reinit ARI (↑)", "{:.2f}"),
               ("dur_kl", "Duration KL (↓)", "{:.2f}")]
    fig, axes = plt.subplots(1, 3, figsize=(150 * MM, 60 * MM))
    for ax, (key, title, fmt) in zip(axes, metrics):
        # HSMM holdout perplexity not comparable (Viterbi-EM) -> bar omitted, noted in caption
        vals = [(np.nan if (key == "ho_perplexity" and m == "HSMM") else res[m].get(key))
                for m in names]
        cols = ["#08306b" if m == "HSMM" else "0.72" for m in names]   # HSMM dark navy
        bars = ax.bar(range(len(names)), [0 if v is None or np.isnan(v) else v for v in vals],
                      color=cols, edgecolor="black", linewidth=0.5)
        for b, v in zip(bars, vals):
            if v is not None and not np.isnan(v):
                ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, b.get_height()),
                            textcoords="offset points", xytext=(0, 3), ha="center",
                            va="bottom", rotation=90, fontsize=4.2)   # offset above bar
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([short[m] for m in names], rotation=40, ha="right", fontsize=6)
        ax.set_title(title, fontsize=8.5); ax.tick_params(length=2)
        ax.margins(y=0.26)
    fig.tight_layout()
    save(fig, "model_ladder")


def regime_day_night(gg):
    cday = gg["dominant_day"].value_counts().to_dict()
    cnight = gg["dominant_night"].value_counts().to_dict()
    fig, axes = plt.subplots(1, 2, figsize=(W2, 92 * MM))
    for ax, col, lab, cnt in [(axes[0], "dominant_day", "Day (10:00–16:00)", cday),
                              (axes[1], "dominant_night", "Night (22:00–23:00)", cnight)]:
        for r in REG_ORDER:
            sub = gg[gg[col] == r]
            if len(sub):
                sub.plot(ax=ax, color=REG_COLOR[r], linewidth=0)
        ax.set_title(lab, fontsize=9); ax.set_aspect("equal"); ax.axis("off")
        ax.legend(handles=reg_handles(cnt), loc="upper left", frameon=True,
                  framealpha=0.9, edgecolor="0.7", fontsize=5, handlelength=0.9,
                  handletextpad=0.4, labelspacing=0.3, borderpad=0.3)
    fig.tight_layout()
    save(fig, "regime_day_night")


def regime_diurnal():
    d = json.loads((C.STATS_DIR / "regime_diurnal.json").read_text("utf-8"))
    P = np.array(d["P_regime_given_tb_train"])
    names = {int(k): v for k, v in d["regime_names"].items()}
    fig, ax = plt.subplots(figsize=(120 * MM, 72 * MM))
    for k in range(P.shape[1]):
        nm = names[k]
        ax.plot(range(len(TB)), P[:, k], ls=REG_LS[nm], color=REG_COLOR[nm],
                marker=REG_MK[nm], ms=2.6, mfc="white", mew=0.5, lw=1.1, label=REG_LABEL[nm])
    ax.set_xticks(range(len(TB))); ax.set_xticklabels(HHMM, rotation=90, fontsize=5.5)
    ax.set_xlabel("Time of day", fontsize=7)
    ax.set_ylabel("P(regime | time bin)", fontsize=7)
    ax.tick_params(axis="y", labelsize=6); ax.tick_params(length=2)
    ax.set_ylim(0, 0.78)                       # headroom for in-plot top legend
    ax.legend(loc="upper center", ncol=4, frameon=False, fontsize=5.6,
              handlelength=1.8, columnspacing=1.0, handletextpad=0.4, borderaxespad=0.2)
    fig.tight_layout()
    save(fig, "regime_diurnal")


def activity_vs_plan(gg):
    import geopandas as gpd
    z500 = gpd.read_file(C.PROCESSED_DIR / "plan" / "ZON500" / "UPIS_SHP_ZON500.shp",
                         encoding="cp949").set_crs(5174, allow_override=True).to_crs(5179)
    fig, ax = plt.subplots(figsize=(W2, W2 * 0.92))
    gg.plot(ax=ax, color="0.80", linewidth=0, zorder=1)
    for cd, (lab, col) in TIER.items():
        sub = z500[z500["CODE"] == cd]
        if len(sub):
            sub.plot(ax=ax, facecolor=col, alpha=0.75, linewidth=0, zorder=2)
            sub.boundary.plot(ax=ax, color="black", linewidth=0.5, zorder=3)  # solid outline
    act = gg[gg["dominant_functional"].isin(["activity_A", "activity_B"])]
    ax.scatter(act.geometry.centroid.x, act.geometry.centroid.y, s=3.2,
               color="#d62728", marker="o", linewidth=0, zorder=4)
    ax.set_aspect("equal"); ax.axis("off")
    leg = [mpatches.Patch(facecolor="0.80", label="study cells"),
           mlines.Line2D([], [], marker="o", color="w", markerfacecolor="#d62728",
                         markersize=5, label="activity-core cell")]
    leg += [mpatches.Patch(facecolor=col, alpha=0.75, edgecolor="black", linewidth=0.5,
                           label=lab) for cd, (lab, col) in TIER.items()]
    ax.legend(handles=leg, loc="lower left", frameon=True, framealpha=0.9, fontsize=7)
    save(fig, "activity_vs_plan")


def external_validation():
    tv = pl.read_parquet(C.PROCESSED_DIR / "transit_validation.parquet")
    pv = json.loads((C.STATS_DIR / "place_validation_stats.json").read_text("utf-8"))
    cat_en = {"발달상권": "commercial\ndistrict", "관광특구": "tourism\nzone",
              "인구밀집지역": "population-\nconcentrated\narea", "공원": "urban\npark",
              "고궁·문화유산": "historic\nheritage"}
    fig, ax = plt.subplots(1, 2, figsize=(W2, 78 * MM))
    x = tv["cat_resid_k1"].to_numpy(); y = tv["res_score"].to_numpy()
    ax[0].scatter(x, y, s=14, facecolor="#0072B2", edgecolor="none", alpha=0.55)
    b = np.polyfit(x, y, 1); xs = np.linspace(x.min(), x.max(), 10)
    ax[0].plot(xs, np.polyval(b, xs), color="black", lw=1.6)
    ax[0].axhline(0, color="0.6", lw=0.6)
    ax[0].set_xlabel("Catchment residential-regime share (k=1)")
    ax[0].set_ylabel("Station residential-commute score")
    ax[0].text(0.05, 0.92, "r = 0.55", transform=ax[0].transAxes, fontsize=8)
    ax[0].set_title("(a)", loc="left", fontsize=9, fontweight="bold")
    cr = pv["category_regime"]; cats = [cat_en.get(r["category"], r["category"]) for r in cr]
    actv = [r["mean_activity_share"] for r in cr]; resv = [r["mean_residential_share"] for r in cr]
    xi = np.arange(len(cats)); w = 0.38
    ax[1].bar(xi - w/2, actv, w, color="#D55E00", edgecolor="black", linewidth=0.5, label="activity share")
    ax[1].bar(xi + w/2, resv, w, color="#56B4E9", edgecolor="black", linewidth=0.5,
              hatch="////", label="residential share")
    ax[1].set_xticks(xi); ax[1].set_xticklabels(cats, fontsize=6.5)
    ax[1].set_ylabel("Mean regime share")
    ax[1].set_title("(b)", loc="left", fontsize=9, fontweight="bold")
    ax[1].legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    save(fig, "external_validation")


def presence_vs_dwell():
    d = json.loads((C.STATS_DIR / "presence_vs_dwell_stats.json").read_text("utf-8"))
    e = d["external_validation_delta"]["k1"]; dn = d["daynight_switch"]
    fig, ax = plt.subplots(1, 2, figsize=(W2, 70 * MM))
    labels = ["Residential\naxis", "Activity\naxis", "Full\ncomposition"]
    dwell = [e["dwell_res_axis"]["R2"], e["dwell_act_axis"]["R2"], e["dwell_full_comp"]["adj_R2"]]
    po = [e["po_res_axis"]["R2"], e["po_act_axis"]["R2"], e["po_full_comp"]["adj_R2"]]
    xi = np.arange(3); w = 0.38
    ax[0].bar(xi - w/2, dwell, w, color="#0072B2", edgecolor="black", linewidth=0.5, label="dwell-aware")
    ax[0].bar(xi + w/2, po, w, color="#E69F00", edgecolor="black", linewidth=0.5,
              hatch="////", label="presence-only")
    ax[0].set_xticks(xi); ax[0].set_xticklabels(labels, fontsize=7)
    ax[0].set_ylabel("External-validation R²")
    ax[0].set_title("(a)", loc="left", fontsize=9, fontweight="bold")
    ax[0].legend(frameon=False, fontsize=7.5)
    ax[1].bar([0, 1], [dn["dwell_aware_pct"], dn["presence_only_pct"]], 0.38,
              color=["#0072B2", "#E69F00"], edgecolor="black", linewidth=0.5, hatch=["", "////"])
    ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(["dwell-aware", "presence-only"], fontsize=7.5)
    ax[1].set_ylabel("% cells switching day↔night regime")
    ax[1].set_title("(b)", loc="left", fontsize=9, fontweight="bold")
    fig.tight_layout()
    save(fig, "presence_vs_dwell")


# ================================================================ supplementary
def regime_map(gg):
    counts = gg["dominant_functional"].value_counts().to_dict()
    fig, ax = plt.subplots(figsize=(95 * MM, 95 * MM))
    for r in REG_ORDER:
        sub = gg[gg["dominant_functional"] == r]
        if len(sub):
            sub.plot(ax=ax, color=REG_COLOR[r], linewidth=0)
    ax.set_aspect("equal"); ax.axis("off")
    ax.legend(handles=reg_handles(counts), loc="upper left", frameon=True,
              framealpha=0.9, edgecolor="0.7", fontsize=3, handlelength=0.55,
              handletextpad=0.25, labelspacing=0.18, borderpad=0.18, markerscale=0.6)
    save(fig, "regime_map")


def k_selection():
    summ = json.loads((C.MODELS_DIR / "ksweep_summary.json").read_text("utf-8"))
    Ks = [s["K"] for s in summ]
    K_STAR = 7
    panels = [
        ("bic", "BIC (train) ↓", OI_BLUE, None),
        ("holdout_perplexity", "Holdout perplexity ↓", OI_VERM, None),
        ("reinit_ari", "Reinit ARI ↑", OI_GREEN, 0.6)]
    fig, axes = plt.subplots(1, 3, figsize=(150 * MM, 58 * MM))
    for ax, (key, title, col, thr) in zip(axes, panels):
        y = [s[key] for s in summ]
        ax.plot(Ks, y, "-o", color=col, ms=4, mfc="white", mew=1.1, zorder=3)
        # highlight the selected K
        ys = summ[Ks.index(K_STAR)][key]
        ax.scatter([K_STAR], [ys], s=70, color=col, marker="*", zorder=5,
                   edgecolor="black", linewidth=0.5)
        ax.axvline(K_STAR, color="0.55", lw=0.8, ls="--", zorder=1)
        if thr is not None:
            ax.axhline(thr, color="0.4", lw=0.7, ls=":", zorder=1)
            ax.text(Ks[0], thr, " stability floor (0.6)", va="bottom", ha="left",
                    fontsize=5.6, color="0.35")
            ax.set_ylim(0, 1)
        ax.set_title(title, fontsize=8.5)
        ax.set_xlabel("Number of states K")
        ax.set_xticks(Ks); ax.tick_params(length=2)
        ax.grid(alpha=0.25, linewidth=0.5)
    axes[0].annotate("selected K = 7", xy=(K_STAR, summ[Ks.index(K_STAR)]["bic"]),
                     xytext=(0.5, 0.82), textcoords="axes fraction", fontsize=6.5,
                     ha="left", color="0.2",
                     arrowprops=dict(arrowstyle="->", color="0.5", lw=0.7))
    fig.tight_layout()
    save(fig, "k_selection")


def _purity_curve():
    """Per-unit observed purity + size-matched contiguous-patch null
    (SEED, N_PATCH as in step 18)."""
    R = load_step("robust18", "18_conclusions_robustness.py")   # build_adj, region_grow, purity_of
    reg = pl.read_parquet(C.PROCESSED_DIR / "cell_regime_summary.parquet")
    rob = pl.read_parquet(C.PROCESSED_DIR / "regime_robustness.parquet").select(
        ["cell_id", "grid_ix", "grid_iy"])
    plan = pl.read_parquet(C.PROCESSED_DIR / "cell_plan_map.parquet").select(
        ["cell_id", "sg_local", "sg_region"])
    reg = reg.join(rob, on="cell_id").join(plan, on="cell_id")
    nbrs, _ = R.build_adj(reg["grid_ix"].to_numpy(), reg["grid_iy"].to_numpy())
    dom = reg["dominant_functional"].to_list()
    isf = [d not in ("structural_missing", "none") for d in dom]
    n = len(dom)
    rng = np.random.default_rng(C.SEED)
    units = {"administrative dong": reg["admin_dong"].to_list(),
             "district living zone": reg["sg_local"].to_list(),
             "wide-area living zone": reg["sg_region"].to_list()}
    rows = {"tier": [], "size": [], "obs": [], "null": []}
    for tier, labels in units.items():
        by = {}
        for i, u in enumerate(labels):
            if u is not None:
                by.setdefault(u, []).append(i)
        for u, idx in by.items():
            p = R.purity_of(idx, dom, isf)
            if p is None:
                continue
            m = len(idx)
            ps = []
            for _ in range(R.N_PATCH):
                s = int(rng.integers(0, n))
                pv = R.purity_of(R.region_grow(s, m, nbrs, rng), dom, isf)
                if pv is not None:
                    ps.append(pv)
            rows["tier"].append(tier); rows["size"].append(m)
            rows["obs"].append(p); rows["null"].append(float(np.mean(ps)) if ps else np.nan)
    return rows


def purity_vs_size():
    d = _purity_curve()
    tier_col = {"administrative dong": OI_BLUE, "district living zone": OI_ORANGE,
                "wide-area living zone": OI_VERM}
    size = np.array(d["size"]); obs = np.array(d["obs"])
    null = np.array(d["null"]); tier = np.array(d["tier"])
    fig, ax = plt.subplots(figsize=(120 * MM, 78 * MM))
    # size-matched random-patch null: one grey cloud (depends on size, not unit type)
    ax.scatter(size, null, s=16, facecolor="none", edgecolor="0.55", linewidth=0.7,
               marker="s", label="size-matched random-patch null", zorder=2)
    for t, c in tier_col.items():
        m = tier == t
        ax.scatter(size[m], obs[m], s=22, color=c, alpha=0.8, linewidth=0,
                   label=f"observed — {t}", zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("Unit size (number of cells, log scale)")
    ax.set_ylabel("Functional-regime purity")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.legend(loc="upper right", frameon=True, framealpha=0.9, edgecolor="0.7",
              fontsize=6.8, handletextpad=0.4, labelspacing=0.35)
    fig.tight_layout()
    save(fig, "purity_vs_size")


def dwell_distribution():
    d = json.loads((C.STATS_DIR / "dwell_summary.json").read_text("utf-8"))
    shares = d["A_overall_share"]
    # start-time x dwell interaction (col-normalised share within each start-time bin)
    M = np.array(d["start_tz_by_dur"], dtype=float)
    Mn = M / M.sum(axis=0, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(W2, 72 * MM),
                             gridspec_kw={"width_ratios": [1, 1.45]})
    # (a) overall dwell-duration share
    a = axes[0]
    vals = [shares[k] for k in DUR_ORDER]
    a.bar(range(len(DUR_ORDER)), vals, color=OI_BLUE, edgecolor="black", linewidth=0.5)
    for i, v in enumerate(vals):
        a.text(i, v, f"{v*100:.1f}%", ha="center", va="bottom", fontsize=6)
    a.set_xticks(range(len(DUR_ORDER)))
    a.set_xticklabels([DUR_LABEL[k] for k in DUR_ORDER], rotation=40, ha="right")
    a.set_xlabel("Dwell duration (minutes)")
    a.set_ylabel("Population-weighted share")
    a.set_ylim(0, max(vals) * 1.18)
    a.set_title("(a)", loc="left", fontsize=9, fontweight="bold")
    a.tick_params(length=2)
    # (b) start-time x dwell interaction
    b = axes[1]
    im = b.imshow(Mn, aspect="auto", cmap="viridis", origin="lower", vmin=0, vmax=Mn.max())
    b.set_yticks(range(len(DUR_ORDER)))
    b.set_yticklabels([DUR_LABEL[k] for k in DUR_ORDER])
    b.set_xticks(range(len(TZ_ORDER)))
    b.set_xticklabels(TZ_LABEL, rotation=90, fontsize=5.6)
    b.set_xlabel("Stay start-time bin (hour)")
    b.set_ylabel("Dwell duration (min)")
    b.set_title("(b)  share within each start-time bin", loc="left", fontsize=9,
                fontweight="bold")
    b.tick_params(length=2)
    cb = fig.colorbar(im, ax=b, fraction=0.045, pad=0.02)
    cb.set_label("Share", fontsize=7); cb.ax.tick_params(labelsize=6)
    fig.tight_layout()
    save(fig, "dwell_distribution")


def main():
    plt.rcParams.update(MAIN_RC)
    gg = grid_regime()
    model_ladder(); regime_day_night(gg); regime_diurnal(); activity_vs_plan(gg)
    external_validation(); presence_vs_dwell(); regime_map(gg)
    plt.rcParams.update(SI_RC)
    k_selection(); dwell_distribution(); purity_vs_size()
    print("DONE")


if __name__ == "__main__":
    main()
