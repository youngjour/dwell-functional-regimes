"""Step 07 - shared-state Classical Gaussian HMM K-sweep {4..9} -> K* selection.

Pooled emissions AND pooled transitions (K-selection simplification; cell-specific
transitions deferred to the model ladder). Diagonal Gaussian emissions.
5 random reinits per K (seeded). Train = 2025.01-2026.03 sample; eval = holdout.

Metrics: BIC(train), holdout avg log-lik / perplexity, reinit ARI stability.
Emission means inverted to original feature space (inverse-CLR) for interpretation
+ state x time_bin activation. Outputs: results/models/k{K}/, results/models/ksweep_summary.json,
results/eda/G_*, H_*.png.

Run (background): python src/07_hmm_ksweep.py
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from hmmlearn.hmm import GaussianHMM
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

SEED = C.SEED
K_RANGE = [4, 5, 6, 7, 8, 9]
N_REINIT = 5
N_ITER = 40
TOL = 1e-2
IMPL = "scaling"            # faster than default "log" for K-sweep
ARI_SUBSAMPLE = 80_000
TB_ORDER = ["00"] + [f"{h:02d}" for h in range(6, 24)]
plt.rcParams.update({"figure.dpi": 110, "font.size": 9})


def n_params(K: int, D: int) -> int:
    # diag Gaussian: means K*D, covars K*D, startprob K-1, transmat K*(K-1)
    return 2 * K * D + (K - 1) + K * (K - 1)


def softmax(v):
    e = np.exp(v - v.max())
    return e / e.sum()


def fit_one(X, lengths, K, rs):
    m = GaussianHMM(n_components=K, covariance_type="diag", n_iter=N_ITER,
                    tol=TOL, random_state=rs, init_params="stmc", params="stmc",
                    verbose=False, min_covar=1e-3, implementation=IMPL)
    m.fit(X, lengths)
    return m


def main():
    h = (C.MODELS_DIR / "latest_input.txt").read_text().strip()
    indir = C.MODELS_DIR / f"input_{h}"
    Xtr = np.load(indir / "train_X.npy")
    Ltr = np.load(indir / "train_lengths.npy")
    Xho = np.load(indir / "holdout_X.npy")
    Lho = np.load(indir / "holdout_lengths.npy")
    meta = json.loads((indir / "feature_meta.json").read_text("utf-8"))
    D = Xtr.shape[1]
    pm, ps = meta["scaler"]["presence_log"]
    sm, ss = meta["scaler"]["stay_vol_log"]
    MIX, DWELL = meta["mix_parts"], meta["dwell_parts"]
    tr_meta = pl.read_parquet(indir / "train_meta.parquet")
    print(f"input {h}: train {Xtr.shape} holdout {Xho.shape} D={D}", flush=True)

    rng = np.random.default_rng(SEED)
    ari_idx = rng.choice(Xtr.shape[0], min(ARI_SUBSAMPLE, Xtr.shape[0]),
                         replace=False)

    summary = []
    best_models = {}
    for K in K_RANGE:
        t0 = time.time()
        fits = []
        for r in range(N_REINIT):
            try:
                m = fit_one(Xtr, Ltr, K, SEED + 1000 * K + r)
                ll = m.score(Xtr, Ltr)
                fits.append((ll, m))
            except Exception as e:  # noqa: BLE001
                print(f"  K={K} reinit{r} FAILED {e!r}", flush=True)
        if not fits:
            continue
        fits.sort(key=lambda t: t[0], reverse=True)
        best_ll, best = fits[0]
        # reinit stability: pairwise ARI of state labels on subsample
        labs = [m.predict(Xtr, Ltr)[ari_idx] for _, m in fits]
        aris = [adjusted_rand_score(labs[i], labs[j])
                for i in range(len(labs)) for j in range(i + 1, len(labs))]
        ari_mean = float(np.mean(aris)) if aris else float("nan")
        Ntr = Xtr.shape[0]
        bic = -2 * best_ll + n_params(K, D) * np.log(Ntr)
        ho_ll = best.score(Xho, Lho)
        ho_avg = ho_ll / Xho.shape[0]
        perplexity = float(np.exp(-ho_avg))
        conv = best.monitor_.converged
        summary.append({
            "K": K, "train_ll": best_ll, "bic": bic, "holdout_avg_ll": ho_avg,
            "holdout_perplexity": perplexity, "reinit_ari": ari_mean,
            "n_params": n_params(K, D), "converged": bool(conv),
            "ll_spread": float(fits[0][0] - fits[-1][0]), "sec": round(time.time()-t0, 1),
        })
        best_models[K] = best
        kdir = C.MODELS_DIR / f"k{K}"
        kdir.mkdir(parents=True, exist_ok=True)
        with open(kdir / "model.pkl", "wb") as f:
            pickle.dump(best, f)
        print(f"K={K}: BIC={bic:.0f} ho_avgLL={ho_avg:.4f} ppl={perplexity:.3f} "
              f"ARI={ari_mean:.3f} conv={conv} ({summary[-1]['sec']}s)", flush=True)

        # ---- emission interpretation (inverse transforms) ----
        states = best.predict(Xtr, Ltr)
        prior = np.bincount(states, minlength=K) / len(states)
        rows = []
        for k in range(K):
            mu = best.means_[k]
            comp_mix = softmax(mu[2:2 + len(MIX)])
            comp_dw = softmax(mu[2 + len(MIX):2 + len(MIX) + len(DWELL)])
            rec = {"K": K, "state": k, "prior_pct": round(100 * prior[k], 2),
                   "mean_total_pop": round(float(np.expm1(mu[0] * ps + pm)), 1),
                   "stay_vol": round(float(np.expm1(mu[1] * ss + sm)), 1),
                   "dwell_missing_rate": round(float(mu[2 + len(MIX) + len(DWELL)]), 3),
                   "short000": round(float(comp_dw[0]), 3),
                   "dw060": round(float(comp_dw[2]), 3),
                   "long240": round(float(comp_dw[5]), 3)}
            top = np.argsort(comp_mix)[::-1][:3]
            rec["top_demo"] = ";".join(
                f"{MIX[i].split('_',1)[1]}={comp_mix[i]:.2f}" for i in top)
            rows.append(rec)
        pl.DataFrame(rows).write_csv(kdir / "emission_table.csv")

        # state x time_bin activation heatmap
        tm = tr_meta.with_columns(pl.Series("state", states))
        ct = (tm.group_by(["time_bin", "state"]).len()
              .pivot(values="len", index="time_bin", on="state").fill_null(0)
              .sort("time_bin"))
        Mtb = np.zeros((K, len(TB_ORDER)))
        for r in ct.iter_rows(named=True):
            ti = TB_ORDER.index(r["time_bin"])
            for k in range(K):
                if str(k) in r and r[str(k)] is not None:
                    Mtb[k, ti] = r[str(k)]
        Mtb = Mtb / Mtb.sum(axis=0, keepdims=True)  # P(state|time_bin)
        fig, ax = plt.subplots(figsize=(11, 4))
        im = ax.imshow(Mtb, aspect="auto", cmap="viridis", origin="lower")
        ax.set_yticks(range(K)); ax.set_yticklabels([f"s{k}" for k in range(K)])
        ax.set_xticks(range(len(TB_ORDER))); ax.set_xticklabels(TB_ORDER, rotation=90)
        ax.set_title(f"K={K}: P(state | time_bin)"); ax.set_xlabel("time_bin")
        fig.colorbar(im, ax=ax, fraction=0.04)
        fig.tight_layout(); fig.savefig(C.EDA_DIR / f"H_state_timebin_K{K}.png")
        plt.close(fig)

    # ---- summary metrics figure ----
    Ks = [s["K"] for s in summary]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].plot(Ks, [s["bic"] for s in summary], "-o"); ax[0].set_title("BIC (train, lower better)")
    ax[0].set_xlabel("K")
    ax[1].plot(Ks, [s["holdout_perplexity"] for s in summary], "-o", color="tab:red")
    ax[1].set_title("Holdout perplexity (lower better)"); ax[1].set_xlabel("K")
    ax[2].plot(Ks, [s["reinit_ari"] for s in summary], "-o", color="tab:green")
    ax[2].set_title("Reinit ARI (higher=stable)"); ax[2].set_xlabel("K"); ax[2].set_ylim(0, 1)
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(C.EDA_DIR / "G_ksweep_metrics.png")
    plt.close(fig)

    (C.MODELS_DIR / "ksweep_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("DONE")


if __name__ == "__main__":
    main()
