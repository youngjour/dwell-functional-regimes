"""Step 09 - K=7 model ladder: LDA | HMM | Sticky HMM | HSMM (+ HMM K=6 sensitivity).

Shared input v2 (z-scored CLR), shared K and (for HMM-family) shared emission form
to isolate the model-type effect. Pooled transitions. 5 reinit each.

Metrics: BIC(train), holdout avg LL/perplexity split 2026.04 vs 2026.05, reinit ARI,
temporal coherence (Viterbi P(s_t=s_{t+1})), duration calibration KL(empirical||implied).
Interpretation: inverse-transform emission table + P(state|time_bin) plots.

Run (bg): python src/09_fit_ladder.py
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
import scipy.sparse as sp
from hmmlearn.hmm import GaussianHMM
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
import importlib.util
_spec = importlib.util.spec_from_file_location("_edhsmm",
                                               str(Path(__file__).parent / "_edhsmm.py"))
edh = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(edh)

SEED = C.SEED
N_REINIT = 5
HMM_ITER = 30
TB_ORDER = ["00"] + [f"{h:02d}" for h in range(6, 24)]
KAPPAS = [1, 10, 50, 100]
LDA_Q = 5
plt.rcParams.update({"figure.dpi": 110, "font.size": 9})


def softmax(v):
    e = np.exp(v - v.max()); return e / e.sum()


def load():
    h = (C.MODELS_DIR / "latest_input_v2.txt").read_text().strip()
    d = C.MODELS_DIR / f"input_v2_{h}"
    meta = json.loads((d / "feature_meta.json").read_text("utf-8"))
    Xtr3 = np.load(d / "train_X3d.npy"); Xho3 = np.load(d / "holdout_X3d.npy")
    ho_meta = pl.read_parquet(d / "holdout_meta.parquet")
    T = Xtr3.shape[1]
    ym = ho_meta["year_month"].to_numpy().reshape(-1, T)[:, 0]   # per-seq month
    return h, d, meta, Xtr3, Xho3, ym, T


def flat(X3):
    N, T, D = X3.shape
    return X3.reshape(N * T, D), np.full(N, T, dtype=int)


# ---------- metrics ----------
def bic(train_ll, npar, nobs):
    return -2 * train_ll + npar * np.log(nobs)


def coherence(labels):
    return float((labels[:, 1:] == labels[:, :-1]).mean())


def empirical_dur_pmf(labels, K, T):
    """run-length pmf per state, excluding boundary-touching (censored) runs."""
    out = np.zeros((K, T))
    N = labels.shape[0]
    change = labels[:, 1:] != labels[:, :-1]
    seg_id = np.zeros((N, T), int); seg_id[:, 1:] = np.cumsum(change, axis=1)
    gkey = (np.arange(N)[:, None] * T + seg_id).ravel()
    uniq, first_idx, counts = np.unique(gkey, return_index=True, return_counts=True)
    seg_state = labels.ravel()[first_idx]
    # boundary: a segment touches t==0 or t==T-1
    seg_start_t = first_idx % T
    seg_end_t = seg_start_t + counts - 1
    interior = (seg_start_t > 0) & (seg_end_t < T - 1)
    for k in range(K):
        m = (seg_state == k) & interior
        for c in counts[m]:
            if 1 <= c <= T:
                out[k, c - 1] += 1
    rs = out.sum(1, keepdims=True)
    return np.where(rs > 0, out / np.where(rs > 0, rs, 1), 0.0)


def geom_dur_pmf(transmat, T):
    K = transmat.shape[0]
    a = np.clip(np.diag(transmat), 1e-6, 1 - 1e-6)
    d = np.arange(1, T + 1)
    pmf = a[:, None] ** (d - 1) * (1 - a)[:, None]
    return pmf / pmf.sum(1, keepdims=True)


def dur_kl(labels, implied, K, T):
    emp = empirical_dur_pmf(labels, K, T)
    prior = np.bincount(labels.ravel(), minlength=K).astype(float)
    prior /= prior.sum()
    eps = 1e-9
    kls = []
    for k in range(K):
        if emp[k].sum() == 0:
            continue
        p = emp[k] + eps; q = implied[k] + eps
        p /= p.sum(); q /= q.sum()
        kls.append(prior[k] * np.sum(p * np.log(p / q)))
    return float(np.sum(kls))


def reinit_ari(label_list, idx):
    a = [L.ravel()[idx] for L in label_list]
    v = [adjusted_rand_score(a[i], a[j])
         for i in range(len(a)) for j in range(i + 1, len(a))]
    return float(np.mean(v)) if v else float("nan")


# ---------- emission interpretation (HMM-family) ----------
def interp_means(means, meta, K):
    pm, ps = meta["scaler"]["presence_log"]; sm, ss = meta["scaler"]["stay_vol_log"]
    mzc, mzs = np.array(meta["clr_mix_z"][0]), np.array(meta["clr_mix_z"][1])
    dzc, dzs = np.array(meta["clr_dwell_z"][0]), np.array(meta["clr_dwell_z"][1])
    MIX, DW = meta["mix_parts"], meta["dwell_parts"]
    rows = []
    for k in range(K):
        mu = means[k]
        clr_mix = mu[2:2 + len(MIX)] * mzs + mzc
        clr_dw = mu[2 + len(MIX):2 + len(MIX) + len(DW)] * dzs + dzc
        cm = softmax(clr_mix); cd = softmax(clr_dw)
        top = np.argsort(cm)[::-1][:3]
        rows.append({
            "state": k,
            "mean_total_pop": round(float(np.expm1(mu[0] * ps + pm)), 1),
            "stay_vol": round(float(np.expm1(mu[1] * ss + sm)), 1),
            "short000": round(float(cd[0]), 3), "long240": round(float(cd[5]), 3),
            "dwell_missing": round(float(mu[2 + len(MIX) + len(DW)]), 3),
            "top_demo": ";".join(f"{MIX[i].split('_',1)[1]}={cm[i]:.2f}" for i in top),
        })
    return rows


def state_timebin_plot(labels, K, title, path):
    T = labels.shape[1]
    M = np.zeros((K, T))
    for ti in range(T):
        c = np.bincount(labels[:, ti], minlength=K)
        M[:, ti] = c
    M = M / M.sum(0, keepdims=True)
    fig, ax = plt.subplots(figsize=(11, 3.6))
    im = ax.imshow(M, aspect="auto", cmap="viridis", origin="lower")
    ax.set_yticks(range(K)); ax.set_yticklabels([f"s{k}" for k in range(K)])
    ax.set_xticks(range(T)); ax.set_xticklabels(TB_ORDER, rotation=90)
    ax.set_title(title); ax.set_xlabel("time_bin")
    fig.colorbar(im, ax=ax, fraction=0.04); fig.tight_layout()
    fig.savefig(path); plt.close(fig)


# ---------- fitters ----------
def fit_hmm(X2, lengths, K, rs, transmat_prior=None):
    # scaling is fast but underflows on some inits; fall back to stable log impl.
    for impl in ("scaling", "log"):
        m = GaussianHMM(n_components=K, covariance_type="diag", n_iter=HMM_ITER,
                        tol=1e-2, random_state=rs, min_covar=1e-3,
                        implementation=impl)
        if transmat_prior is not None:
            m.transmat_prior = transmat_prior
        try:
            m.fit(X2, lengths)
            if np.isfinite(m.score(X2, lengths)):
                return m
        except ValueError as e:
            if "underflow" not in str(e):
                raise
    raise ValueError("HMM fit failed under both scaling and log implementations")


def run_hmm_family(name, X2, ltr, Xho2, lho, Xho_apr, Xho_may, K, meta, T,
                   ari_idx, transmat_prior=None, sticky=False):
    fits = []
    for r in range(N_REINIT):
        try:
            m = fit_hmm(X2, ltr, K, SEED + 31 * r, transmat_prior)
            fits.append((m.score(X2, ltr), m))
        except Exception as e:  # noqa: BLE001
            print(f"  {name} reinit{r} FAIL {e!r}", flush=True)
    if not fits:
        raise RuntimeError(f"{name}: all reinits failed")
    fits.sort(key=lambda t: t[0], reverse=True)
    best_ll, best = fits[0]
    D = X2.shape[1]
    npar = 2 * K * D + (K - 1) + K * (K - 1)
    labtr = best.predict(X2, ltr).reshape(-1, T)
    lab_list = [mm.predict(X2, ltr).reshape(-1, T) for _, mm in fits]
    implied = geom_dur_pmf(best.transmat_, T)
    res = dict(
        model=name, K=K, train_ll=best_ll, bic=bic(best_ll, npar, X2.shape[0]),
        ho_avg_ll=best.score(Xho2, lho) / Xho2.shape[0],
        ho_apr_avg=best.score(*Xho_apr) / Xho_apr[0].shape[0],
        ho_may_avg=best.score(*Xho_may) / Xho_may[0].shape[0],
        reinit_ari=reinit_ari(lab_list, ari_idx),
        coherence=coherence(labtr), dur_kl=dur_kl(labtr, implied, K, T),
        n_params=npar, ll_spread=float(fits[0][0] - fits[-1][0]),
    )
    res["ho_perplexity"] = float(np.exp(-res["ho_avg_ll"]))
    return res, best, labtr


def run_hsmm(X3, Xho3, Xho3_apr, Xho3_may, K, meta, T, ari_idx):
    fits = []
    for r in range(N_REINIT):
        m = edh.EDHSMM(K, n_iter=25, random_state=SEED + 31 * r).fit(X3)
        fits.append((m.train_ll_, m))
    fits.sort(key=lambda t: t[0], reverse=True)
    best_ll, best = fits[0]
    D = X3.shape[2]
    npar = best.n_params(D)
    labtr = best.predict(X3)
    lab_list = [mm.predict(X3) for _, mm in fits]
    res = dict(
        model="HSMM", K=K, train_ll=best_ll, bic=bic(best_ll, npar, X3.shape[0] * T),
        ho_avg_ll=best.score(Xho3) / (Xho3.shape[0] * T),
        ho_apr_avg=best.score(Xho3_apr) / (Xho3_apr.shape[0] * T) if Xho3_apr.shape[0] else float("nan"),
        ho_may_avg=best.score(Xho3_may) / (Xho3_may.shape[0] * T) if Xho3_may.shape[0] else float("nan"),
        reinit_ari=reinit_ari(lab_list, ari_idx),
        coherence=coherence(labtr), dur_kl=dur_kl(labtr, best.dur_pmf_, K, T),
        n_params=npar, ll_spread=float(fits[0][0] - fits[-1][0]),
    )
    res["ho_perplexity"] = float(np.exp(-res["ho_avg_ll"]))
    return res, best, labtr


# ---------- LDA baseline ----------
def discretize(Xcont, edges):
    N, Dc = Xcont.shape; Q = edges.shape[1] + 1
    rows = np.repeat(np.arange(N), Dc)
    bins = np.empty((N, Dc), int)
    for j in range(Dc):
        bins[:, j] = np.digitize(Xcont[:, j], edges[j])
    cols = (np.arange(Dc)[None, :] * Q + bins).ravel()
    return rows, cols, Dc * Q


def build_lda_dt(X2, edges, ind_off):
    cont = X2[:, :24]; ind = X2[:, 24:26]
    rows, cols, base = discretize(cont, edges)
    data = np.ones_like(rows, float)
    # indicators as extra tokens
    ir, ic = np.where(ind > 0.5)
    rows2 = np.concatenate([rows, ir]); cols2 = np.concatenate([cols, base + ic])
    data2 = np.concatenate([data, np.ones_like(ir, float)])
    V = base + 2
    return sp.csr_matrix((data2, (rows2, cols2)), shape=(X2.shape[0], V))


def run_lda(X2, Xho2, T, ho_meta_ym, ari_idx, meta_pl_tr):
    edges = np.quantile(X2[:, :24], np.linspace(0, 1, LDA_Q + 1)[1:-1], axis=0).T
    dt_tr = build_lda_dt(X2, edges, None)
    dt_ho = build_lda_dt(Xho2, edges, None)
    fits = []
    for r in range(N_REINIT):
        lda = LatentDirichletAllocation(n_components=7, learning_method="online",
                                        max_iter=15, batch_size=4096,
                                        random_state=SEED + 31 * r, n_jobs=1)
        lda.fit(dt_tr)
        fits.append((-lda.perplexity(dt_tr), lda))
    fits.sort(key=lambda t: t[0], reverse=True)
    best = fits[0][1]
    labtr = best.transform(dt_tr).argmax(1).reshape(-1, T)
    lab_list = [f[1].transform(dt_tr).argmax(1).reshape(-1, T) for f in fits]
    res = dict(
        model="LDA", K=7, train_ll=float("nan"), bic=float("nan"),
        ho_avg_ll=float("nan"), ho_apr_avg=float("nan"), ho_may_avg=float("nan"),
        reinit_ari=reinit_ari(lab_list, ari_idx),
        coherence=coherence(labtr), dur_kl=float("nan"),
        n_params=int(best.components_.size), ll_spread=float("nan"),
        ho_perplexity=float(best.perplexity(dt_ho)),
    )
    return res, best, labtr


def main():
    h, d, meta, Xtr3, Xho3, ho_ym, T = load()
    outdir = C.MODELS_DIR / f"ladder_{h}"
    outdir.mkdir(parents=True, exist_ok=True)
    Xtr2, ltr = flat(Xtr3); Xho2, lho = flat(Xho3)
    apr = ho_ym == "202604"; may = ho_ym == "202605"
    Xho3_apr, Xho3_may = Xho3[apr], Xho3[may]
    Xho2_apr, lho_apr = flat(Xho3_apr); Xho2_may, lho_may = flat(Xho3_may)
    rng = np.random.default_rng(SEED)
    ari_idx = rng.choice(Xtr3.shape[0] * T, min(80000, Xtr3.shape[0] * T), replace=False)
    print(f"ladder input {h}: train {Xtr3.shape} holdout {Xho3.shape} "
          f"(apr {apr.sum()} may {may.sum()} seqs)", flush=True)

    results = []; labels_store = {}

    # --- HMM K7, K6 ---
    for K in [7, 6]:
        t0 = time.time()
        res, m, lab = run_hmm_family(f"HMM_K{K}", Xtr2, ltr, Xho2, lho,
                                     (Xho2_apr, lho_apr), (Xho2_may, lho_may),
                                     K, meta, T, ari_idx)
        results.append(res); labels_store[f"HMM_K{K}"] = lab
        pickle.dump(m, open(outdir / f"HMM_K{K}.pkl", "wb"))
        pl.DataFrame(interp_means(m.means_, meta, K)).write_csv(outdir / f"HMM_K{K}_emission.csv")
        state_timebin_plot(lab, K, f"HMM K={K}: P(state|time_bin)",
                           C.EDA_DIR / f"I_timebin_HMM_K{K}.png")
        print(f"HMM_K{K}: BIC={res['bic']:.0f} ho_ppl={res['ho_perplexity']:.1f} "
              f"ARI={res['reinit_ari']:.3f} coh={res['coherence']:.3f} "
              f"durKL={res['dur_kl']:.3f} ({time.time()-t0:.0f}s)", flush=True)

    # --- Sticky HMM K7 x kappa ---
    for kap in KAPPAS:
        t0 = time.time()
        tp = 1.0 + kap * np.eye(7)
        res, m, lab = run_hmm_family(f"Sticky_k{kap}", Xtr2, ltr, Xho2, lho,
                                     (Xho2_apr, lho_apr), (Xho2_may, lho_may),
                                     7, meta, T, ari_idx, transmat_prior=tp, sticky=True)
        res["kappa"] = kap
        results.append(res); labels_store[f"Sticky_k{kap}"] = lab
        pickle.dump(m, open(outdir / f"Sticky_k{kap}.pkl", "wb"))
        pl.DataFrame(interp_means(m.means_, meta, 7)).write_csv(outdir / f"Sticky_k{kap}_emission.csv")
        print(f"Sticky_k{kap}: BIC={res['bic']:.0f} ho_ppl={res['ho_perplexity']:.1f} "
              f"ARI={res['reinit_ari']:.3f} coh={res['coherence']:.3f} "
              f"durKL={res['dur_kl']:.3f} ({time.time()-t0:.0f}s)", flush=True)

    # --- HSMM K7 ---
    t0 = time.time()
    res, m, lab = run_hsmm(Xtr3, Xho3, Xho3_apr, Xho3_may, 7, meta, T, ari_idx)
    results.append(res); labels_store["HSMM"] = lab
    pickle.dump({"means_": m.means_, "vars_": m.vars_, "transmat_": m.transmat_,
                 "startprob_": m.startprob_, "dur_pmf_": m.dur_pmf_,
                 "dur_mean_": m.dur_mean_}, open(outdir / "HSMM.pkl", "wb"))
    pl.DataFrame(interp_means(m.means_, meta, 7)).write_csv(outdir / "HSMM_emission.csv")
    state_timebin_plot(lab, 7, "HSMM K=7: P(state|time_bin)",
                       C.EDA_DIR / "I_timebin_HSMM.png")
    np.save(outdir / "HSMM_dur_pmf.npy", m.dur_pmf_)
    print(f"HSMM: BIC={res['bic']:.0f} ho_ppl={res['ho_perplexity']:.1f} "
          f"ARI={res['reinit_ari']:.3f} coh={res['coherence']:.3f} "
          f"durKL={res['dur_kl']:.3f} dur_mean={np.round(m.dur_mean_,2)} "
          f"({time.time()-t0:.0f}s)", flush=True)

    # --- LDA ---
    t0 = time.time()
    res, lda, lab = run_lda(Xtr2, Xho2, T, ho_ym, ari_idx, None)
    results.append(res); labels_store["LDA"] = lab
    state_timebin_plot(lab, 7, "LDA 7-topic: P(topic|time_bin)",
                       C.EDA_DIR / "I_timebin_LDA.png")
    print(f"LDA: ho_ppl={res['ho_perplexity']:.1f} ARI={res['reinit_ari']:.3f} "
          f"coh={res['coherence']:.3f} ({time.time()-t0:.0f}s)", flush=True)

    # save labels + summary
    np.savez_compressed(outdir / "viterbi_labels.npz", **labels_store)
    (outdir / "ladder_summary.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    # comparison figures
    comp_fig(results)
    print("DONE")


def comp_fig(results):
    fam = [r for r in results if r["model"] != "LDA"]
    names = [r["model"] for r in fam]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    ax[0].bar(names, [r["ho_perplexity"] for r in fam], color="tab:red")
    ax[0].set_title("Holdout perplexity (lower)"); ax[0].tick_params(axis="x", rotation=40)
    ax[1].bar(names, [r["coherence"] for r in fam], color="tab:blue")
    ax[1].set_title("Temporal coherence P(s_t=s_t+1)"); ax[1].tick_params(axis="x", rotation=40)
    ax[2].bar(names, [r["dur_kl"] for r in fam], color="tab:green")
    ax[2].set_title("Duration KL(emp||implied) (lower)"); ax[2].tick_params(axis="x", rotation=40)
    for a in ax:
        a.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(C.EDA_DIR / "J_ladder_compare.png"); plt.close(fig)


if __name__ == "__main__":
    main()
