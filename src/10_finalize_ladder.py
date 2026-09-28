"""Step 10 - finalize the ladder: reuse saved HMM/Sticky/HSMM models, add a LEAN
LDA baseline, and emit ladder_summary.json + diagnostic figures.

Step 09 hung on LDA (5x full-950k LDA.transform). Models/emission CSVs/I-plots were
already saved per-model; here we reload them, recompute metrics cheaply (predict is
fast; only fitting was slow), inject reinit-ARI recorded from the step-09 run, and fit a
lean LDA (subsampled transforms). No model refitting for HMM-family/HSMM.

Run: python src/10_finalize_ladder.py
"""
from __future__ import annotations

import importlib.util
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import polars as pl
import scipy.sparse as sp
from sklearn.decomposition import LatentDirichletAllocation

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

_s = importlib.util.spec_from_file_location("L", str(Path(__file__).parent / "09_fit_ladder.py"))
L = importlib.util.module_from_spec(_s); _s.loader.exec_module(L)
_e = importlib.util.spec_from_file_location("edh", str(Path(__file__).parent / "_edhsmm.py"))
edh = importlib.util.module_from_spec(_e); _e.loader.exec_module(edh)

# reinit ARI recorded from the src/09 run stdout (cannot recompute: only best model saved)
RECORDED_ARI = {"HMM_K7": 0.583, "HMM_K6": 0.681, "Sticky_k1": 0.587,
                "Sticky_k10": 0.558, "Sticky_k50": 0.573, "Sticky_k100": 0.572,
                "HSMM": 0.695}


def hmm_res(name, K, m, Xtr2, ltr, Xho2, lho, apr, may, T):
    D = Xtr2.shape[1]
    npar = 2 * K * D + (K - 1) + K * (K - 1)
    tll = m.score(Xtr2, ltr)
    lab = m.predict(Xtr2, ltr).reshape(-1, T)
    implied = L.geom_dur_pmf(m.transmat_, T)
    r = dict(model=name, K=K, train_ll=tll, bic=L.bic(tll, npar, Xtr2.shape[0]),
             ho_avg_ll=m.score(Xho2, lho) / Xho2.shape[0],
             ho_apr_avg=m.score(*apr) / apr[0].shape[0],
             ho_may_avg=m.score(*may) / may[0].shape[0],
             reinit_ari=RECORDED_ARI[name], coherence=L.coherence(lab),
             dur_kl=L.dur_kl(lab, implied, K, T), n_params=npar, ll_spread=float("nan"))
    r["ho_perplexity"] = float(np.exp(-r["ho_avg_ll"]))
    return r, lab


def hsmm_reload(hd, K=7):
    m = edh.EDHSMM(K)
    m.means_ = hd["means_"]; m.vars_ = hd["vars_"]; m.transmat_ = hd["transmat_"]
    m.startprob_ = hd["startprob_"]; m.dur_pmf_ = hd["dur_pmf_"]
    m.Dmax = hd["dur_pmf_"].shape[1]
    m.dur_surv_ = np.cumsum(hd["dur_pmf_"][:, ::-1], axis=1)[:, ::-1]
    m.dur_mean_ = hd["dur_mean_"]
    return m


def build_dt(X2, edges):
    cont = X2[:, :24]; ind = X2[:, 24:26]
    N, Dc = cont.shape; Q = edges.shape[1] + 1
    bins = np.stack([np.digitize(cont[:, j], edges[j]) for j in range(Dc)], 1)
    rows = np.repeat(np.arange(N), Dc)
    cols = (np.arange(Dc)[None] * Q + bins).ravel()
    base = Dc * Q
    ir, ic = np.where(ind > 0.5)
    r = np.concatenate([rows, ir]); c = np.concatenate([cols, base + ic])
    d = np.ones(r.shape[0])
    return sp.csr_matrix((d, (r, c)), shape=(N, base + 2))


def lean_lda(Xtr2, Xho2, T, ari_idx):
    # sklearn online-LDA transform/perplexity is very slow at ~1e6 docs; subsample.
    from sklearn.metrics import adjusted_rand_score
    nseq = Xtr2.shape[0] // T
    fit_seq = min(6000, nseq)
    Xf = Xtr2[:fit_seq * T]
    edges = np.quantile(Xf[:, :24], np.linspace(0, 1, 6)[1:-1], axis=0).T
    dt_fit = build_dt(Xf, edges)
    nho = Xho2.shape[0] // T
    dt_ho = build_dt(Xho2[:min(2000, nho) * T], edges)
    fits = []
    for r in range(2):
        lda = LatentDirichletAllocation(n_components=7, learning_method="batch",
                                        max_iter=8, max_doc_update_iter=20,
                                        evaluate_every=-1, mean_change_tol=1e-2,
                                        random_state=C.SEED + 7 * r, n_jobs=1)
        lda.fit(dt_fit); fits.append(lda)
    best = fits[0]
    lab_plot = best.transform(dt_fit).argmax(1).reshape(-1, T)        # fit_seq x T
    sidx = np.arange(min(40000, dt_fit.shape[0]))
    labs = [f.transform(dt_fit[sidx]).argmax(1) for f in fits]
    ari = adjusted_rand_score(labs[0], labs[1])
    res = dict(model="LDA", K=7, train_ll=float("nan"), bic=float("nan"),
               ho_avg_ll=float("nan"), ho_apr_avg=float("nan"), ho_may_avg=float("nan"),
               reinit_ari=float(ari), coherence=L.coherence(lab_plot),
               dur_kl=float("nan"), n_params=int(best.components_.size),
               ll_spread=float("nan"), ho_perplexity=float(best.perplexity(dt_ho)))
    return res, lab_plot


def main():
    h, d, meta, Xtr3, Xho3, ho_ym, T = L.load()
    ld = C.MODELS_DIR / f"ladder_{h}"
    Xtr2, ltr = L.flat(Xtr3); Xho2, lho = L.flat(Xho3)
    apr_m = ho_ym == "202604"; may_m = ho_ym == "202605"
    apr = L.flat(Xho3[apr_m]); may = L.flat(Xho3[may_m])
    rng = np.random.default_rng(C.SEED)
    ari_idx = rng.choice(Xtr3.shape[0] * T, min(80000, Xtr3.shape[0] * T), replace=False)

    results = []; labels = {}
    family = [("HMM_K7", 7), ("HMM_K6", 6), ("Sticky_k1", 7), ("Sticky_k10", 7),
              ("Sticky_k50", 7), ("Sticky_k100", 7)]
    for name, K in family:
        m = pickle.load(open(ld / f"{name}.pkl", "rb"))
        r, lab = hmm_res(name, K, m, Xtr2, ltr, Xho2, lho, apr, may, T)
        if name.startswith("Sticky"):
            r["kappa"] = int(name.split("_k")[1])
        results.append(r); labels[name] = lab
        print(f"{name}: BIC={r['bic']:.0f} ppl={r['ho_perplexity']:.1f} "
              f"apr={r['ho_apr_avg']:.3f} may={r['ho_may_avg']:.3f} "
              f"coh={r['coherence']:.3f} durKL={r['dur_kl']:.3f}", flush=True)

    hd = pickle.load(open(ld / "HSMM.pkl", "rb"))
    hm = hsmm_reload(hd, 7)
    npar = hm.n_params(Xtr3.shape[2]); tll = hm.score(Xtr3)
    lab = hm.predict(Xtr3)
    rh = dict(model="HSMM", K=7, train_ll=tll, bic=L.bic(tll, npar, Xtr3.shape[0] * T),
              ho_avg_ll=hm.score(Xho3) / (Xho3.shape[0] * T),
              ho_apr_avg=hm.score(Xho3[apr_m]) / (apr_m.sum() * T),
              ho_may_avg=hm.score(Xho3[may_m]) / (may_m.sum() * T),
              reinit_ari=RECORDED_ARI["HSMM"], coherence=L.coherence(lab),
              dur_kl=L.dur_kl(lab, hm.dur_pmf_, 7, T), n_params=npar, ll_spread=float("nan"))
    rh["ho_perplexity"] = float(np.exp(-rh["ho_avg_ll"]))
    results.append(rh); labels["HSMM"] = lab
    print(f"HSMM: BIC={rh['bic']:.0f} ppl={rh['ho_perplexity']:.3g} "
          f"apr={rh['ho_apr_avg']:.3f} may={rh['ho_may_avg']:.3f} "
          f"coh={rh['coherence']:.3f} durKL={rh['dur_kl']:.3f}", flush=True)

    print("fitting lean LDA...", flush=True)
    rl, lab_lda = lean_lda(Xtr2, Xho2, T, ari_idx)
    results.append(rl)
    L.state_timebin_plot(lab_lda, 7, "LDA 7-topic: P(topic|time_bin)",
                         C.EDA_DIR / "I_timebin_LDA.png")
    print(f"LDA: ppl={rl['ho_perplexity']:.1f} ARI={rl['reinit_ari']:.3f} "
          f"coh={rl['coherence']:.3f}", flush=True)

    np.savez_compressed(ld / "viterbi_labels.npz", **labels)
    (ld / "ladder_summary.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    L.comp_fig(results)
    print("DONE")


if __name__ == "__main__":
    main()
