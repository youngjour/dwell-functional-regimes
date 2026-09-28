"""Batched Gaussian explicit-duration HSMM (EDHSMM) for fixed-length sequences.

- Diagonal-covariance Gaussian emissions.
- Per-state duration ~ Gamma (discretised & renormalised over 1..Dmax) -> "Gamma
  duration prior". No self-transition (A diag = 0); staying is modelled by duration.
- Training: Viterbi-EM (hard EM) — robust & tractable for K=7, T=19.
- Scoring: right-censored forward log-likelihood (last segment uses the duration
  survival function S_k(d)=P(D>=d)), so per-day diurnal cycles are not penalised
  for ending mid-state.

All sequences share length T (built by src/08). Shapes: X (N, T, D).
Vectorised over N; loops over t (<=19) and d (<=19).
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp
from sklearn.cluster import KMeans

NEG = -1e30


class EDHSMM:
    def __init__(self, n_states, n_iter=25, tol=1e-2, min_covar=1e-3,
                 dur_max=None, random_state=0, verbose=False):
        self.K = n_states
        self.n_iter = n_iter
        self.tol = tol
        self.min_covar = min_covar
        self.dur_max = dur_max
        self.rs = random_state
        self.verbose = verbose

    # ---- emission log-prob (N,T,K) ----
    def _log_b(self, X):
        N, T, D = X.shape
        mu, var = self.means_, self.vars_
        # (N,T,K): -0.5*sum_d[log(2pi var)+ (x-mu)^2/var]
        cst = -0.5 * (D * np.log(2 * np.pi) + np.log(var).sum(axis=1))  # (K,)
        x2 = X[:, :, None, :]                                           # (N,T,1,D)
        q = ((x2 - mu[None, None]) ** 2 / var[None, None]).sum(-1)      # (N,T,K)
        return cst[None, None] - 0.5 * q

    def _seg_emit(self, logB):
        # cumB[n,t,k] = sum_{u<=t} logB ; seg(s..t)=cumB[t]-cumB[s-1]
        cum = np.cumsum(logB, axis=1)
        z = np.zeros((logB.shape[0], 1, self.K))
        return np.concatenate([z, cum], axis=1)  # (N,T+1,K); seg(s..t)=cum[t+1]-cum[s]

    # ---- init ----
    def _init(self, X):
        N, T, D = X.shape
        flat = X.reshape(-1, D)
        rng = np.random.default_rng(self.rs)
        idx = rng.choice(flat.shape[0], min(60000, flat.shape[0]), replace=False)
        km = KMeans(self.K, n_init=3, random_state=self.rs).fit(flat[idx])
        self.means_ = km.cluster_centers_.astype(np.float64)
        self.vars_ = np.tile(flat.var(0) + self.min_covar, (self.K, 1))
        self.startprob_ = np.full(self.K, 1.0 / self.K)
        A = np.ones((self.K, self.K)) - np.eye(self.K)
        self.transmat_ = A / A.sum(1, keepdims=True)
        self.Dmax = self.dur_max or T
        # init duration: Gamma mean ~3
        self._set_duration(np.full(self.K, 3.0), np.full(self.K, 2.0))

    def _set_duration(self, mean, var):
        d = np.arange(1, self.Dmax + 1)
        from scipy.stats import gamma
        pd = np.zeros((self.K, self.Dmax))
        for k in range(self.K):
            m = max(mean[k], 1.05); v = max(var[k], 0.25)
            scale = v / m; shape = m / scale
            self.dur_shape_[k] = shape if hasattr(self, "dur_shape_") else shape
            w = gamma.pdf(d, a=shape, scale=scale)
            w = np.where(np.isfinite(w) & (w > 0), w, 1e-12)
            pd[k] = w / w.sum()
        self.dur_pmf_ = np.clip(pd, 1e-12, None)
        self.dur_pmf_ /= self.dur_pmf_.sum(1, keepdims=True)
        self.dur_surv_ = np.cumsum(self.dur_pmf_[:, ::-1], axis=1)[:, ::-1]  # P(D>=d)
        self.dur_mean_ = (np.arange(1, self.Dmax + 1)[None] * self.dur_pmf_).sum(1)

    # ---- Viterbi (labels) ----
    def _viterbi(self, X):
        logB = self._log_b(X)
        N, T, K = logB.shape
        segc = self._seg_emit(logB)
        logpd = np.log(self.dur_pmf_); logA = np.log(self.transmat_ + 1e-300)
        logpi = np.log(self.startprob_ + 1e-300)
        np.fill_diagonal(logA, NEG)
        delta = np.full((N, T, K), NEG)
        bptr_d = np.zeros((N, T, K), np.int32)
        bptr_i = np.zeros((N, T, K), np.int32)
        for t in range(T):
            dmax = min(t + 1, self.Dmax)
            for d in range(1, dmax + 1):
                s = t - d + 1
                seg = segc[:, t + 1, :] - segc[:, s, :]            # (N,K)
                cand = seg + logpd[:, d - 1][None, :]              # (N,K)
                if s == 0:
                    sc = cand + logpi[None, :]
                    better = sc > delta[:, t, :]
                    delta[:, t, :] = np.where(better, sc, delta[:, t, :])
                    bptr_d[:, t, :] = np.where(better, d, bptr_d[:, t, :])
                    bptr_i[:, t, :] = np.where(better, -1, bptr_i[:, t, :])
                else:
                    prev = delta[:, s - 1, :]                       # (N,K_i)
                    m = prev[:, :, None] + logA[None]               # (N,i,j)
                    best_i = np.argmax(m, axis=1)                   # (N,j)
                    best_v = np.max(m, axis=1)                      # (N,j)
                    sc = cand + best_v
                    better = sc > delta[:, t, :]
                    delta[:, t, :] = np.where(better, sc, delta[:, t, :])
                    bptr_d[:, t, :] = np.where(better, d, bptr_d[:, t, :])
                    bptr_i[:, t, :] = np.where(better, best_i, bptr_i[:, t, :])
        labels = np.zeros((N, T), np.int32)
        ar = np.arange(N)
        score = np.max(delta[:, T - 1, :], axis=1)
        state = np.argmax(delta[:, T - 1, :], axis=1)
        endpos = np.full(N, T - 1)
        rem = bptr_d[ar, T - 1, state].copy()           # remaining len of cur seg
        for t in range(T - 1, -1, -1):
            labels[:, t] = state
            rem -= 1
            new_seg = (rem == 0) & (t > 0)
            if t > 0:
                prev_state = bptr_i[ar, endpos, state]
                ns = np.where(new_seg & (prev_state >= 0), prev_state, state)
                nend = np.where(new_seg, t - 1, endpos)
                nrem = np.where(new_seg, bptr_d[ar, np.clip(t - 1, 0, None), ns], rem)
                state, endpos, rem = ns, nend, nrem
        return labels, score

    # ---- right-censored forward log-lik (per seq) ----
    def _forward_ll(self, X):
        logB = self._log_b(X)
        N, T, K = logB.shape
        segc = self._seg_emit(logB)
        logpd = np.log(self.dur_pmf_); logsurv = np.log(self.dur_surv_)
        logA = np.log(self.transmat_ + 1e-300); np.fill_diagonal(logA, NEG)
        logpi = np.log(self.startprob_ + 1e-300)
        alpha = np.full((N, T, K), NEG)   # completed-segment ends at t in k
        for t in range(T):
            dmax = min(t + 1, self.Dmax)
            terms = []
            for d in range(1, dmax + 1):
                s = t - d + 1
                seg = segc[:, t + 1, :] - segc[:, s, :]
                if s == 0:
                    inflow = logpi[None, :]
                else:
                    inflow = logsumexp(alpha[:, s - 1, :][:, :, None] + logA[None],
                                       axis=1)
                terms.append(seg + logpd[:, d - 1][None, :] + inflow)
            alpha[:, t, :] = logsumexp(np.stack(terms, 0), axis=0)
        # censored termination: last segment ends at T-1 incomplete (use survival)
        cens = []
        for d in range(1, self.Dmax + 1):
            s = T - d
            if s < 0:
                break
            seg = segc[:, T, :] - segc[:, s, :]
            if s == 0:
                inflow = logpi[None, :]
            else:
                inflow = logsumexp(alpha[:, s - 1, :][:, :, None] + logA[None], axis=1)
            cens.append(seg + logsurv[:, d - 1][None, :] + inflow)
        ll = logsumexp(np.stack(cens, 0), axis=(0, 2))   # (N,)
        return ll

    def score(self, X):
        return float(self._forward_ll(X).sum())

    def predict(self, X):
        return self._viterbi(X)[0]

    # ---- Viterbi-EM ----
    def fit(self, X):
        self.dur_shape_ = np.full(self.K, 2.0)
        self._init(X)
        N, T, D = X.shape
        self.history_ = []
        prev = -np.inf
        for it in range(self.n_iter):
            labels, vit = self._viterbi(X)
            # M-step emissions
            flat = X.reshape(-1, D); lab = labels.reshape(-1)
            for k in range(self.K):
                m = lab == k
                if m.sum() > 1:
                    self.means_[k] = flat[m].mean(0)
                    self.vars_[k] = flat[m].var(0) + self.min_covar
            # transitions + start + durations from runs (vectorised)
            change = labels[:, 1:] != labels[:, :-1]            # (N,T-1)
            A = np.zeros((self.K, self.K))
            froms = labels[:, :-1][change]; tos = labels[:, 1:][change]
            np.add.at(A, (froms, tos), 1)
            np.fill_diagonal(A, 0)
            start = np.bincount(labels[:, 0], minlength=self.K).astype(float)
            seg_id = np.zeros((N, T), int)
            seg_id[:, 1:] = np.cumsum(change, axis=1)
            gkey = (np.arange(N)[:, None] * T + seg_id).ravel()
            uniq, first_idx, counts = np.unique(gkey, return_index=True,
                                                return_counts=True)
            seg_state = labels.ravel()[first_idx]
            durs = [counts[seg_state == k] for k in range(self.K)]
            rs = A.sum(1, keepdims=True)
            self.transmat_ = np.where(rs > 0, A / np.where(rs > 0, rs, 1),
                                      (np.ones((self.K, self.K)) - np.eye(self.K)) / (self.K - 1))
            self.startprob_ = (start + 1e-3) / (start.sum() + 1e-3 * self.K)
            dmean = np.array([d.mean() if len(d) > 0 else 3.0 for d in durs])
            dvar = np.array([d.var() if len(d) > 1 else 2.0 for d in durs])
            self._set_duration(dmean, np.maximum(dvar, 0.25))
            self.emp_dur_ = durs
            ll = float(vit.sum())                 # Viterbi best-path score (monitor)
            self.history_.append(ll)
            if self.verbose:
                print(f"  EDHSMM it{it} vit_ll={ll:.0f}", flush=True)
            if abs(ll - prev) < self.tol * abs(prev) and it > 2:
                break
            prev = ll
        self.converged_ = True
        self.train_ll_ = self.score(X)            # final censored forward LL
        return self

    def n_params(self, D):
        K = self.K
        emit = 2 * K * D
        trans = K * (K - 2)          # off-diag rows sum-1, diag fixed 0
        start = K - 1
        dur = 2 * K                  # Gamma shape+scale per state
        return emit + trans + start + dur
