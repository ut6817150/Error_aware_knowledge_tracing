"""Transition integration, the pooled scalar channel on the learn rate.

Joint-trained BKT whose transition, not emission, carries the pooled
misconception channel. At every turn the five chains' filtered states,
conditioned on the solution row and strictly earlier turns only, are
pooled to one scalar m, and the learn rate is suppressed by it:

    P(learn at t)           = learn_k * (1 - gamma * m_t)
    P(correct | mastered)   = 1 - s_k
    P(correct | unmastered) = g_k

The emission is plain BKT, untouched, so any delta against the engine
baseline is the transition's alone. gamma is one global parameter, a
relative suppression of the learn rate per unit of chain state, at
learn 0.30, gamma 0.5, and m 0.8 the effective rate is 0.18, and it is
descriptive, not a causal impasse rate, learn and gamma are coupled
through the same expected counts and compensate for one another. pin_gamma pins it, 0 gives the engine's own BKT
and must reproduce the emission-stage validity row exactly, 1 is full
suppression, a live belief at m near 1 blocks that turn's learning.

Training is joint EM with a time-varying transition. The E-step's
forward-backward carries T_t per turn, prior keeps its closed form,
guess and slip keep theirs, the emission is unconditioned, learn_k and
gamma are coupled through the expected transition counts and fit by
bounded coordinate ascent on the expected complete-data log-likelihood,
a generalized-EM step. The chains stay stage-one frozen. Alignment,
protocol, and evaluation mirror emission_integration_pooled.py, and a
fix to either belongs in both.
"""

import ast

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from scripts.chain import TriggerChain
from scripts.misconception_chains import MisconceptionChains
from scripts.pooling import Pooling
from scripts.emission_integration_pooled import CHAIN_OF_RECORD

CLIP = 1e-4


class TransitionIntegrationPooled:
    """The pooled scalar on the learn rate, one global gamma."""

    def __init__(self, train_data, test_data, chains=None, pooling="max",
                 seed=221, max_iterations=100, tolerance=1e-3,
                 pin_gamma=None):
        self.pooling = pooling
        self.pool = Pooling.get(pooling)
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        if chains is None:
            chains = MisconceptionChains(train_data, test_data,
                                         chain_class=TriggerChain,
                                         chain_kwargs=dict(CHAIN_OF_RECORD))
            chains.run()
        self.chains = chains  # fitted, frozen, never refit here
        rng = np.random.RandomState(seed)
        rng_gamma = np.random.RandomState(seed + 1)
        self.pin_gamma = pin_gamma
        # gamma draws from its own stream so every gamma mode, pinned or
        # free, gives identical per-KC initializations
        self.gamma = (float(pin_gamma) if pin_gamma is not None
                      else 0.4 + 0.2 * rng_gamma.random_sample())
        self.train = self._sequences(train_data)
        self.test = self._sequences(test_data)
        kcs = sorted({k for seqs in self.train.values() for k in seqs})
        self.parameters = {k: {"prior": 0.2 + 0.4 * rng.random_sample(),
                               "learn": 0.1 + 0.3 * rng.random_sample(),
                               "guess": 0.1 + 0.3 * rng.random_sample(),
                               "slip": 0.05 + 0.2 * rng.random_sample()}
                           for k in kcs}
        self.metrics = {}

    # ------------------------------------------------------------------ data
    def _sequences(self, dataframe):
        """Per dialogue, per KC, ordered (correct, m, position) over real
        turns. The chains filter the full unit sequence, solution row
        first, and the m paired with turn k is their pooled state before
        that turn. BKT sequences start at turn 1."""
        states = self.chains.predict_states(dataframe)
        data = dataframe.copy()
        turn = data["turn"].astype("string").str.strip().str.lower()
        data["_pos"] = pd.to_numeric(turn.str.extract(r"(\d+)")[0],
                                     errors="coerce")
        data = data.dropna(subset=["_pos"]).sort_values(
            ["dialogue_id", "_pos"])
        data["_correct"] = (data["correct"].astype("string").str.strip()
                            .str.upper().isin(["TRUE", "1", "1.0"])
                            ).astype(int)
        data["_compact"] = data.groupby("dialogue_id").cumcount() + 1
        out = {}
        for d, group in data.groupby("dialogue_id", sort=False):
            m = self.pool(states[d])
            per_kc = {}
            for _, row in group.iterrows():
                idx = int(row["_compact"])
                if idx >= len(m):
                    raise ValueError(
                        f"dialogue {d}: retained turn at compact position "
                        f"{idx} beyond {len(m)} chain states, the BKT frame "
                        f"and the chain sequence disagree")
                m_t = float(m[idx])
                for kc in ast.literal_eval(row["kcs"]):
                    per_kc.setdefault(kc, []).append(
                        (int(row["_correct"]), m_t, int(row["_compact"])))
            out[d] = per_kc
        return out

    # ------------------------------------------------------------ transition
    def _learn_at(self, kc_params, m):
        """The suppressed learn rate entering the transition after the
        turn whose channel state is m."""
        return max(kc_params["learn"] * (1.0 - self.gamma * m), CLIP)

    # ------------------------------------------------------------------- EM
    def fit(self):
        previous = None
        for _ in range(self.max_iterations):
            stats = {k: np.zeros(2) for k in self.parameters}
            emit = {k: [] for k in self.parameters}    # gamma0, gamma1, c
            trans = {k: [] for k in self.parameters}   # xi01, gamma_t0, m
            log_likelihood = 0.0
            for per_kc in self.train.values():
                for kc, seq in per_kc.items():
                    p = self.parameters[kc]
                    n = len(seq)
                    pm, pu = 1 - p["slip"], p["guess"]
                    alpha = np.zeros((n, 2)); scale = np.zeros(n)
                    emis = np.zeros((n, 2))
                    Ts = np.zeros((n, 2, 2))  # transition applied AFTER t
                    for t, (c, m, _) in enumerate(seq):
                        emis[t] = (pu, pm) if c else (1 - pu, 1 - pm)
                        learn = self._learn_at(p, m)
                        Ts[t] = np.array([[1 - learn, learn], [0.0, 1.0]])
                        state = (np.array([1 - p["prior"], p["prior"]])
                                 if t == 0 else alpha[t - 1] @ Ts[t - 1])
                        alpha[t] = state * emis[t]
                        scale[t] = alpha[t].sum(); alpha[t] /= scale[t]
                    log_likelihood += np.log(scale).sum()
                    back = np.ones((n, 2))
                    for t in range(n - 2, -1, -1):
                        back[t] = Ts[t] @ (emis[t + 1] * back[t + 1]) \
                            / scale[t + 1]
                    gamma_p = alpha * back
                    gamma_p /= gamma_p.sum(axis=1, keepdims=True)
                    stats[kc][0] += gamma_p[0, 1]; stats[kc][1] += 1
                    for t in range(n - 1):
                        xi = alpha[t][:, None] * Ts[t] \
                            * (emis[t + 1] * back[t + 1])
                        xi /= xi.sum()
                        # the transition after turn t is conditioned on
                        # turn t's channel state
                        trans[kc].append((xi[0, 1], gamma_p[t, 0],
                                          seq[t][1]))
                    for t, (c, m, _) in enumerate(seq):
                        emit[kc].append((gamma_p[t, 0], gamma_p[t, 1], c))
            if previous is not None and \
                    abs(log_likelihood - previous) <= self.tolerance:
                break
            previous = log_likelihood
            self._maximize(stats, emit, trans)
        self.log_likelihood = float(log_likelihood)
        return self

    def _maximize(self, stats, emit, trans):
        clip = lambda x: float(np.clip(x, CLIP, 1 - CLIP))
        trans_by_kc = {}
        for kc, p in self.parameters.items():
            p["prior"] = clip(stats[kc][0] / max(stats[kc][1], 1e-9))
            rows = np.array(emit[kc])  # gamma0, gamma1, correct
            g0, g1, c = rows.T
            p["guess"] = clip((g0 * c).sum() / max(g0.sum(), 1e-9))
            p["slip"] = clip((g1 * (1 - c)).sum() / max(g1.sum(), 1e-9))
            trans_by_kc[kc] = np.array(trans[kc])  # xi01, gamma_t0, m
        # learn_k given gamma, then gamma given all learn_k, each a bounded
        # 1-d maximization of the expected transition log-likelihood
        for kc, p in self.parameters.items():
            rows = trans_by_kc[kc]
            if len(rows) == 0:
                continue
            p["learn"] = clip(self._scalar_fit(
                lambda v, r=rows: self._transition_ll(r, learn=v),
                p["learn"]))
        if self.pin_gamma is None:
            self.gamma = clip(self._scalar_fit(
                lambda v: sum(self._transition_ll(
                    trans_by_kc[k], learn=self.parameters[k]["learn"],
                    gamma=v)
                    for k in trans_by_kc if len(trans_by_kc[k])),
                self.gamma))

    def _transition_ll(self, rows, learn, gamma=None):
        """Expected transition log-likelihood, xi01 * log(learn_t) +
        (gamma_t0 - xi01) * log(1 - learn_t) per step."""
        gamma = self.gamma if gamma is None else gamma
        xi01, g0, m = rows.T
        learn_t = np.clip(learn * (1.0 - gamma * m), CLIP, 1 - CLIP)
        stay = np.clip(g0 - xi01, 0.0, None)
        return float((xi01 * np.log(learn_t)
                      + stay * np.log(1 - learn_t)).sum())

    @staticmethod
    def _scalar_fit(objective, current):
        result = minimize_scalar(lambda v: -objective(v),
                                 bounds=(CLIP, 1 - CLIP), method="bounded")
        return result.x if result.success else current

    # ------------------------------------------------------------ prediction
    def predict(self, sequences):
        """Per dialogue and turn, the mean over KCs of P(correct), from the
        filtered mastery before each observation, the update walking the
        suppressed transition. Unseen KCs contribute 0.5."""
        rows = []
        for d, per_kc in sequences.items():
            turn_scores = {}
            for kc, seq in per_kc.items():
                p = self.parameters.get(kc)
                if p is None:
                    for c, _, pos in seq:
                        turn_scores.setdefault(pos, []).append((0.5, c))
                    continue
                pm, pu = 1 - p["slip"], p["guess"]
                state = p["prior"]
                for c, m, pos in seq:
                    prediction = state * pm + (1 - state) * pu
                    turn_scores.setdefault(pos, []).append((prediction, c))
                    observed = pm if c else 1 - pm
                    other = pu if c else 1 - pu
                    posterior = state * observed / max(
                        state * observed + (1 - state) * other, 1e-12)
                    state = posterior + (1 - posterior) \
                        * self._learn_at(p, m)
            for pos, pairs in turn_scores.items():
                rows.append({
                    "dialogue_id": d, "turn_index": pos,
                    "prediction": float(np.mean([x for x, _ in pairs])),
                    "correct": int(round(np.mean([y for _, y in pairs]))),
                })
        return pd.DataFrame(rows)

    def evaluate(self):
        """Paper-aligned, per-KC predictions averaged to true turns, each
        dialogue's first scored turn updates the filter and is excluded,
        exactly 0.5 classifies to 0."""
        frame = self.predict(self.test)
        first = frame.groupby("dialogue_id")["turn_index"].transform("min")
        frame = frame[frame["turn_index"] > first]
        labels = frame["correct"].to_numpy()
        scores = frame["prediction"].to_numpy()
        predicted = np.round(scores) == 1
        ranks = pd.Series(scores).rank().to_numpy()
        pos, neg = labels.sum(), (labels == 0).sum()
        tp = (predicted & (labels == 1)).sum()
        self.metrics = {
            "accuracy": round(float((predicted == labels).mean()), 4),
            "auc": round(float((ranks[labels == 1].sum()
                                - pos * (pos + 1) / 2) / (pos * neg)), 4),
            "f1": round(float(2 * tp / max(
                2 * tp + (predicted & (labels == 0)).sum()
                + (~predicted & (labels == 1)).sum(), 1)), 4),
            "turns": int(len(labels)),
            "gamma": round(float(self.gamma), 4),
            "pooling": self.pooling,
        }
        return self.metrics

    def run(self):
        self.fit()
        return self.evaluate()