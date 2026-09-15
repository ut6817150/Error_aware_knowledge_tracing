"""Transition integration, the grouped two-channel suppression on learn.

Joint-trained BKT whose transition carries the two grouped misconception
chains, conceptual and procedural, as separate additive suppressors of
the learn rate, the emission left plain BKT:

    P(learn at t) = learn_k * (1 - gamma_conceptual * m_c
                                 - gamma_procedural * m_p)
    P(correct | mastered)   = 1 - s_k
    P(correct | unmastered) = g_k

with the suppression factor floored at a clip, under pinned gammas the
two terms can exceed one and learning truncates to the floor. Each gamma
is a relative suppression of the learn rate per unit of its group's
state, descriptive, learn and the gammas are coupled through the same
expected counts and compensate, and the two raw gammas are not on a
common scale, the group states' distributions differ, so comparisons
read at contribution level, gamma times mean state.

The chains are GroupedChains under the chain of record, fitted first and
frozen, their two filtered states before each turn are the inputs,
solution row consumed by the chains, never by BKT. pin_gamma pins both
gammas, 0 gives the engine's own BKT. Alignment, protocol, and
evaluation mirror transition_integration_pooled.py, and a fix to either
belongs in both.
"""

import ast

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from scripts.chain import TriggerChain
from scripts.misconception_chains_grouped import GroupedChains
from scripts.emission_integration_pooled import CHAIN_OF_RECORD

CLIP = 1e-4


class TransitionIntegrationGrouped:
    """The two grouped states on the learn rate, named gammas."""

    def __init__(self, train_data, test_data, chains=None, seed=221,
                 max_iterations=100, tolerance=1e-3, pin_gamma=None):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        if chains is None:
            chains = GroupedChains(train_data, test_data,
                                   chain_class=TriggerChain,
                                   chain_kwargs=dict(CHAIN_OF_RECORD))
            chains.run()
        self.chains = chains  # fitted, frozen, never refit here
        rng = np.random.RandomState(seed)
        rng_gamma = np.random.RandomState(seed + 1)
        self.pin_gamma = pin_gamma
        # gammas draw from their own stream so every gamma mode, pinned or
        # free, gives identical per-KC initializations
        draw = lambda: (float(pin_gamma) if pin_gamma is not None
                        else 0.4 + 0.2 * rng_gamma.random_sample())
        self.gamma_conceptual = draw()
        self.gamma_procedural = draw()
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
        """Per dialogue, per KC, ordered (correct, m_c, m_p, position)
        over real turns, the two states before each turn."""
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
            track = states[d]  # (units, 2), conceptual then procedural
            per_kc = {}
            for _, row in group.iterrows():
                idx = int(row["_compact"])
                if idx >= len(track):
                    raise ValueError(
                        f"dialogue {d}: retained turn at compact position "
                        f"{idx} beyond {len(track)} chain states, the BKT "
                        f"frame and the chain sequence disagree")
                m_c, m_p = float(track[idx, 0]), float(track[idx, 1])
                for kc in ast.literal_eval(row["kcs"]):
                    per_kc.setdefault(kc, []).append(
                        (int(row["_correct"]), m_c, m_p,
                         int(row["_compact"])))
            out[d] = per_kc
        return out

    # ------------------------------------------------------------ transition
    def _learn_at(self, kc_params, m_c, m_p):
        """The suppressed learn rate after a turn with states m_c, m_p."""
        factor = 1.0 - self.gamma_conceptual * m_c \
            - self.gamma_procedural * m_p
        return max(kc_params["learn"] * factor, CLIP)

    # ------------------------------------------------------------------- EM
    def fit(self):
        previous = None
        for _ in range(self.max_iterations):
            stats = {k: np.zeros(2) for k in self.parameters}
            emit = {k: [] for k in self.parameters}
            trans = {k: [] for k in self.parameters}
            log_likelihood = 0.0
            for per_kc in self.train.values():
                for kc, seq in per_kc.items():
                    p = self.parameters[kc]
                    n = len(seq)
                    pm, pu = 1 - p["slip"], p["guess"]
                    alpha = np.zeros((n, 2)); scale = np.zeros(n)
                    emis = np.zeros((n, 2))
                    Ts = np.zeros((n, 2, 2))
                    for t, (c, m_c, m_p, _) in enumerate(seq):
                        emis[t] = (pu, pm) if c else (1 - pu, 1 - pm)
                        learn = self._learn_at(p, m_c, m_p)
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
                        trans[kc].append((xi[0, 1], gamma_p[t, 0],
                                          seq[t][1], seq[t][2]))
                    for t, (c, m_c, m_p, _) in enumerate(seq):
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
            rows = np.array(emit[kc])
            g0, g1, c = rows.T
            p["guess"] = clip((g0 * c).sum() / max(g0.sum(), 1e-9))
            p["slip"] = clip((g1 * (1 - c)).sum() / max(g1.sum(), 1e-9))
            trans_by_kc[kc] = np.array(trans[kc])  # xi01, g0, m_c, m_p
        for kc, p in self.parameters.items():
            rows = trans_by_kc[kc]
            if len(rows) == 0:
                continue
            p["learn"] = clip(self._scalar_fit(
                lambda v, r=rows: self._transition_ll(r, learn=v),
                p["learn"]))
        if self.pin_gamma is not None:
            return
        for name in ("gamma_conceptual", "gamma_procedural"):
            def objective(v, name=name):
                held = getattr(self, name)
                setattr(self, name, v)
                value = sum(self._transition_ll(
                    trans_by_kc[k], learn=self.parameters[k]["learn"])
                    for k in trans_by_kc if len(trans_by_kc[k]))
                setattr(self, name, held)
                return value
            setattr(self, name,
                    clip(self._scalar_fit(objective, getattr(self, name))))

    def _transition_ll(self, rows, learn):
        """Expected transition log-likelihood under the current gammas."""
        xi01, g0, m_c, m_p = rows.T
        factor = 1.0 - self.gamma_conceptual * m_c \
            - self.gamma_procedural * m_p
        learn_t = np.clip(learn * factor, CLIP, 1 - CLIP)
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
        """Per dialogue and turn, the mean over KCs of P(correct), the
        update walking the suppressed transition. Unseen KCs contribute
        0.5."""
        rows = []
        for d, per_kc in sequences.items():
            turn_scores = {}
            for kc, seq in per_kc.items():
                p = self.parameters.get(kc)
                if p is None:
                    for c, _, _, pos in seq:
                        turn_scores.setdefault(pos, []).append((0.5, c))
                    continue
                pm, pu = 1 - p["slip"], p["guess"]
                state = p["prior"]
                for c, m_c, m_p, pos in seq:
                    prediction = state * pm + (1 - state) * pu
                    turn_scores.setdefault(pos, []).append((prediction, c))
                    observed = pm if c else 1 - pm
                    other = pu if c else 1 - pu
                    posterior = state * observed / max(
                        state * observed + (1 - state) * other, 1e-12)
                    state = posterior + (1 - posterior) \
                        * self._learn_at(p, m_c, m_p)
            for pos, pairs in turn_scores.items():
                rows.append({
                    "dialogue_id": d, "turn_index": pos,
                    "prediction": float(np.mean([x for x, _ in pairs])),
                    "correct": int(round(np.mean([y for _, y in pairs]))),
                })
        return pd.DataFrame(rows)

    def evaluate(self):
        """Paper-aligned, first scored turn updates and is excluded,
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
            "gamma_conceptual": round(float(self.gamma_conceptual), 4),
            "gamma_procedural": round(float(self.gamma_procedural), 4),
        }
        return self.metrics

    def run(self):
        self.fit()
        return self.evaluate()
