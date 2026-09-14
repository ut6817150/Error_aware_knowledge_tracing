"""Emission integration, F2, the grouped two-vector channel beside slip.

Joint-trained BKT whose emission carries the two grouped misconception
chains, conceptual and procedural, as separate additive channels beside
slip, every capture rate a named scalar parameter:

    P(correct | mastered)   = 1 - s_k - beta_mastered_conceptual * m_c
                                       - beta_mastered_procedural * m_p
    P(correct | unmastered) = g_k - beta_unmastered_conceptual * m_c
                                  - beta_unmastered_procedural * m_p

on the wired branches, clipped at the floor. No pooling anywhere, the
additive sum is the combination rule. The chains are GroupedChains under
the chain of record, fitted first and frozen, their two filtered states
before each turn are the channel inputs, solution row consumed by the
chains, never by BKT. The three connection subclasses wire mastered,
unmastered, or both, wired betas fitted independently by coordinate
ascent, pin_beta pins every wired beta.

This class is deliberately standalone rather than a child of the F1
face, separation over sharing, so the EM here mirrors
emission_integration_pooled.py, and a fix to either belongs in both.
"""

import ast

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from scripts.chain import TriggerChain
from scripts.misconception_chains_grouped import GroupedChains
from scripts.emission_integration_pooled import CHAIN_OF_RECORD

CLIP = 1e-4


class EmissionIntegrationGrouped:
    """F2 parent. Subclasses set CONNECTION to wire the channel's branch."""

    CONNECTION = None

    def __init__(self, train_data, test_data, chains=None, seed=221,
                 max_iterations=100, tolerance=1e-3, pin_beta=None):
        if self.CONNECTION not in ("mastered", "unmastered", "both"):
            raise TypeError("instantiate a connection subclass, "
                            "Mastered, Unmastered, or Both")
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        if chains is None:
            chains = GroupedChains(train_data, test_data,
                                   chain_class=TriggerChain,
                                   chain_kwargs=dict(CHAIN_OF_RECORD))
            chains.run()
        self.chains = chains  # fitted, frozen, never refit here
        rng = np.random.RandomState(seed)
        self.pin_beta = pin_beta
        draw = lambda: (float(pin_beta) if pin_beta is not None
                        else 0.4 + 0.2 * rng.random_sample())
        self.beta_mastered_conceptual = draw()
        self.beta_mastered_procedural = draw()
        self.beta_unmastered_conceptual = draw()
        self.beta_unmastered_procedural = draw()
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
        """Per dialogue, per KC, ordered (correct, m_c, m_p, position) over
        real turns. The chains filter the full unit sequence, solution row
        first, and the states paired with turn k are theirs before that
        turn, so turn 1 consumes the post-solution states. BKT sequences
        start at turn 1."""
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

    # -------------------------------------------------------------- emission
    def _p_correct(self, kc_params, m_c, m_p):
        """(P(correct | mastered), P(correct | unmastered))."""
        mastered = 1.0 - kc_params["slip"]
        unmastered = kc_params["guess"]
        if self.CONNECTION in ("mastered", "both"):
            mastered = (mastered
                        - self.beta_mastered_conceptual * m_c
                        - self.beta_mastered_procedural * m_p)
        if self.CONNECTION in ("unmastered", "both"):
            unmastered = (unmastered
                          - self.beta_unmastered_conceptual * m_c
                          - self.beta_unmastered_procedural * m_p)
        return max(mastered, CLIP), max(unmastered, CLIP)

    # ------------------------------------------------------------------- EM
    def fit(self):
        previous = None
        for _ in range(self.max_iterations):
            stats = {k: np.zeros(4) for k in self.parameters}
            weighted = {k: [] for k in self.parameters}
            log_likelihood = 0.0
            for per_kc in self.train.values():
                for kc, seq in per_kc.items():
                    p = self.parameters[kc]
                    n = len(seq)
                    alpha = np.zeros((n, 2)); scale = np.zeros(n)
                    state = np.array([1 - p["prior"], p["prior"]])
                    T = np.array([[1 - p["learn"], p["learn"]],
                                  [0.0, 1.0]])
                    emis = np.zeros((n, 2))
                    for t, (c, m_c, m_p, _) in enumerate(seq):
                        pm, pu = self._p_correct(p, m_c, m_p)
                        emis[t] = (pu, pm) if c else (1 - pu, 1 - pm)
                        state = alpha[t - 1] @ T if t else state
                        alpha[t] = state * emis[t]
                        scale[t] = alpha[t].sum(); alpha[t] /= scale[t]
                    log_likelihood += np.log(scale).sum()
                    back = np.ones((n, 2))
                    for t in range(n - 2, -1, -1):
                        back[t] = T @ (emis[t + 1] * back[t + 1]) / scale[t + 1]
                    gamma = alpha * back
                    gamma /= gamma.sum(axis=1, keepdims=True)
                    stats[kc][0] += gamma[0, 1]; stats[kc][1] += 1
                    for t in range(n - 1):
                        xi = alpha[t][:, None] * T * (emis[t + 1] * back[t + 1])
                        xi /= xi.sum()
                        stats[kc][2] += xi[0, 1]
                        stats[kc][3] += gamma[t, 0]
                    for t, (c, m_c, m_p, _) in enumerate(seq):
                        weighted[kc].append(
                            (gamma[t, 0], gamma[t, 1], c, m_c, m_p))
            if previous is not None and \
                    abs(log_likelihood - previous) <= self.tolerance:
                break
            previous = log_likelihood
            self._maximize(stats, weighted)
        self.log_likelihood = float(log_likelihood)
        return self

    def _maximize(self, stats, weighted):
        clip = lambda x: float(np.clip(x, CLIP, 1 - CLIP))
        rows_by_kc = {}
        for kc, p in self.parameters.items():
            s = stats[kc]
            rows = np.array(weighted[kc])  # g0, g1, correct, m_c, m_p
            rows_by_kc[kc] = rows
            p["prior"] = clip(s[0] / max(s[1], 1e-9))
            p["learn"] = clip(s[2] / max(s[3], 1e-9))
            g0, c = rows[:, 0], rows[:, 2]
            if self.CONNECTION == "mastered":
                p["guess"] = clip((g0 * c).sum() / max(g0.sum(), 1e-9))
            else:
                p["guess"] = clip(self._scalar_fit(
                    lambda v: self._emission_ll(rows, guess=v,
                                                slip=p["slip"]),
                    p["guess"]))
            p["slip"] = clip(self._scalar_fit(
                lambda v: self._emission_ll(rows, guess=p["guess"], slip=v),
                p["slip"]))
        if self.pin_beta is not None:
            return
        wired = []
        if self.CONNECTION in ("mastered", "both"):
            wired += ["beta_mastered_conceptual", "beta_mastered_procedural"]
        if self.CONNECTION in ("unmastered", "both"):
            wired += ["beta_unmastered_conceptual",
                      "beta_unmastered_procedural"]
        for name in wired:
            def objective(v, name=name):
                held = getattr(self, name)
                setattr(self, name, v)
                value = sum(self._emission_ll(
                    rows_by_kc[k], guess=self.parameters[k]["guess"],
                    slip=self.parameters[k]["slip"]) for k in rows_by_kc)
                setattr(self, name, held)
                return value
            setattr(self, name,
                    clip(self._scalar_fit(objective, getattr(self, name))))

    def _emission_ll(self, rows, guess, slip):
        """Expected complete-data log-likelihood of the emission block,
        under the current beta attributes."""
        g0, g1, c = rows[:, 0], rows[:, 1], rows[:, 2]
        m_c, m_p = rows[:, 3], rows[:, 4]
        pm = 1.0 - slip
        pu = guess
        if self.CONNECTION in ("mastered", "both"):
            pm = pm - (self.beta_mastered_conceptual * m_c
                       + self.beta_mastered_procedural * m_p)
        if self.CONNECTION in ("unmastered", "both"):
            pu = pu - (self.beta_unmastered_conceptual * m_c
                       + self.beta_unmastered_procedural * m_p)
        pm = np.clip(pm, CLIP, 1 - CLIP)
        pu = np.clip(pu, CLIP, 1 - CLIP)
        return float((g1 * (c * np.log(pm) + (1 - c) * np.log(1 - pm))
                      + g0 * (c * np.log(pu) + (1 - c) * np.log(1 - pu))
                      ).sum())

    @staticmethod
    def _scalar_fit(objective, current):
        result = minimize_scalar(lambda v: -objective(v),
                                 bounds=(CLIP, 1 - CLIP), method="bounded")
        return result.x if result.success else current

    # ------------------------------------------------------------ prediction
    def predict(self, sequences):
        """Per dialogue and turn, the mean over KCs of P(correct), from the
        filtered mastery before each observation. Unseen KCs contribute
        0.5, no state, no channel."""
        rows = []
        for d, per_kc in sequences.items():
            turn_scores = {}
            for kc, seq in per_kc.items():
                p = self.parameters.get(kc)
                if p is None:
                    for c, _, _, pos in seq:
                        turn_scores.setdefault(pos, []).append((0.5, c))
                    continue
                state = p["prior"]
                for c, m_c, m_p, pos in seq:
                    pm, pu = self._p_correct(p, m_c, m_p)
                    prediction = state * pm + (1 - state) * pu
                    turn_scores.setdefault(pos, []).append((prediction, c))
                    observed = pm if c else 1 - pm
                    other = pu if c else 1 - pu
                    posterior = state * observed / max(
                        state * observed + (1 - state) * other, 1e-12)
                    state = posterior + (1 - posterior) * p["learn"]
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
        wired = self.CONNECTION
        self.metrics = {
            "accuracy": round(float((predicted == labels).mean()), 4),
            "auc": round(float((ranks[labels == 1].sum()
                                - pos * (pos + 1) / 2) / (pos * neg)), 4),
            "f1": round(float(2 * tp / max(
                2 * tp + (predicted & (labels == 0)).sum()
                + (~predicted & (labels == 1)).sum(), 1)), 4),
            "turns": int(len(labels)),
            "beta_mastered_conceptual":
                round(self.beta_mastered_conceptual, 4)
                if wired in ("mastered", "both") else None,
            "beta_mastered_procedural":
                round(self.beta_mastered_procedural, 4)
                if wired in ("mastered", "both") else None,
            "beta_unmastered_conceptual":
                round(self.beta_unmastered_conceptual, 4)
                if wired in ("unmastered", "both") else None,
            "beta_unmastered_procedural":
                round(self.beta_unmastered_procedural, 4)
                if wired in ("unmastered", "both") else None,
            "connection": wired,
        }
        return self.metrics

    def run(self):
        self.fit()
        return self.evaluate()


class EmissionIntegrationGroupedMastered(EmissionIntegrationGrouped):
    """Both group channels beside slip, mastered branch only."""
    CONNECTION = "mastered"


class EmissionIntegrationGroupedUnmastered(EmissionIntegrationGrouped):
    """Both group channels on the guess branch only."""
    CONNECTION = "unmastered"


class EmissionIntegrationGroupedBoth(EmissionIntegrationGrouped):
    """Both group channels on both branches, four betas."""
    CONNECTION = "both"