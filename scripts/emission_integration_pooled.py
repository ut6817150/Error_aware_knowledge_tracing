"""Emission integration, F1, the pooled scalar channel beside slip.

Joint-trained BKT whose emission carries a second, parallel channel from
the misconception chains. The chains are fitted first, on the annotation
cells alone with the solution row as their first observation, and stay
frozen. At every real turn the five filtered states, conditioned on
strictly earlier units only, are pooled to one scalar m, so the m
entering turn 1 is the chains' post-solution state. BKT itself trains
from turn 1, the paper's population, the solution row is never a BKT
observation. The channel is additive, competing risks with disjoint causes, every
error has one cause, ordinary slip or belief capture, so the error
probabilities add and beta reads as the belief-caused share of error.

The branch wiring is a class attribute set by the three subclasses,
EmissionIntegrationPooledMastered places the channel beside slip only,

    P(correct | mastered)   = 1 - s_k - beta * m_t
    P(correct | unmastered) = g_k

Unmastered subtracts its own beta_unmastered * m_t from the guess
branch instead. Both wires both branches with separate betas, fitted
independently, the mastered and unmastered capture rates need not agree.
pin_beta pins every wired beta to the given value. Probabilities are clipped at the floor, so where beta * m exceeds
the branch's own success mass, capture saturates the branch to certain
incorrectness, the truncation the falsification cells will exhibit.

Training is joint EM per KC with the time-varying emission. Prior and
learn keep closed-form M-steps, guess stays closed form when the channel
does not touch its branch, slip per KC and the one global beta are
coupled and fitted by coordinate ascent on the expected complete-data
log-likelihood. Correctness is the only training target.
"""

import ast

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from scripts.chain import TriggerChain
from scripts.misconception_chains import MisconceptionChains
from scripts.pooling import Pooling

CLIP = 1e-4
# the ruled chain-of-record configuration from notebook 03, strong snap
CHAIN_OF_RECORD = {"pi": 0.05, "onset": 1e-6, "pP_i": 0.02, "pP_a": 0.97}


class EmissionIntegrationPooled:
    """F1 parent. Subclasses set CONNECTION to wire the channel's branch."""

    CONNECTION = None

    def __init__(self, train_data, test_data, chains=None, pooling="max",
                 seed=221, max_iterations=100, tolerance=1e-3,
                 pin_beta=None):
        if self.CONNECTION not in ("mastered", "unmastered", "both"):
            raise TypeError("instantiate a connection subclass, "
                            "Mastered, Unmastered, or Both")
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
        self.pin_beta = pin_beta
        self.beta_mastered = (pin_beta if pin_beta is not None
                              else 0.4 + 0.2 * rng.random_sample())
        self.beta_unmastered = (pin_beta if pin_beta is not None
                                else 0.4 + 0.2 * rng.random_sample())
        self.train = self._sequences(train_data)
        self.test = self._sequences(test_data)
        kcs = sorted({k for seqs in self.train.values() for k in seqs})
        # seeded random initialization per KC, the baseline's convention,
        # so a beta-pinned-to-zero run is a like-for-like BKT refit
        self.parameters = {k: {"prior": 0.2 + 0.4 * rng.random_sample(),
                               "learn": 0.1 + 0.3 * rng.random_sample(),
                               "guess": 0.1 + 0.3 * rng.random_sample(),
                               "slip": 0.05 + 0.2 * rng.random_sample()}
                           for k in kcs}
        self.metrics = {}

    # ------------------------------------------------------------------ data
    def _sequences(self, dataframe):
        """Per dialogue, per KC, ordered (correct, m) pairs over real turns.

        The chains filter the full unit sequence, solution row first, and
        the m paired with turn k is their state before that turn, so turn
        1 consumes the post-solution state. BKT sequences start at turn 1.
        """
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
        # compact position within the retained sequence, the chain array is
        # solution at 0 then retained turns in order, so the k-th retained
        # turn reads chain state k regardless of gaps in original numbering
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
                m_t = float(m[idx])  # state before this retained turn
                for kc in ast.literal_eval(row["kcs"]):
                    per_kc.setdefault(kc, []).append(
                        (int(row["_correct"]), m_t, int(row["_compact"])))
            out[d] = per_kc
        return out

    # -------------------------------------------------------------- emission
    def _p_correct(self, kc_params, m):
        """(P(correct | mastered), P(correct | unmastered)) at level m."""
        mastered = 1.0 - kc_params["slip"]
        unmastered = kc_params["guess"]
        if self.CONNECTION in ("mastered", "both"):
            mastered = mastered - self.beta_mastered * m
        if self.CONNECTION in ("unmastered", "both"):
            unmastered = unmastered - self.beta_unmastered * m
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
                    for t, (c, m, _) in enumerate(seq):
                        pm, pu = self._p_correct(p, m)
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
                    for t, (c, m, _) in enumerate(seq):
                        weighted[kc].append((gamma[t, 0], gamma[t, 1], c, m))
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
            rows = np.array(weighted[kc])  # gamma0, gamma1, correct, m
            rows_by_kc[kc] = rows
            p["prior"] = clip(s[0] / max(s[1], 1e-9))
            p["learn"] = clip(s[2] / max(s[3], 1e-9))
            g0, _, c, _ = rows.T
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
        if self.pin_beta is None:
            if self.CONNECTION in ("mastered", "both"):
                self.beta_mastered = clip(self._scalar_fit(
                    lambda v: sum(self._emission_ll(
                        rows_by_kc[k], beta_mastered=v,
                        guess=self.parameters[k]["guess"],
                        slip=self.parameters[k]["slip"])
                        for k in rows_by_kc), self.beta_mastered))
            if self.CONNECTION in ("unmastered", "both"):
                self.beta_unmastered = clip(self._scalar_fit(
                    lambda v: sum(self._emission_ll(
                        rows_by_kc[k], beta_unmastered=v,
                        guess=self.parameters[k]["guess"],
                        slip=self.parameters[k]["slip"])
                        for k in rows_by_kc), self.beta_unmastered))

    def _emission_ll(self, rows, guess, slip, beta_mastered=None,
                     beta_unmastered=None):
        """Expected complete-data log-likelihood of the emission block."""
        bm = self.beta_mastered if beta_mastered is None else beta_mastered
        bu = self.beta_unmastered if beta_unmastered is None else beta_unmastered
        g0, g1, c, m = rows.T
        pm = 1.0 - slip - (bm * m if self.CONNECTION in ("mastered", "both")
                           else 0.0)
        pu = guess - (bu * m if self.CONNECTION in ("unmastered", "both")
                      else 0.0)
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
        filtered mastery before each observation."""
        rows = []
        for d, per_kc in sequences.items():
            turn_scores = {}
            for kc, seq in per_kc.items():
                p = self.parameters.get(kc)
                if p is None:
                    # unseen KC, fixed 0.5, no state, no gate, the baseline
                    for c, _, pos in seq:
                        turn_scores.setdefault(pos, []).append((0.5, c))
                    continue
                state = p["prior"]
                for c, m, pos in seq:
                    pm, pu = self._p_correct(p, m)
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
        """Paper-aligned, per-KC predictions averaged to true turns,
        metrics on the test turns, the frozen M1 protocol."""
        frame = self.predict(self.test)
        # paper alignment, each dialogue's first scored turn updates the
        # filter but is excluded from metrics, the frozen M1 protocol
        first = frame.groupby("dialogue_id")["turn_index"].transform("min")
        frame = frame[frame["turn_index"] > first]
        labels = frame["correct"].to_numpy()
        scores = frame["prediction"].to_numpy()
        predicted = np.round(scores) == 1  # 0.5 rounds to 0, the baseline
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
            "beta_mastered": round(float(self.beta_mastered), 4)
                if self.CONNECTION in ("mastered", "both") else None,
            "beta_unmastered": round(float(self.beta_unmastered), 4)
                if self.CONNECTION in ("unmastered", "both") else None,
            "pooling": self.pooling,
            "connection": self.CONNECTION,
        }
        return self.metrics

    def run(self):
        self.fit()
        return self.evaluate()


class EmissionIntegrationPooledMastered(EmissionIntegrationPooled):
    """The channel beside slip, mastered branch only."""
    CONNECTION = "mastered"


class EmissionIntegrationPooledUnmastered(EmissionIntegrationPooled):
    """The channel on the guess branch only, the falsification cell."""
    CONNECTION = "unmastered"


class EmissionIntegrationPooledBoth(EmissionIntegrationPooled):
    """Capture ignores mastery, the channel on both branches."""
    CONNECTION = "both"