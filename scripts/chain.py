"""A two-state hidden Markov chain for one misconception family.

States are inactive (0) and active (1), the belief not held versus held.
Observations are symbol sequences coded N=0, A=1, P=2, one sequence per
dialogue. P is evidence for active, A for inactive, and N emits nothing,
the likelihood is one for both states so the state only moves through the
transition. Fitting is EM via forward-backward. Prediction uses the
forward pass alone, so the state at each unit conditions on strictly
earlier units only.

Parameters, all plain attributes:
    pi       P(active) at the first unit, before any evidence
    onset    P(inactive -> active) per unit
    resolve  P(active -> inactive) per unit
    pP_a     P(P | active, informative), the exhibition rate
    pP_i     P(P | inactive, informative), the false-alarm rate
"""

import numpy as np
import pandas as pd

PRIOR = 2.0  # asymmetric emission pseudo-counts, break label symmetry


class Chain:
    """One misconception chain. Subclasses constrain parameters via PINS."""

    PINS = {}

    def __init__(self, seed=221, max_iterations=200, tolerance=1e-4,
                 ceiling=None):
        self.pins = dict(self.PINS)
        self.ceiling = ceiling
        self.seed = seed
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        rng = np.random.RandomState(seed)
        self.pi = 0.4 + 0.2 * rng.random_sample()
        self.onset = 0.02 + 0.05 * rng.random_sample()
        self.resolve = 0.05 + 0.15 * rng.random_sample()
        self.pP_a = 0.8 + 0.1 * rng.random_sample()
        self.pP_i = 0.05 + 0.1 * rng.random_sample()
        self._pin()
        self.log_likelihood = None
        self.metrics = {}

    def _pin(self):
        for name, value in self.pins.items():
            setattr(self, name, value)

    # ------------------------------------------------------------ model parts
    def emission_matrix(self):
        """Rows inactive, active; columns N, A, P. N is one for both states."""
        return np.array([[1.0, 1 - self.pP_i, self.pP_i],
                         [1.0, 1 - self.pP_a, self.pP_a]])

    def transition_matrix(self):
        return np.array([[1 - self.onset, self.onset],
                         [self.resolve, 1 - self.resolve]])

    # ------------------------------------------------------------ inference
    def forward(self, symbols):
        """Scaled forward pass. Returns alpha (n, 2), scales (n,), and
        predicted (n,), the P(active) at each unit given strictly earlier
        units, the causally valid quantity for prediction and for the gate."""
        E, T = self.emission_matrix(), self.transition_matrix()
        n = len(symbols)
        alpha = np.zeros((n, 2))
        scales = np.zeros(n)
        predicted = np.zeros(n)
        state = np.array([1 - self.pi, self.pi])
        for t in range(n):
            state = alpha[t - 1] @ T if t else state
            predicted[t] = state[1]
            alpha[t] = state * E[:, symbols[t]]
            scales[t] = alpha[t].sum()
            alpha[t] /= scales[t]
            if self.ceiling is not None and alpha[t, 1] > self.ceiling:
                # bounded confidence, the state never exceeds the ceiling, so
                # stacked P evidence cannot buy immunity to a later A
                alpha[t] = np.array([1 - self.ceiling, self.ceiling])
        return alpha, scales, predicted

    def backward(self, symbols, scales):
        """Scaled backward pass, matching the forward scaling."""
        E, T = self.emission_matrix(), self.transition_matrix()
        n = len(symbols)
        beta = np.ones((n, 2))
        for t in range(n - 2, -1, -1):
            beta[t] = T @ (E[:, symbols[t + 1]] * beta[t + 1]) / scales[t + 1]
        return beta

    def predict_state(self, symbols):
        """P(active) before each unit, forward pass only."""
        return self.forward(symbols)[2]

    def predict_evidence(self, symbols):
        """One-step-ahead P(next informative cell shows P) at each unit."""
        state = self.predict_state(symbols)
        return state * self.pP_a + (1 - state) * self.pP_i

    # ------------------------------------------------------------ EM fitting
    def expectation(self, symbols):
        """Smoothed posteriors for one sequence, gamma (n, 2) and the
        pairwise xi summed over transitions (2, 2), plus the sequence
        log-likelihood. Used only inside fitting, never for prediction."""
        E, T = self.emission_matrix(), self.transition_matrix()
        alpha, scales, _ = self.forward(symbols)
        beta = self.backward(symbols, scales)
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True)
        xi_sum = np.zeros((2, 2))
        for t in range(len(symbols) - 1):
            xi = alpha[t][:, None] * T * (E[:, symbols[t + 1]] * beta[t + 1])
            xi_sum += xi / xi.sum()
        return gamma, xi_sum, float(np.log(scales).sum())

    def maximization(self, counts, emission_counts):
        """Closed-form M-step from pooled expected counts, then re-pin."""
        clip = lambda x, hi=1 - 1e-4: float(np.clip(x, 1e-4, hi))
        self.pi = clip(counts["pi_num"] / counts["sequences"])
        self.onset = clip(counts["onset_num"] / max(counts["inactive"], 1e-9), 0.5)
        self.resolve = clip(counts["resolve_num"] / max(counts["active"], 1e-9), 0.9)
        self.pP_i = clip(emission_counts[0, 1] / emission_counts[0].sum())
        self.pP_a = clip(emission_counts[1, 1] / emission_counts[1].sum())
        self._pin()

    def fit(self, sequences):
        """EM over a list of symbol sequences until the likelihood converges."""
        previous = None
        for _ in range(self.max_iterations):
            counts = {"pi_num": 0.0, "sequences": 0.0, "onset_num": 0.0,
                      "inactive": 0.0, "resolve_num": 0.0, "active": 0.0}
            emission_counts = np.array([[0.8 * PRIOR, 0.2 * PRIOR],   # inactive: A, P
                                        [0.2 * PRIOR, 0.8 * PRIOR]])  # active:   A, P
            log_likelihood = 0.0
            for symbols in sequences:
                gamma, xi_sum, sequence_ll = self.expectation(symbols)
                log_likelihood += sequence_ll
                counts["pi_num"] += gamma[0, 1]
                counts["sequences"] += 1
                counts["onset_num"] += xi_sum[0, 1]
                counts["inactive"] += gamma[:-1, 0].sum()
                counts["resolve_num"] += xi_sum[1, 0]
                counts["active"] += gamma[:-1, 1].sum()
                for t, symbol in enumerate(symbols):
                    if symbol:
                        emission_counts[:, symbol - 1] += gamma[t]
            if previous is not None and abs(log_likelihood - previous) <= self.tolerance:
                break
            previous = log_likelihood
            self.maximization(counts, emission_counts)
        self.log_likelihood = float(log_likelihood)
        return self

    # ------------------------------------------------------------ evaluation
    def evaluate(self, sequences):
        """Score one-step-ahead evidence prediction on held-out sequences.

        Every informative cell is predicted from the state before it, P is
        the positive class. Returns accuracy, tpr (recall on P cells), tnr
        (recall on A cells), auc, and f1, also stored on self.metrics."""
        labels, scores = [], []
        for symbols in sequences:
            q = self.predict_evidence(symbols)
            for t, symbol in enumerate(symbols):
                if symbol:
                    labels.append(1 if symbol == 2 else 0)
                    scores.append(q[t])
        labels, scores = np.array(labels), np.array(scores)
        predicted = scores >= 0.5
        pos, neg = labels.sum(), (labels == 0).sum()
        ranks = pd.Series(scores).rank().to_numpy()
        tp = (predicted & (labels == 1)).sum()
        auc = np.nan if not pos or not neg else \
            (ranks[labels == 1].sum() - pos * (pos + 1) / 2) / (pos * neg)
        self.metrics = {
            "accuracy": round(float((predicted == labels).mean()), 4)
                if len(labels) else np.nan,
            "tpr": round(float(predicted[labels == 1].mean()), 4)
                if pos else np.nan,
            "tnr": round(float((~predicted[labels == 0]).mean()), 4)
                if neg else np.nan,
            "auc": round(float(auc), 4) if pos and neg else np.nan,
            "f1": round(float(2 * tp / max(2 * tp + (predicted & (labels == 0)).sum()
                                           + (~predicted & (labels == 1)).sum(), 1)), 4),
            "informative_cells": int(len(labels)),
        }
        return self.metrics

    @classmethod
    def from_frame(cls, dataframe, column, **kwargs):
        """Build and fit a chain straight from a dataframe and column title."""
        return cls(**kwargs).fit(list(extract_sequences(dataframe, column).values()))

    def parameters(self):
        return {"pi": round(self.pi, 4), "onset": round(self.onset, 4),
                "resolve": round(self.resolve, 4), "pP_i": round(self.pP_i, 4),
                "pP_a": round(self.pP_a, 4),
                "expected_dwell_turns": round(1 / self.resolve, 1)}


class TriggerChain(Chain):
    """A chain whose parameters can be pinned individually by name.

    There are no preset pins, TriggerChain() pins nothing and behaves
    exactly like Chain, every constraint is stated explicitly at the call
    site. Pass a number to pin that parameter at that value through every
    EM iteration, leave it None to let it fit freely. The configuration
    that makes this a trigger chain, inactive until the first P, is
    pi at a small epsilon, onset at 1e-6, and pP_i anchored low, spelled
    out wherever it is used."""

    def __init__(self, pi=None, onset=None, resolve=None, pP_i=None,
                 pP_a=None, **kwargs):
        super().__init__(**kwargs)
        values = {"pi": pi, "onset": onset, "resolve": resolve,
                  "pP_i": pP_i, "pP_a": pP_a}
        self.pins = {name: value for name, value in values.items()
                     if value is not None}
        self._pin()


def extract_sequences(dataframe, column):
    """The one parser. Per dialogue, units ordered solution first then by
    turn number, the named column coded N=0, A=1, P=2. Returns a dict
    dialogue_id -> integer array."""
    data = dataframe.copy()
    turn = data["turn"].astype("string").str.strip().str.lower()
    data["_pos"] = pd.to_numeric(turn.str.extract(r"(\d+)")[0], errors="coerce")
    data.loc[turn.eq("solution"), "_pos"] = -1
    data = data.dropna(subset=["_pos"]).sort_values(["dialogue_id", "_pos"])
    symbols = (data[column].astype("string").str.strip().str.upper()
               .map({"N": 0, "A": 1, "P": 2}))
    data = data.assign(_sym=symbols).dropna(subset=["_sym"])
    return {d: g["_sym"].to_numpy(int)
            for d, g in data.groupby("dialogue_id", sort=False)}