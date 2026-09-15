"""Overlay, late fusion of the frozen BKT with the frozen chains.

Nothing here refits anything. The baseline is M1 itself, BKTModel from
bkt_model.py, imported directly, trained once if a fitted model is not
passed, its turn-level predictions frozen, and the chains supply q_t, the probability the next annotation cell shows a
present, P(misconception = present), per family:

    q = pP_i + (pP_a - pP_i) * state

pooled across families by a named Pooling method. q is used rather than
the disposition by design, the overlay identity reads the chain's own
predicted observable as an error probability, P(present) = P(incorrect).

Four fusion methods, each a pure function of the frozen prediction p
and pooled q, any fitted scalar estimated on the training predictions
by Bernoulli log-likelihood, bounded:

    pseudo_kc    p' = (K * p + (1 - q)) / (K + 1),  K = KCs on the turn
    linear_pool  p' = (1 - w) * p + w * (1 - q),    w fitted
    product      p' = p * (1 - q)
    frozen_gate  p' = p * (1 - beta * q),           beta fitted

Attribution is algebraic, no EM, no compensation, no initialization
sensitivity, the delta against the baseline is the fusion's alone, and
the baseline row reproduces M1's paper-aligned metrics exactly. Fitted
scalars estimate over all scored training turns, first turns included,
matching the population M1's own EM consumed, with zero inside the
feasible set so a method can select the baseline exactly. One stated
limitation, the training predictions are in-sample, M1 fitted on the
same outcomes, which biases the fusion toward finding signal, so a
null under this optimism is conservative, out-of-fold stacking is the
stricter design left to future work.

M1's predict() scores the test split only, so the training-side
predictions the fitted scalars need are produced by _m1_frames, a
mirror of that predict walk run over either prepared split with the
fitted parameters, kept in step with bkt_model.py, a fix to either
belongs in both. Alignment joins on the original turn_number, M1's own
key, the compact position is used only to index the chain states.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from scripts.chain import TriggerChain
from scripts.misconception_chains import MisconceptionChains
from scripts.pooling import Pooling
from scripts.bkt_model import BKTModel
from scripts.emission_integration_pooled import CHAIN_OF_RECORD

CLIP = 1e-4
METHODS = ("pseudo_kc", "linear_pool", "product", "frozen_gate")


class Overlay:
    """Late fusion over frozen baseline predictions and chain evidence."""

    def __init__(self, train_data, test_data, chains=None, m1=None,
                 seed=221):
        """m1 is a trained BKTModel, trained here if not passed. chains
        is a fitted MisconceptionChains, fitted here under the chain of
        record if not passed."""
        if chains is None:
            chains = MisconceptionChains(train_data, test_data,
                                         chain_class=TriggerChain,
                                         chain_kwargs=dict(CHAIN_OF_RECORD))
            chains.run()
        self.chains = chains
        if m1 is None:
            m1 = BKTModel(train_data, test_data, seed=seed).train()
        if not m1.parameters:
            raise ValueError("the BKTModel must be trained")
        self.m1 = m1
        # frozen turn-level predictions, train for fitting, test for scoring
        self.frames = {
            "train": self._merge(train_data, self._m1_frames("train")),
            "test": self._merge(test_data, self._m1_frames("test")),
        }
        self.fitted = {}

    def _m1_frames(self, split):
        """M1's predict walk over either prepared split, mirroring
        bkt_model.predict, unseen KCs at 0.5, per-KC filtered mastery,
        predictions averaged by true turn."""
        data = self.m1.train_data if split == "train" else self.m1.test_data
        mastery = {}
        rows = []
        for row in data.itertuples(index=False):
            parameters = self.m1.parameters.get(row.kc)
            if parameters is None:
                rows.append({"dialogue_id": row.dialogue_id,
                             "turn_index": row.turn_number,
                             "correct": row.correct, "prediction": 0.5})
                continue
            key = (row.dialogue_id, row.kc)
            mastered = mastery.get(key, parameters["prior"])
            correct_probability = (mastered * (1 - parameters["slip"])
                                   + (1 - mastered) * parameters["guess"])
            rows.append({"dialogue_id": row.dialogue_id,
                         "turn_index": row.turn_number,
                         "correct": row.correct,
                         "prediction": correct_probability})
            if row.correct and correct_probability > 0:
                mastered *= (1 - parameters["slip"]) / correct_probability
            elif not row.correct and correct_probability < 1:
                mastered *= parameters["slip"] / (1 - correct_probability)
            mastery[key] = mastered + (1 - mastered) * parameters["learn"]
        frame = pd.DataFrame(rows)
        return (frame.groupby(["dialogue_id", "turn_index"], as_index=False)
                .agg(correct=("correct", "first"),
                     prediction=("prediction", "mean"))
                .sort_values(["dialogue_id", "turn_index"]))

    # ------------------------------------------------------------------ data
    def _evidence(self, dataframe):
        """Per dialogue, the (units, families) array of q, the probability
        the next cell shows P, from each fitted chain's own emission."""
        states = self.chains.predict_states(dataframe)
        out = {}
        for d, track in states.items():
            q = np.zeros_like(track)
            for j, family in enumerate(self.chains.families):
                chain = getattr(self.chains, family)
                q[:, j] = track[:, j] * chain.pP_a \
                    + (1 - track[:, j]) * chain.pP_i
            out[d] = q
        return out

    def _merge(self, dataframe, predictions):
        """Join the frozen predictions with per-turn q columns and the
        KC count, aligned by compact position."""
        evidence = self._evidence(dataframe)
        data = dataframe.copy()
        turn = data["turn"].astype("string").str.strip().str.lower()
        data["_pos"] = pd.to_numeric(turn.str.extract(r"(\d+)")[0],
                                     errors="coerce")
        data = data.dropna(subset=["_pos"]).sort_values(
            ["dialogue_id", "_pos"])
        data["_compact"] = data.groupby("dialogue_id").cumcount() + 1
        import ast as _ast
        data["kc_count"] = data["kcs"].map(
            lambda s: len(_ast.literal_eval(s)))
        rows = []
        for _, row in data.iterrows():
            q = evidence[row["dialogue_id"]]
            idx = int(row["_compact"])
            if idx >= len(q):
                raise ValueError(
                    f"dialogue {row['dialogue_id']}: turn at compact "
                    f"position {idx} beyond {len(q)} chain states")
            rows.append({"dialogue_id": row["dialogue_id"],
                         "turn_index": int(row["_pos"]),
                         "kc_count": int(row["kc_count"]),
                         **{f"q_{f}": q[idx, j] for j, f
                            in enumerate(self.chains.families)}})
        frame = predictions.merge(pd.DataFrame(rows),
                                  on=["dialogue_id", "turn_index"])
        return frame

    def _pooled_q(self, frame, pooling):
        columns = [f"q_{f}" for f in self.chains.families]
        return Pooling.get(pooling)(frame[columns].to_numpy())

    # ---------------------------------------------------------------- fusion
    def _fuse(self, frame, method, pooling, parameter=None):
        p = frame["prediction"].to_numpy()
        q = self._pooled_q(frame, pooling)
        if method == "pseudo_kc":
            K = frame["kc_count"].to_numpy()
            fused = (K * p + (1 - q)) / (K + 1)
        elif method == "linear_pool":
            fused = (1 - parameter) * p + parameter * (1 - q)
        elif method == "product":
            fused = p * (1 - q)
        elif method == "frozen_gate":
            fused = p * (1 - parameter * q)
        else:
            raise ValueError(f"unknown method {method}")
        return np.clip(fused, CLIP, 1 - CLIP)

    def _fit_parameter(self, method, pooling):
        """The one scalar, fitted on the training predictions by
        Bernoulli log-likelihood, over all scored training turns, first
        turns included, the researcher's ruling, M1's own EM consumed
        them so every pipeline stage fits on the same population, the
        first-turn exclusion remains an evaluation convention only.
        Sensitivity is immaterial, the aligned linear-max weight moves
        0.040 to 0.030 between conventions."""
        frame = self.frames["train"]
        labels = frame["correct"].to_numpy()

        def objective(value):
            fused = self._fuse(frame, method, pooling, value)
            return -float((labels * np.log(fused)
                           + (1 - labels) * np.log(1 - fused)).sum())

        result = minimize_scalar(objective, bounds=(0.0, 1 - CLIP),
                                 method="bounded")
        # the optimizer cannot land exactly on a bound, snap to zero when
        # the boundary value is at least as good, so a fitted method can
        # select the baseline exactly
        if objective(0.0) <= objective(float(result.x)):
            return 0.0
        return float(result.x)

    # ------------------------------------------------------------ evaluation
    def evaluate(self, method, pooling="max"):
        """Paper-aligned test metrics for one fusion row, the first scored
        turn per dialogue excluded, exactly 0.5 classifies to 0."""
        parameter = None
        if method in ("linear_pool", "frozen_gate"):
            parameter = self.fitted.setdefault(
                (method, pooling), self._fit_parameter(method, pooling))
        frame = self.frames["test"].copy()
        if method == "baseline":
            frame["fused"] = frame["prediction"]
        else:
            frame["fused"] = self._fuse(frame, method, pooling, parameter)
        first = frame.groupby("dialogue_id")["turn_index"].transform("min")
        frame = frame[frame["turn_index"] > first]
        labels = frame["correct"].to_numpy()
        scores = frame["fused"].to_numpy()
        predicted = np.round(scores) == 1
        ranks = pd.Series(scores).rank().to_numpy()
        pos, neg = labels.sum(), (labels == 0).sum()
        tp = (predicted & (labels == 1)).sum()
        return {
            "accuracy": round(float((predicted == labels).mean()), 4),
            "auc": round(float((ranks[labels == 1].sum()
                                - pos * (pos + 1) / 2) / (pos * neg)), 4),
            "f1": round(float(2 * tp / max(
                2 * tp + (predicted & (labels == 0)).sum()
                + (~predicted & (labels == 1)).sum(), 1)), 4),
            "turns": int(len(labels)),
            "method": method,
            "pooling": None if method == "baseline" else pooling,
            "parameter": None if parameter is None else round(parameter, 4),
        }