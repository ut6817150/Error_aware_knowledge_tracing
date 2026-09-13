"""A NumPy reimplementation of the paper's pyBKT baseline."""

from ast import literal_eval

import numpy as np
import pandas as pd


class BKTModel:
    """Train and evaluate the paper's four-parameter BKT model for each KC."""

    def __init__(self, train_data, test_data, seed=221):
        self.train_data = self._prepare_data(train_data)
        self.test_data = self._prepare_data(test_data)
        self.seed = seed
        self.random_state = np.random.RandomState(seed)
        self.parameters = {}
        self.metrics = {}

    def _prepare_data(self, dataframe):
        data = dataframe.copy()
        turn_zero = (
            data["turn"]
            .astype("string")
            .str.strip()
            .str.lower()
            .isin(["solution", "turn 0", "0"])
        )
        data = data[~turn_zero].copy()
        data["correct"] = (
            data["correct"]
            .astype("string")
            .str.strip()
            .str.lower()
            .map({"true": 1, "false": 0})
        )
        data["kcs"] = data["kcs"].map(
            lambda value: literal_eval(value) if isinstance(value, str) else value
        )
        data = data.dropna(subset=["correct"]).explode("kcs")
        data = data.dropna(subset=["kcs"]).rename(columns={"kcs": "kc"})
        data["turn_number"] = pd.to_numeric(
            data["turn"].astype("string").str.extract(r"(\d+)")[0]
        )
        data["correct"] = data["correct"].astype(int)

        return data[["dialogue_id", "turn_number", "correct", "kc"]].sort_values(
            ["dialogue_id", "turn_number"]
        )

    def _initial_parameters(self):
        """Draw one initialization using pyBKT 1.4.1's seeded procedure."""
        # pyBKT's random_model_uni draws four Dirichlet-distributed arrays before
        # overwriting the four scalar BKT parameters below. Repeating those draws
        # preserves its seeded random-number sequence.
        self.random_state.gamma(
            np.tile(np.transpose([[20, 4], [1, 20]]), (1, 1)).reshape((1, 2, 2)),
            1,
        )
        self.random_state.gamma(np.array([[5], [0.5]]), 1)
        self.random_state.gamma(np.array([[0.5], [5]]), 1)
        self.random_state.gamma(np.array([[100], [1]]), 1)

        return (
            self.random_state.random_sample(),
            self.random_state.random_sample() * 0.40,
            self.random_state.random_sample() * 0.40,
            self.random_state.random_sample() * 0.30,
        )

    def _fit_parameters(self, sequences, max_iterations=100, tolerance=0.005):
        """Run one pyBKT-style EM fit for a single KC."""
        prior, learn, guess, slip = self._initial_parameters()
        previous_log_likelihood = None

        for iteration in range(max_iterations):
            prior_sum = learn_sum = unmastered_sum = 0
            guess_sum = guess_total = slip_sum = slip_total = 0
            log_likelihood = 0.0

            transition = np.array([[1 - learn, learn], [0, 1]])

            for sequence in sequences:
                observations = np.asarray(sequence, dtype=int)
                emissions = np.column_stack(
                    [
                        np.where(observations == 1, guess, 1 - guess),
                        np.where(observations == 1, 1 - slip, slip),
                    ]
                )

                alpha = np.zeros((len(observations), 2))
                scales = np.zeros(len(observations))
                alpha[0] = np.array([1 - prior, prior]) * emissions[0]
                scales[0] = alpha[0].sum()
                alpha[0] /= scales[0]

                for turn in range(1, len(observations)):
                    alpha[turn] = (alpha[turn - 1] @ transition) * emissions[turn]
                    scales[turn] = alpha[turn].sum()
                    alpha[turn] /= scales[turn]

                log_likelihood += np.log(scales).sum()

                beta = np.ones((len(observations), 2))
                for turn in range(len(observations) - 2, -1, -1):
                    beta[turn] = transition @ (
                        emissions[turn + 1] * beta[turn + 1]
                    )
                    beta[turn] /= scales[turn + 1]

                gamma = alpha * beta
                gamma /= gamma.sum(axis=1, keepdims=True)

                prior_sum += gamma[0, 1]
                guess_sum += (gamma[:, 0] * observations).sum()
                guess_total += gamma[:, 0].sum()
                slip_sum += (gamma[:, 1] * (1 - observations)).sum()
                slip_total += gamma[:, 1].sum()

                for turn in range(len(observations) - 1):
                    transitions = (
                        alpha[turn, :, None]
                        * transition
                        * (emissions[turn + 1] * beta[turn + 1])[None, :]
                    )
                    transitions /= transitions.sum()
                    learn_sum += transitions[0, 1]
                    unmastered_sum += gamma[turn, 0]

            # pyBKT checks the likelihood before the next M-step and only starts
            # checking after three likelihood evaluations.
            if (
                iteration > 1
                and previous_log_likelihood is not None
                and abs(log_likelihood - previous_log_likelihood) <= tolerance
            ):
                break
            previous_log_likelihood = log_likelihood

            prior = prior_sum / len(sequences)
            # These zero-count defaults reproduce pyBKT's M-step normalization.
            learn = learn_sum / unmastered_sum if unmastered_sum else 1.0
            guess = guess_sum / guess_total if guess_total else 0.5
            slip = slip_sum / slip_total if slip_total else 0.5

        return {
            "prior": float(prior),
            "learn": float(learn),
            "guess": float(guess),
            "slip": float(slip),
        }

    def train(self):
        """Fit every KC once, matching ``BKT(seed=221, num_fits=1)``."""
        sequences_by_kc = {}

        for kc, kc_data in self.train_data.groupby("kc", sort=False):
            sequences = [
                dialogue["correct"].tolist()
                for _, dialogue in kc_data.groupby("dialogue_id", sort=False)
            ]
            sequences_by_kc[kc] = sequences

        self.parameters = {
            kc: self._fit_parameters(sequences)
            for kc, sequences in sequences_by_kc.items()
        }
        return self

    def predict(self):
        """Predict test correctness and average KC predictions by true turn."""
        if not self.parameters:
            raise ValueError("Call train() before predict().")

        mastery = {}
        predictions = []

        for row in self.test_data.itertuples(index=False):
            parameters = self.parameters.get(row.kc)
            if parameters is None:
                # pyBKT initializes prediction columns to 0.5 and only overwrites
                # rows belonging to skills observed during training.
                predictions.append(
                    {
                        "dialogue_id": row.dialogue_id,
                        "turn_number": row.turn_number,
                        "correct": row.correct,
                        "probability": 0.5,
                    }
                )
                continue

            key = (row.dialogue_id, row.kc)
            probability_mastered = mastery.get(key, parameters["prior"])
            probability_correct = (
                probability_mastered * (1 - parameters["slip"])
                + (1 - probability_mastered) * parameters["guess"]
            )
            predictions.append(
                {
                    "dialogue_id": row.dialogue_id,
                    "turn_number": row.turn_number,
                    "correct": row.correct,
                    "probability": probability_correct,
                }
            )

            if row.correct and probability_correct > 0:
                probability_mastered *= (1 - parameters["slip"]) / probability_correct
            elif not row.correct and probability_correct < 1:
                probability_mastered *= parameters["slip"] / (1 - probability_correct)
            # A degenerate fitted KC can assign zero probability to the observed
            # response. Its Bayesian posterior is then undefined; retaining the
            # pre-observation state is the smallest finite numerical safeguard.
            mastery[key] = probability_mastered + (
                1 - probability_mastered
            ) * parameters["learn"]

        predictions = pd.DataFrame(predictions)
        return (
            predictions.groupby(["dialogue_id", "turn_number"], as_index=False)
            .agg(correct=("correct", "first"), probability=("probability", "mean"))
            .sort_values(["dialogue_id", "turn_number"])
        )

    def _calculate_metrics(self, predictions):
        labels = predictions["correct"].to_numpy()
        probabilities = predictions["probability"].to_numpy()
        predicted_labels = np.round(probabilities).astype(int)

        accuracy = (predicted_labels == labels).mean()
        true_positive = ((predicted_labels == 1) & (labels == 1)).sum()
        false_positive = ((predicted_labels == 1) & (labels == 0)).sum()
        false_negative = ((predicted_labels == 0) & (labels == 1)).sum()
        f1 = 2 * true_positive / (2 * true_positive + false_positive + false_negative)

        ranks = pd.Series(probabilities).rank().to_numpy()
        positive = labels.sum()
        negative = len(labels) - positive
        auc = (ranks[labels == 1].sum() - positive * (positive + 1) / 2) / (
            positive * negative
        )

        return {
            "Accuracy": float(accuracy),
            "AUC": float(auc),
            "F1": float(f1),
        }

    def standard_evaluation(self):
        """Evaluate predictions on every retained tagged test turn."""
        self.metrics["standard_evaluation"] = self._calculate_metrics(self.predict())
        return self.metrics["standard_evaluation"]

    def paper_aligned_evaluation(self):
        """Evaluate after using, but not scoring, each dialogue's first turn."""
        predictions = self.predict()
        first_turns = predictions.groupby("dialogue_id")["turn_number"].idxmin()
        predictions = predictions.drop(index=first_turns)
        self.metrics["paper_aligned_evaluation"] = self._calculate_metrics(predictions)
        return self.metrics["paper_aligned_evaluation"]

    def run(self):
        """Train, run both evaluations, and return the stored metrics."""
        self.train()
        self.standard_evaluation()
        self.paper_aligned_evaluation()
        return self.metrics
