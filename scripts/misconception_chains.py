"""Five misconception chains, one Chain per family, over one dataframe pair.

The container owns the data, it takes the train and test frames from
load_paper_filtered_data, extracts each family column's sequences through
the one shared parser, fits a Chain per family, and evaluates each on the
test set. The chains live as their own attributes, model.comprehension
through model.steps. Pass chain_class=TriggerChain for the constrained
variant.
"""

import numpy as np
import pandas as pd

from scripts.chain import Chain, extract_sequences

FAMILIES = ["comprehension", "relevance", "principles", "wrong_operation", "steps"]


class MisconceptionChains:
    """Train and evaluate one Chain per misconception family."""

    def __init__(self, train_data, test_data, chain_class=Chain, seed=221,
                 chain_kwargs=None):
        self.families = list(FAMILIES)
        self.train_data = self.reformat(train_data)
        self.test_data = self.reformat(test_data)
        self.chain_class = chain_class
        self.chain_kwargs = chain_kwargs or {}
        self.seed = seed
        self.metrics = {}

    def reformat(self, dataframe):
        """Identity hook. Subclasses reshape frames here, both the
        constructor and predict_states route every frame through it."""
        return dataframe

    def fit_chain(self, column):
        """Fit one family's chain on train, evaluate it on test, store it."""
        chain = self.chain_class(seed=self.seed + self.families.index(column),
                                 **self.chain_kwargs)
        chain.fit(list(extract_sequences(self.train_data, column).values()))
        chain.evaluate(list(extract_sequences(self.test_data, column).values()))
        setattr(self, column, chain)
        return chain

    def run(self):
        """Fit and evaluate all five chains, return the per-family metrics."""
        for column in self.families:
            self.metrics[column] = self.fit_chain(column).metrics
        return self.metrics

    def predict_states(self, dataframe=None):
        """Stacked chain states for the gate. Per dialogue, an array of shape
        (units, 5) in FAMILIES order, each column that chain's P(active)
        before the unit, conditioned on strictly earlier units only."""
        frame = self.test_data if dataframe is None else self.reformat(dataframe)
        per_family = {column: extract_sequences(frame, column)
                      for column in self.families}
        out = {}
        for d in per_family[self.families[0]]:
            out[d] = np.column_stack(
                [getattr(self, column).predict_state(per_family[column][d])
                 for column in self.families])
        return out

    def summary(self):
        """Fitted parameters and test metrics, one row per family."""
        return pd.DataFrame(
            [{"chain": column, **getattr(self, column).parameters(),
              **getattr(self, column).metrics} for column in self.families])