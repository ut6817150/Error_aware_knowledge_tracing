"""Pooled misconception chains, one chain per group of families.

PooledChains merges each group's member cells into one column, priority
P over A over N under the coding N=0, A=1, P=2, the group latent reading
"any member belief is active." A P from any member is an existence proof
and wins outright. The documented compromise, a lone member's A becomes
the whole group's A, evidence of inactivity for a disjunction whose other
members may have gone unprobed; requiring all-member A's would starve the
pooled chains of absence evidence entirely.

Everything else, fitting, evaluation, summaries, predict_states, and the
chains stored as attributes, model.conceptual and model.procedural, is
inherited unchanged from MisconceptionChains.
"""

import numpy as np
import pandas as pd

from scripts.misconception_chains import FAMILIES, MisconceptionChains

GROUPS = {
    "conceptual": ["comprehension", "relevance", "principles"],
    "procedural": ["wrong_operation", "steps"],
}
_CODE = {"N": 0, "A": 1, "P": 2}
_BACK = {0: "N", 1: "A", 2: "P"}


class GroupedChains(MisconceptionChains):
    """One chain per group, columns built by the P over A over N merge."""

    GROUPS = GROUPS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.families = list(self.GROUPS)

    def reformat(self, dataframe):
        data = dataframe.copy()
        coded = {
            family: data[family].astype("string").str.strip().str.upper()
            .map(_CODE).fillna(0).astype(int)
            for family in FAMILIES if family in data.columns
        }
        for group, members in self.GROUPS.items():
            merged = np.maximum.reduce([coded[m].to_numpy() for m in members])
            data[group] = pd.Series(merged, index=data.index).map(_BACK)
        return data

