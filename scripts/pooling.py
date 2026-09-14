"""Pooling, combining the misconception chains' P(present) into one scalar.

Each method takes a (units, chains) array of filtered P(present) values,
one column per chain, and returns one value per unit. Every method is a
stated hypothesis about how misconception risks combine, named so a
results row can cite it, and Pooling.get(name) resolves a method by
string so every integration face shares this one registry.
"""

import numpy as np


class Pooling:
    """Registry of pooling functions over chain states."""

    @staticmethod
    def max(states):
        """The strongest chain rules. A second live belief adds nothing,
        risks are nested rather than compounding. Immune to floor noise,
        five unevidenced chains contribute only their largest floor."""
        return states.max(axis=1)

    @staticmethod
    def noisy_or(states):
        """Independent risks compound. The probability at least one belief
        fires, one minus the product of complements. Never below max, and
        inflates under floor noise, five chains at 0.07 pool to 0.30."""
        return 1.0 - np.prod(1.0 - states, axis=1)

    @staticmethod
    def top2_or(states):
        """Compounding without floor inflation. Noisy-OR over the two
        strongest chains only, keeps the two-live-beliefs effect while the
        remaining floors contribute nothing."""
        top2 = np.sort(states, axis=1)[:, -2:]
        return 1.0 - np.prod(1.0 - top2, axis=1)

    @staticmethod
    def mean(states):
        """Risk as average load. Dilutes, one live belief among four quiet
        chains reads near 0.2, which contradicts the any-belief-suffices
        reading, included as the predicted-to-fail row."""
        return states.mean(axis=1)

    @staticmethod
    def second_max(states):
        """The second-strongest chain. Not a gate candidate, the
        stratification variable for the co-elevation diagnostic, how live
        the second belief is at each unit."""
        return np.sort(states, axis=1)[:, -2]

    @classmethod
    def get(cls, name):
        method = getattr(cls, name, None)
        if method is None or name.startswith("_") or name == "get":
            raise ValueError(f"unknown pooling '{name}', available: "
                             f"{', '.join(cls.available())}")
        return method

    @classmethod
    def available(cls):
        return [m for m in ("max", "noisy_or", "top2_or", "mean",
                            "second_max")]
