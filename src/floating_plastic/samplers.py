"""Deterministic task-specific sampling helpers."""

import numpy as np


class OriginalPriorCyclicSampler:
    """Traverse every supplied FloPWD record once per seeded shuffled cycle.

    No class balancing, severity filtering, or severity weighting is applied.
    A requested batch may end mid-cycle; subsequent requests continue at the
    next index, and each completed cycle contains every original record once.
    """

    def __init__(self, records, seed=42):
        ordered = sorted(records, key=lambda record: record.filename)
        if not ordered:
            raise ValueError("severity sampler requires FloPWD training records")
        if len({record.filename for record in ordered}) != len(ordered):
            raise ValueError("severity sampler records must have unique filenames")
        if any(record.source_domain != "flopwd" for record in ordered):
            raise ValueError("severity stream may contain only FloPWD records")
        if any(not record.severity_target_available for record in ordered):
            raise ValueError("severity stream requires genuine available FloPWD targets")
        self.records = ordered
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)
        self._permutation = np.asarray([], dtype=int)
        self._position = 0
        self.draw_count = 0
        self.completed_cycles = 0

    def next_indices(self, count):
        count = int(count)
        if count < 0:
            raise ValueError("severity batch size cannot be negative")
        selected = []
        while len(selected) < count:
            if self._position >= len(self._permutation):
                self._permutation = self._rng.permutation(len(self.records))
                self._position = 0
                if self.draw_count:
                    self.completed_cycles += 1
            take = min(count - len(selected), len(self._permutation) - self._position)
            selected.extend(self._permutation[self._position:self._position + take].tolist())
            self._position += take
        self.draw_count += count
        return np.asarray(selected, dtype=int)

    def next_records(self, count):
        return [self.records[int(index)] for index in self.next_indices(count)]

    def state_dict(self):
        """Return JSON-safe state for exact continuation across checkpoints."""
        return {
            "seed": self.seed,
            "filenames": [record.filename for record in self.records],
            "rng_state": self._rng.bit_generator.state,
            "permutation": self._permutation.tolist(),
            "position": self._position,
            "draw_count": self.draw_count,
            "completed_cycles": self.completed_cycles,
        }

    def load_state_dict(self, state):
        """Restore a state produced for this exact ordered training population."""
        expected = [record.filename for record in self.records]
        if state.get("seed") != self.seed or state.get("filenames") != expected:
            raise ValueError("severity sampler checkpoint does not match seed/population")
        permutation = np.asarray(state.get("permutation", []), dtype=int)
        if len(permutation) and sorted(permutation.tolist()) != list(range(len(self.records))):
            raise ValueError("severity sampler checkpoint has an invalid permutation")
        position = int(state.get("position", 0))
        if not 0 <= position <= len(permutation):
            raise ValueError("severity sampler checkpoint has an invalid position")
        self._rng.bit_generator.state = state["rng_state"]
        self._permutation = permutation
        self._position = position
        self.draw_count = int(state.get("draw_count", 0))
        self.completed_cycles = int(state.get("completed_cycles", 0))


def severity_distribution(records):
    values = np.asarray([float(record.severity_target) for record in records], dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("severity distribution requires finite labeled records")
    return {
        "sample_count": int(len(values)), "mean": float(values.mean()),
        "median": float(np.median(values)), "standard_deviation": float(values.std()),
        "zero_count": int(np.sum(values == 0)),
        "zero_percentage": float(100 * np.mean(values == 0)),
        "minimum": float(values.min()), "maximum": float(values.max()),
        "p25": float(np.percentile(values, 25)), "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
    }
