"""Defines an example abstract domain using interval matrices."""
import numpy as np

class IntervalDomain:
    """An Interval-matrix abstract weight set domain."""
    def __init__(self):
        """Initialize the IntervalDomain."""
        self.is_convex = True

    def alpha(self, matrices):
        """Compute the interval abstraction of a set of matrices."""
        matrices = np.asarray(list(matrices))
        lower = np.min(matrices, axis=0)
        upper = np.max(matrices, axis=0)
        return IntervalMatrix(lower, upper)

class IntervalMatrix:
    """An interval matrix, where each entry is an interval.

    We represent the interval matrix with two 'regular' matrices, one which
    contains only the lower-bounds of the intervals and the other representing
    the upper bounds.
    """
    def __init__(self, lower, upper):
        """Initialize an IntervalMatrix given lower, upper bounds."""
        self.lower = lower
        self.upper = upper
        assert np.all(lower <= upper) or (lower == np.inf and upper == -np.inf)

    def __eq__(self, other):
        """True iff @self and @other represent the same Interval Matrix.

        NOTE: Be careful when relying on this method for matrices that might be
        bottom. It works as long as the convention that bottom is always
        represented as lower = -np.inf, upper = np.inf is followed. Currently
        this is asserted on construction of the IntervalMatrix.
        """
        return (np.all(self.lower == other.lower)
                and np.all(self.upper == other.upper))

    def __str__(self):
        """Human-readable version of the interval matrix."""
        def formatted_row(i):
            return ", ".join(f"[{l}, {u}]"
                             for l, u in zip(self.lower[i], self.upper[i]))
        return ("["
                + "\n ".join(map(formatted_row, range(self.lower.shape[0])))
                + "]")
