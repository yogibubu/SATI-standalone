"""Numerical truncation failures that an adaptive basis can remedy."""


class LAMTruncationError(RuntimeError):
    def __init__(self, axis, message):
        super().__init__(message)
        self.axis = axis
