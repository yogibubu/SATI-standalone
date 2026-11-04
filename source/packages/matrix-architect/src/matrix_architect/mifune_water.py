"""Compatibility facade for the MIFUNE rigid-water evaluator."""


class Tip3pFbMifuneEvaluator:
    """Lazily construct the evaluator from its new owning package."""

    def __new__(cls, *args, **kwargs):
        from matrix_mifune.zaff_water import Tip3pFbMifuneEvaluator as implementation

        return implementation(*args, **kwargs)


__all__ = ["Tip3pFbMifuneEvaluator"]
