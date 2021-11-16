from .squeeze_excite import SqueezeAndExcite

ATT_LOOKUP = {
    "se": "SqueezeAndExcite",
    # "sce": "SqueezeAndCExcite",
}

__all__ = ["ATT_LOOKUP", "SqueezeAndExcite" ]