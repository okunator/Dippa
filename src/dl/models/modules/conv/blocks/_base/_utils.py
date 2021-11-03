def make_divisible(
        val: int,
        divisor: int=8,
        min_value: int=None,
        round_limit: float=.9
    ) -> float:
    """
    From: 
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/helpers.py
    """

    min_value = min_value or divisor
    new_v = max(min_value, int(val + divisor / 2) // divisor * divisor)

    # Make sure that round down does not go down by more than 10%.
    if new_v < round_limit * val:
        new_v += divisor

    return new_v