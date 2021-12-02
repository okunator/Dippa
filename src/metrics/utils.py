from typing import Callable

# metrics = vars()

from .metrics import (
    PQ, AJI, AJI_plus, DICE2,
    conventional_metrics, split_and_merge
)

METRIC_LOOKUP = {
    "pq": PQ,
    "aji": AJI,
    "aji+": AJI_plus,
    "dice2": DICE2,
    "conventional": conventional_metrics,
    "split_and_merge": split_and_merge
}


def benchmark_metric(name: str) -> Callable:
    """
    Initialize a benchmark metric function.

    Args:
    -----------
        name (str):
            The name of the metric. Use lowercase letters.

    Returns:
    -----------
        Callable: The metric function
    """
    # allowed = list(metrics["METRIC_LOOKUP"].keys())
    allowed = list(METRIC_LOOKUP.keys())
    assert name in allowed, (
        f"Illegal metric given. Got: {name}. Allowed ones: {allowed}"
    )
    
    # key = METRIC_LOOKUP[name]
    # metric = metrics[key]
    metric = METRIC_LOOKUP[name]

    return metric


__all__ = ["benchmark_metric"]