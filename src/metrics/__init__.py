from .metrics import (
    PQ, AJI, AJI_plus, DICE2,
    conventional_metrics, split_and_merge
)

from .benchmarker import Benchmarker


METRIC_LOOKUP = {
    "pq": "PQ",
    "aji": "AJI",
    "aji+": "AJI_plus",
    "dice2": "DICE2",
    "conventional": "conventional_metrics",
    "split_and_merge": "split_and_merge"
}


__all__ = [
    "METRIC_LOOKUP", "Benchmarker", "PQ", "AJI", "AJI_plus", "DICE2",
    "conventional_metrics", "split_and_merge"
]