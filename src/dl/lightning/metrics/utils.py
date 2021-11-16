from torchmetrics import Metric

from . import *


metrics = vars()


def metric_func(name: str=None, **kwargs) -> Metric:
    """
    Initialize the metric.

    Args:
    -----------
        name (str, default=None):
            The name of the metric. Use lowercase letters.

    Returns:
    -----------
        Metric: Initialized torchmetrics Metric.
    """
    allowed = [*metrics['METRIC_LOOKUP'].keys(), None]
    assert name in allowed, (
        f"Illegal metric func given. Allowed ones: {allowed}"
    )
    
    if name is not None:
        key = metrics['METRIC_LOOKUP'][name]
        met_f = metrics[key](**kwargs)
    else:
        met_f = None

    return met_f


__all__ = ["metric_func"]