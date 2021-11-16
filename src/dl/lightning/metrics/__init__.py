from .metric_callbacks import Accuracy, MeanIoU


METRIC_LOOKUP = {
    "miou": "MeanIoU",
    "acc": "Accuracy"
}


__all__ = ["METRIC_LOOKUP", "Accuracy", "MeanIoU"]