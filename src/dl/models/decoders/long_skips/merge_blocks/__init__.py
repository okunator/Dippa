from .cat import CatBlock
from .sum import SumBlock


MERGE_LOOKUP = {
    "summation": "SumBlock",
    "concatenate": "CatBlock"
}


__all__ = ["MERGE_LOOKUP", "CatBlock", "SumBlock"]