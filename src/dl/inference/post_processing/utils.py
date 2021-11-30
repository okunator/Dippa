from ._base._base_processor import PostProcessor

from . import *


post_proc = vars()


def post_processor(name: str, **kwargs) -> PostProcessor:
    """
    Initialize the post-processor
    
    Args:
    -----------
        name (str):
            The name of the post-processor. Use lowercase letters.

    Returns:
    -----------
        PostProcessor: Initialized post-processor.
    """
    allowed = list(post_proc["POST_PROC_LOOKUP"].keys())
    assert name in allowed, (
        f"Illegal post-processor. Got {name}. Allowed ones: {allowed}"
    )

    kwargs = kwargs.copy()
    key = post_proc["POST_PROC_LOOKUP"][name]
    post_proccer = post_proc[key](**kwargs)

    return post_proccer


__all__ = ["post_processor"]
