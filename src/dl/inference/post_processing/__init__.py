from .hover.processor import HoverNetPostProcessor


POST_PROC_LOOKUP = {
    "hover":"HoverNetPostProcessor",
    "cellpose":"CellposePostProcessor",
    "dist":"DistPostProcessor",
    "contour":"ContourPostProcessor",
    "basic":"BasicPostProcessor",
}
