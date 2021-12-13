from ._transforms._base_transforms import (
    rigid_transforms, non_rigid_transforms, hue_saturation_transforms,
    blur_transforms, non_spatial_transforms, random_crop, center_crop,
    resize, normalize, to_tensor
)

from ._transforms._inst_transforms import (
    hover_transform, cellpose_transform, omnipose_transform, dist_transform,
    smooth_dist_transform, contour_transform, rm_borders_transform,
    edgeweight_transform, binarize_transform
)

from ._transforms._composition import apply_each, compose


AUGS_LOOKUP = {
    "rigid": "rigid_transforms",
    "non_rigid": "non_rigid_transforms",
    "hue_sat": "hue_saturation_transforms",
    "blur": "blur_transforms",
    "non_spatial": "non_spatial_transforms",
    "random_crop": "random_crop",
    "center_crop": "center_crop",
    "resize": "resize",
    "normalize": "normalize"
}


AUX_LOOKUP = {
    "hover": "hover_transform",
    "cellpose": "cellpose_transform",
    "omnipose": "omnipose_transform",
    "dist": "smooth_dist_transform",
    "contour": "contour_transform",
    "edge_weight": "edgeweight_transform",
    "binarize": "binarize_transform",
    "rm_borders": "rm_borders_transform"
}


__all__ = [
    "AUGS_LOOKUP", "AUX_LOOKUP", "rigid_transforms", "non_rigid_transforms",
    "hue_saturation_transforms", "blur_transforms", "non_spatial_transforms",
    "random_crop", "center_crop", "resize", "normalize", "to_tensor",
    "compose", "apply_each", "hover_transform", "cellpose_transform",
    "omnipose_transform", "dist_transform", "smooth_dist_transform",
    "contour_transform", "rm_borders_transform", "edgeweight_transform",
    "binarize_transform"
]
