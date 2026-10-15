"""Perturbation definitions for the fixed-parameter evaluation and the constraint sweeps.

Names are kept exactly as in the dissertation runs because they appear in output paths:
fixed evaluation uses ``brightness``/``zoomin``/``zoomout``; sweeps use ``mean``/``zoom_in``/
``zoom_out`` (the PPI tables rename ``mean`` to ``brightness``).
"""

from dataclasses import dataclass

from .reet import constants as C
from .reet.optimisers import PGD, StochasticSearch
from .reet.transforms import (
    BlurTransform, CropTransform, HEDTransform, JPEGTransform, MeanTransform, PixelTransform,
    RandomStainTransform, RotateTransform, StainTransform, ZoomInTransform, ZoomOutTransform,
)


@dataclass(frozen=True)
class Perturbation:
    optimiser: type            # PGD (adversarial) or StochasticSearch
    transform: type
    optimiser_params: dict
    transform_params: dict


PIXEL = Perturbation(PGD, PixelTransform, C.eval_pixel_optimiser_params, C.eval_pixel_transform_params)
STAIN = Perturbation(PGD, StainTransform, C.eval_stain_optimiser_params, C.eval_stain_transform_params)
BRIGHTNESS = Perturbation(PGD, MeanTransform, C.eval_mean_optimiser_params, C.eval_mean_transform_params)
ROTATE = Perturbation(StochasticSearch, RotateTransform, C.eval_rotate_optimiser_params, C.eval_rotate_transform_params)
CROP = Perturbation(StochasticSearch, CropTransform, C.eval_crop_optimiser_params, C.eval_crop_transform_params)
BLUR = Perturbation(StochasticSearch, BlurTransform, C.eval_blur_optimiser_params, C.eval_blur_transform_params)
ZOOM_IN = Perturbation(StochasticSearch, ZoomInTransform, C.eval_zoom_in_optimiser_params, C.eval_zoom_in_transform_params)
ZOOM_OUT = Perturbation(StochasticSearch, ZoomOutTransform, C.eval_zoom_out_optimiser_params, C.eval_zoom_out_transform_params)
HED = Perturbation(StochasticSearch, HEDTransform, C.eval_hed_optimiser_params, C.eval_hed_transform_params)
RANDOM_STAIN = Perturbation(StochasticSearch, RandomStainTransform, C.eval_random_stain_optimiser_params,
                            C.eval_random_stain_transform_params)
JPEG = Perturbation(StochasticSearch, JPEGTransform, C.eval_jpeg_optimiser_params, C.eval_jpeg_transform_params)

# Fixed-parameter evaluation: name -> perturbation (order = row order of the *_stats.xlsx).
FIXED_EVAL = {
    "pixel": PIXEL,
    "stain": STAIN,
    "brightness": BRIGHTNESS,
    "rotate": ROTATE,
    "crop": CROP,
    "blur": BLUR,
    "zoomin": ZOOM_IN,
    "zoomout": ZOOM_OUT,
    "hed": HED,
    "random_stain": RANDOM_STAIN,
    "jpeg": JPEG,
}


@dataclass(frozen=True)
class Sweep:
    """Sweep `param` of `perturbation` linearly over `range` in `steps` points."""
    perturbation: Perturbation
    param: str      # see sweep.sweep_constraint for how each param is applied
    range: tuple    # (start, end); end < start sweeps downwards
    steps: int

    @property
    def step_size(self):
        return (self.range[1] - self.range[0]) / (self.steps - 1)


# Constraint sweeps (dissertation Table 1). HED and stain are not swept.
SWEEPS = {
    "pixel": Sweep(PIXEL, "C", (0.00, 0.05), 20),
    "mean": Sweep(BRIGHTNESS, "C", (0.00, 7.5), 20),
    "random_stain": Sweep(RANDOM_STAIN, "weights", (0.00, 0.5), 25),
    "jpeg": Sweep(JPEG, "quality", (90, 100), 11),
    "blur": Sweep(BLUR, "sigma", (1, 7), 21),
    "rotate": Sweep(ROTATE, "angle", (0, 360), 37),
    "zoom_in": Sweep(ZOOM_IN, "scale", (1.0, 3.0), 20),
    "zoom_out": Sweep(ZOOM_OUT, "scale", (1.0, 0.05), 20),
    "crop": Sweep(CROP, "height", (224, 50), 20),
}
