# Derived from the original reetoolbox project (Apache License 2.0).
# Modified by Dhyey Yajnik, 2025.
#
# Fixed-parameter evaluation settings for each perturbation. These are the values used
# in the dissertation (Table 2). Constraint sweeps override one parameter at a time; see
# pfm_robustness/perturbations.py.

eval_samples = 10
eval_steps = 10
input_range = (0, 255)

eval_rotate_optimiser_params = {
    "samples": eval_samples,
    "weight_ranges": {
        "angle": (0, 360)
    },
    "input_range": input_range
}
eval_rotate_transform_params = {
}

eval_zoom_out_optimiser_params = {
    "samples": eval_samples,
    "weight_ranges": {
        "scale": (0.5, 0.95)
    },
    "input_range": input_range
}
eval_zoom_out_transform_params = {
}

sigma = 0.075
eval_hed_optimiser_params = {
    "samples": eval_samples,
    "weight_ranges": {
        "alpha": (1 - sigma, 1 + sigma),
        "beta": (-sigma, sigma)
    },
    "input_range": input_range
}
eval_hed_transform_params = {
}

eval_mean_optimiser_params = {
    "epsilon": 1,
    "steps": eval_steps,
    "constraint": "l2",
    "C": 0.10,
    "input_range": input_range
}
eval_mean_transform_params = {
    "input_range": input_range,
    "noise_range": (-0.2, 0.2)
}

eval_zoom_in_optimiser_params = {
    "samples": eval_samples,
    "weight_ranges": {
        "scale": (1.1, 2)
    },
    "input_range": input_range
}
eval_zoom_in_transform_params = {
}

eval_crop_optimiser_params = {
    "samples": eval_samples,
    "weight_ranges": {
        "top": (0, 100),
        "left": (0, 100),
        "height": (50, 150),
        "width": (50, 150)
    },
    "input_range": input_range
}
eval_crop_transform_params = {
}

eval_blur_optimiser_params = {
    "samples": eval_samples,
    "weight_ranges": {
        "corner_x": (0, 100),
        "corner_y": (0, 100),
        "height": (50, 120),
        "width": (50, 120),
        "kernel_size": (1, 10),
        "sigma": (3, 20)
    },
    "input_range": input_range
}
eval_blur_transform_params = {
}

eval_pixel_optimiser_params = {
    "epsilon": 0.1,
    "steps": eval_steps,
    "constraint": "l2",
    "C": 0.03,
    "input_range": input_range
}
eval_pixel_transform_params = {
    "input_range": input_range,
    "noise_range": (-0.15, 0.15)
}

eval_stain_optimiser_params = {
    "epsilon": 0.002,
    "steps": eval_steps,
    "constraint": "l2",
    "C": 0.1,
    "input_range": input_range
}
eval_stain_transform_params = {
    "input_range": input_range
}

stain_weight = 0.125
eval_random_stain_optimiser_params = {
    "samples": eval_samples,
    "weight_ranges": {
        "weights": (-stain_weight, stain_weight)
    },
    "input_range": input_range
}
eval_random_stain_transform_params = {
    "input_range": input_range
}

eval_jpeg_optimiser_params = {
    "samples": eval_samples,
    "weight_ranges": {
        "quality": (97.5, 100)
    },
    "input_range": input_range
}
eval_jpeg_transform_params = {
}
