"""ssvision: the surface and mass estimates of "Quantitative Vision for Solid Sampling at the Milligram Scale".

    ssvision.surface   side-view mask -> upper boundary (40 bins) -> RANSAC line -> surface height and tilt
    ssvision.mass      top-view mask -> pixel count -> m = a * n_px ** b; fit a and b, with leave-one-out errors
    ssvision.segment   optional: the two masks from SAM 3, with the paper's prompts and thresholds (needs torch and
                       transformers; not imported here)
    ssvision.sam3_pipeline  the SAM 3 processing of the service that produced the paper's masks, which
                       ssvision.segment calls (docs/segmentation_service.md)
"""
from .mass import fit_power_law, mass_from_pixels, pixel_count, power_law_fit
from .surface import estimate_surface, surface_line

__version__ = "1.0.0"
__all__ = ["estimate_surface", "surface_line", "pixel_count", "mass_from_pixels", "fit_power_law", "power_law_fit"]
