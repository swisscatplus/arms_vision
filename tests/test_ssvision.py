"""The package on synthetic input with a known answer."""
import numpy as np
import pytest

from ssvision import estimate_surface, fit_power_law, mass_from_pixels, pixel_count

# a vertical vial axis: the lip-end reference pixel at y = 50, the other at y = 350, 300 px for 30 mm (0.1 mm/px)
CLICKS = [(200, 50), (200, 350)]


def material_below(slope: float) -> np.ndarray:
    """Material between x = 100 and 300 below the line y = 250 + slope (x - 200), down to y = 380."""
    y, x = np.mgrid[0:400, 0:400]
    return (x >= 100) & (x <= 300) & (y >= 250 + slope * (x - 200)) & (y < 380)


def test_level_surface():
    res = estimate_surface(material_below(0.0), CLICKS, vial_length_mm=30.0)
    assert res["depth_mm"] == pytest.approx(-20.0)          # 200 px below the lip-end pixel, at 0.1 mm/px
    assert res["tilt_deg"] == pytest.approx(0.0, abs=1e-9)
    assert np.allclose(res["p_img"], (200, 250))


def test_tilted_surface_and_crop():
    mask = material_below(0.5)
    res = estimate_surface(mask, CLICKS, vial_length_mm=30.0)
    assert res["tilt_deg"] == pytest.approx(np.degrees(np.arctan(0.5)), abs=0.5)
    assert res["depth_mm"] == pytest.approx(-20.0, abs=0.1)
    crop = estimate_surface(mask[200:390, 90:310], CLICKS, vial_length_mm=30.0, origin=(90, 200))
    assert crop["depth_mm"] == res["depth_mm"] and crop["tilt_deg"] == res["tilt_deg"]


def test_power_law():
    n_px = np.array([300.0, 1000, 2500, 6000, 12000])
    a, b = fit_power_law(n_px, 2e-4 * n_px ** 1.3)
    assert (a, b) == (pytest.approx(2e-4), pytest.approx(1.3))
    assert mass_from_pixels(pixel_count(np.ones((10, 10))), a, b) == pytest.approx(2e-4 * 100 ** 1.3)
    assert mass_from_pixels(0, a, b) == 0.0
