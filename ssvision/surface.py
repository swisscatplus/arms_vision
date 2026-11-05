"""Surface estimate from a side-view mask of the material in the vial.

The geometry of the robot's surface observer, as it ran in the paper's experiments (Methods, Vision-Based Surface
Estimation):

  1. mask -> its largest external contour, every boundary pixel (OpenCV, CHAIN_APPROX_NONE)
  2. upper boundary: the topmost contour point in each of 40 equal-width bins of image x; a bin with fewer than
     3 contour points is dropped
  3. RANSAC line through those points: two-point samples, 600 iterations, random seed 0, inliers within 0.75 mm
     of the line, then a total-least-squares refit on the largest inlier set
  4. the fitted line is spanned across its inliers, which gives two image points on the surface
  5. scale and axis without camera calibration: two fixed reference pixels on the vial axis (click 0 at the lip
     end, click 1 on the body) and the measured length they span (57.0 mm in the paper's set-up)
  6. p = where the fitted line crosses the vial axis. Its signed distance from click 0 along the axis is the depth
     of the surface below the lip (negative = below); the angle of the line in the image, measured from the
     perpendicular to the axis, is the tilt of the surface about the vial-lip x axis (the camera's line of sight)

The functions are those of the robot code (ARMS, tools/solid_sampling/state/surface_observer.py:
compute_surface_plane, surface_edge_from_contour, _upper_boundary_points, _fit_line_ransac, _fit_line_tls,
compute_plane_pose, intersect_line_line_2d). On the stored masks of the paper's runs, estimate_surface reproduces
the stored estimates to float32 rounding, for all 120 trials.
"""
from __future__ import annotations

import cv2
import numpy as np

N_BINS = 40               # horizontal bins of the upper boundary
RANSAC_ITERS = 600        # fixed iteration count
RANSAC_SEED = 0           # fixed seed: the same mask always gives the same line
RANSAC_TOL_MM = 0.75      # inlier half-width
VIAL_LENGTH_MM = 57.0     # length between the two reference pixels on the vial axis, in the paper's set-up


def largest_contour(mask, origin=(0, 0)) -> np.ndarray:
    """Every boundary pixel of the largest external contour of `mask` (any non-zero pixel is material), as an
    (N, 2) array of image x, y.

    `origin` is the position, in the full frame, of the mask's top-left pixel when the mask is a crop of the frame.
    It is added to the contour, so that the rest of the computation runs in full-frame pixels, as on the robot. The
    crop must hold the whole mask with at least one pixel of margin; the result is then identical to the one on
    the full frame."""
    mask_u8 = (np.asarray(mask) > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("the mask is empty")
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(float)
    return contour + np.asarray(origin, dtype=float)


def upper_boundary_points(contour, bins: int = N_BINS) -> np.ndarray:
    """The material/air interface: the topmost contour point in each of `bins` equal-width bins of image x.
    Bins holding fewer than 3 points are dropped (the blob's tapering ends)."""
    pts = np.asarray(contour, dtype=float).reshape(-1, 2)
    x, y = pts[:, 0], pts[:, 1]
    edges = np.linspace(x.min(), x.max(), bins + 1)
    out = []
    for i in range(bins):
        m = (x >= edges[i]) & (x < edges[i + 1])
        if int(m.sum()) < 3:
            continue
        out.append(pts[m][int(np.argmin(y[m]))])
    return np.asarray(out, dtype=float)


def fit_line_tls(points) -> tuple[np.ndarray, np.ndarray]:
    """Total-least-squares line through `points` -> (centroid, unit direction)."""
    pts = np.asarray(points, dtype=float)
    c = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c)
    v = vt[0]
    return c, v / np.linalg.norm(v)


def fit_line_ransac(points, tol_px: float, iters: int = RANSAC_ITERS, seed: int = RANSAC_SEED):
    """Robust line fit -> (centroid, unit direction, inlier mask). Where the material meets the vial wall the
    contour follows the glass; RANSAC keeps the largest set of points consistent with one line."""
    pts = np.asarray(points, dtype=float)
    n_pts = len(pts)
    if n_pts < 4:
        c, v = fit_line_tls(pts)
        return c, v, np.ones(n_pts, dtype=bool)
    rng = np.random.default_rng(seed)
    best_mask, best_n = None, -1
    for _ in range(iters):
        i, j = rng.choice(n_pts, 2, replace=False)
        v = pts[j] - pts[i]
        norm = float(np.linalg.norm(v))
        if norm < 1e-9:
            continue
        v = v / norm
        perp = np.array([-v[1], v[0]])
        mask = np.abs((pts - pts[i]) @ perp) <= tol_px
        if int(mask.sum()) > best_n:
            best_mask, best_n = mask, int(mask.sum())
    c, v = fit_line_tls(pts[best_mask])
    return c, v, best_mask


def surface_line(mask, vial_clicks, vial_length_mm: float = VIAL_LENGTH_MM, tol_mm: float = RANSAC_TOL_MM,
                 origin=(0, 0)):
    """Steps 1-3 on one side-view mask -> (centre, unit direction, upper-boundary points, inlier mask), in pixels
    of the full frame. `vial_clicks` are the two reference pixels on the vial axis, click 0 at the lip end; they set
    the scale of the inlier band (tol_mm)."""
    boundary = upper_boundary_points(largest_contour(mask, origin))
    if len(boundary) < 2:
        raise ValueError("fewer than 2 boundary points: the mask is too small to fit a surface line")
    vc = np.asarray(vial_clicks, dtype=float)
    mm_per_px = vial_length_mm / float(np.linalg.norm(vc[0] - vc[1]))
    centre, direction, inliers = fit_line_ransac(boundary, tol_px=tol_mm / mm_per_px)
    return centre, direction, boundary, inliers


def intersect_line_line_2d(p_bottom, p_top, p0, p1) -> np.ndarray:
    """Intersection of the line p_bottom-p_top with the line p0-p1."""
    a, b, c, d = (np.asarray(p, dtype=float) for p in (p_bottom, p_top, p0, p1))
    r, s = b - a, d - c
    det = r[0] * s[1] - r[1] * s[0]
    if abs(det) < 1e-12:
        raise RuntimeError("Lines are parallel or nearly parallel")
    t = ((c[0] - a[0]) * s[1] - (c[1] - a[1]) * s[0]) / det
    return a + t * r


def compute_plane_pose(p0, p1, p_top, p_bottom, vial_length_m: float):
    """-> (tilt about x_lip in rad, depth of p below the lip along the axis in m, p in image px).
    p0, p1: two image points on the surface line; p_top, p_bottom: the reference pixels (lip end, body)."""
    p0, p1, p_top, p_bottom = (np.asarray(p, dtype=float) for p in (p0, p1, p_top, p_bottom))
    d = (p1 - p0) / np.linalg.norm(p1 - p0)
    d_vial = (p_top - p_bottom) / np.linalg.norm(p_top - p_bottom)
    angle = np.arctan2(d[0] * d_vial[1] - d[1] * d_vial[0], d[0] * d_vial[0] + d[1] * d_vial[1])
    p_img = intersect_line_line_2d(p_bottom, p_top, p0, p1)
    depth_m = vial_length_m / np.linalg.norm(p_top - p_bottom) * np.dot(p_img - p_top, d_vial)
    return -np.pi / 2 - angle, float(depth_m), p_img      # a level surface (edge perpendicular to the axis) -> 0


def estimate_surface(mask, vial_clicks, vial_length_mm: float = VIAL_LENGTH_MM, tol_mm: float = RANSAC_TOL_MM,
                     origin=(0, 0)) -> dict:
    """The whole pipeline on one side-view mask.

    mask          the material in the vial (any non-zero pixel is material); a full frame, or a crop of one
    vial_clicks   the two reference pixels on the vial axis in full-frame pixels, click 0 at the lip end
    vial_length_mm  the length they span
    origin        full-frame position of the mask's top-left pixel, when the mask is a crop

    Returns depth_mm (of the surface below the lip, along the axis; negative = below), tilt_deg (about the vial-lip
    x axis), p_img (where the surface line crosses the axis), and the intermediate steps for drawing: ends (the
    line spanned across its inliers), boundary, inliers, mm_per_px, tol_px. Image coordinates are full-frame."""
    centre, direction, boundary, inliers = surface_line(mask, vial_clicks, vial_length_mm, tol_mm, origin)
    vc = np.asarray(vial_clicks, dtype=float)
    mm_per_px = vial_length_mm / float(np.linalg.norm(vc[0] - vc[1]))
    t = (boundary[inliers] - centre) @ direction
    ends = np.vstack([centre + direction * t.min(), centre + direction * t.max()])
    axis = (vc[0] - vc[1]) / np.linalg.norm(vc[0] - vc[1])
    ends = ends[np.argsort(ends @ np.array([-axis[1], axis[0]]))]     # stable order, perpendicular to the axis
    tilt, depth_m, p_img = compute_plane_pose(ends[0], ends[1], vc[0], vc[1], vial_length_mm / 1000.0)
    return {"depth_mm": depth_m * 1000.0, "tilt_deg": float(np.degrees(tilt)), "p_img": p_img, "ends": ends,
            "boundary": boundary, "inliers": inliers, "mm_per_px": mm_per_px, "tol_px": tol_mm / mm_per_px}
