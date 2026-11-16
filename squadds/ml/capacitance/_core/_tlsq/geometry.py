"""GDS conductor geometry: exact polygon metrics, separation spectra and spectral shape descriptors.

Only metal layers are used: (1,0) ground and (1,10) signal conductors. Port stubs (2,0)/(3,0) are ignored.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import gdstk
import numpy as np
import shapely
from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.spatial import cKDTree
from shapely.geometry import MultiPolygon, Polygon

GROUND_LAYER = (1, 0)
SIGNAL_LAYER = (1, 10)
BOUNDARY_STEP_UM = 0.25
SEP_BINS_UM = np.geomspace(0.5, 2000.0, 41)
N_FOURIER = 16
MIC_TOL = [0.01]  # inscribed-circle tolerance (um); 0.1 for the approximate token variants (PRIOR_SCHEME.md 14.3.1)
RASTER_PX = 512
N_EIG = 16
K_BINS = 48


@dataclass
class Conductors:
    N: Polygon
    S: Polygon
    G: Polygon | MultiPolygon
    pocket: Polygon | MultiPolygon  # etched region inside the chip (contains N and S)
    gap: Polygon | MultiPolygon  # pocket minus signal metal (dielectric surface)
    chip: Polygon
    pads: list = field(default_factory=list)  # extra 0 V bodies (merged into G)

    def metals(self) -> dict:
        return {"N": self.N, "S": self.S, "G": self.G}


def _valid(geom):
    g = shapely.make_valid(geom)
    polys = [p for p in getattr(g, "geoms", [g]) if isinstance(p, (Polygon, MultiPolygon))]
    return shapely.union_all(polys)


def load_conductors(gds_path) -> Conductors:
    lib = gdstk.read_gds(str(gds_path))
    cell = lib.top_level()[0]
    to_um = lib.unit / 1e-6
    sig = [p for p in cell.polygons if (p.layer, p.datatype) == SIGNAL_LAYER]
    gnd = [p for p in cell.polygons if (p.layer, p.datatype) == GROUND_LAYER]
    if len(sig) < 2 or len(gnd) != 1:
        raise ValueError(f"expected >=2 signal + 1 ground polygons, got {len(sig)} + {len(gnd)}")
    polys = [_valid(Polygon(p.points * to_um)) for p in sig]
    if len(polys) == 2:
        N, S = polys
        pads = []
    else:
        # multi-terminal (exp12): GDS order is not the net order. S = star at the origin; N = body fed by the
        # single port stub; the remaining pads are held at 0 V in the solve and are merged into ground.
        port = shapely.union_all([Polygon(p.points * to_um) for p in cell.polygons if (p.layer, p.datatype) != SIGNAL_LAYER and p.layer != GROUND_LAYER[0]])
        s = next((i for i, g in enumerate(polys) if g.contains(shapely.Point(0, 0))), int(np.argmax([g.area for g in polys])))
        n = min((i for i in range(len(polys)) if i != s), key=lambda i: polys[i].distance(port))
        N, S = polys[n], polys[s]
        pads = [g for i, g in enumerate(polys) if i not in (n, s)]
    G0 = _valid(Polygon(gnd[0].points * to_um))
    if abs(G0.area - gnd[0].area() * to_um**2) > 1e-6 * G0.area:
        raise ValueError("ground polygon area changed during repair")
    chip = shapely.box(*G0.bounds)
    pocket = chip.difference(G0)
    G = shapely.union_all([G0, *pads]) if pads else G0
    gap = pocket.difference(N).difference(S)
    for g in pads:
        gap = gap.difference(g)
    return Conductors(N, S, G, pocket, gap, chip, pads)


def sample_boundary(geom, step: float = BOUNDARY_STEP_UM) -> np.ndarray:
    rings = []
    for poly in getattr(geom, "geoms", [geom]):
        rings += [poly.exterior, *poly.interiors]
    pts = []
    for r in rings:
        n = max(int(np.ceil(r.length / step)), 4)
        pts.append(shapely.get_coordinates(shapely.line_interpolate_point(r, np.linspace(0, r.length, n, endpoint=False))))
    return np.concatenate(pts)


def boundaries(c: Conductors) -> dict[str, np.ndarray]:
    # ground edge = boundary of the etched pocket
    g = [sample_boundary(c.pocket)] + [sample_boundary(p) for p in c.pads]
    return {"N": sample_boundary(c.N), "S": sample_boundary(c.S), "G": np.concatenate(g)}


def shape_metrics(poly) -> dict:
    hull = poly.convex_hull
    x0, y0, x1, y1 = poly.bounds
    cx, cy = poly.centroid.x, poly.centroid.y
    ixx, iyy, ixy = _second_moments(poly, cx, cy)
    evals = np.linalg.eigvalsh([[ixx, ixy], [ixy, iyy]])
    return {
        "area": poly.area,
        "perimeter": poly.length,
        "convex_area": hull.area,
        "solidity": poly.area / hull.area,
        "compactness": 4 * np.pi * poly.area / poly.length**2,
        "bbox_w": x1 - x0,
        "bbox_h": y1 - y0,
        "cx": cx,
        "cy": cy,
        "I_major": evals[1],
        "I_minor": evals[0],
        "elongation": np.sqrt(evals[1] / max(evals[0], 1e-30)),
        "inscribed_radius": shapely.maximum_inscribed_circle(poly, MIC_TOL[0]).length if isinstance(poly, Polygon) else np.nan,
        "n_vertices": int(shapely.get_num_coordinates(poly)),
        "n_holes": int(sum(len(p.interiors) for p in getattr(poly, "geoms", [poly]))),
    }


def _second_moments(geom, cx, cy) -> tuple[float, float, float]:
    ixx = iyy = ixy = 0.0
    for poly in getattr(geom, "geoms", [geom]):
        for k, ring in enumerate([poly.exterior, *poly.interiors]):
            xy = np.asarray(ring.coords) - (cx, cy)
            x, y = xy[:-1, 0], xy[:-1, 1]
            xn, yn = xy[1:, 0], xy[1:, 1]
            a = x * yn - xn * y
            sgn = 1.0 if ring.is_ccw == (k == 0) else -1.0
            ixx += sgn * np.sum(a * (y**2 + y * yn + yn**2)) / 12
            iyy += sgn * np.sum(a * (x**2 + x * xn + xn**2)) / 12
            ixy += sgn * np.sum(a * (x * yn + 2 * x * y + 2 * xn * yn + xn * y)) / 24
    return abs(ixx), abs(iyy), ixy


def fourier_descriptors(poly, n_samples: int = 512) -> np.ndarray:
    """|c_k|/|c_1| for k=-N..N of the outer contour (translation/rotation/scale invariant)."""
    ring = max(getattr(poly, "geoms", [poly]), key=lambda p: p.area).exterior
    if not ring.is_ccw:
        ring = shapely.reverse(ring)
    pts = shapely.get_coordinates(shapely.line_interpolate_point(ring, np.linspace(0, 1, n_samples, endpoint=False), normalized=True))
    c = np.fft.fft(pts[:, 0] + 1j * pts[:, 1]) / n_samples
    k = np.r_[np.arange(-N_FOURIER, 0), np.arange(0, N_FOURIER + 1)]
    mags = np.abs(c[k])
    mags[k == 0] = 0.0
    return mags / mags[k == 1][0]


def separation_spectrum(a_pts: np.ndarray, b_pts: np.ndarray) -> tuple[np.ndarray, dict]:
    d, _ = cKDTree(b_pts).query(a_pts)
    hist, _ = np.histogram(d, bins=SEP_BINS_UM, weights=np.full(len(d), BOUNDARY_STEP_UM))
    dmin = float(d.min())
    s = {"dmin": dmin, "P_inv": float(np.sum(BOUNDARY_STEP_UM / np.maximum(d, 0.05)))}
    for f in (1.05, 1.5, 2.0, 3.0):
        s[f"L_{f:g}dmin"] = float(np.sum(d <= f * dmin) * BOUNDARY_STEP_UM)
    for d0 in (5.0, 10.0, 20.0):
        s[f"L_le{d0:g}um"] = float(np.sum(d <= d0) * BOUNDARY_STEP_UM)
    return hist, s


def window(c: Conductors) -> tuple[float, float, float]:
    """Square raster window around the pocket and signal metal: (x0, y0, side) in µm."""
    x0, y0, x1, y1 = shapely.union(c.N, c.S).buffer(0).bounds
    gx0, gy0, gx1, gy1 = c.pocket.bounds
    cx, cy = (gx0 + gx1) / 2, (gy0 + gy1) / 2
    side = 1.05 * max(gx1 - gx0, gy1 - gy0, x1 - x0, y1 - y0)
    return cx - side / 2, cy - side / 2, side


def raster(c: Conductors, npx: int = RASTER_PX) -> tuple[dict[str, np.ndarray], float, tuple[float, float]]:
    wx0, wy0, side = window(c)
    cx, cy = wx0 + side / 2, wy0 + side / 2
    px = side / npx
    xs = cx - side / 2 + (np.arange(npx) + 0.5) * px
    ys = cy - side / 2 + (np.arange(npx) + 0.5) * px
    X, Y = np.meshgrid(xs, ys)
    masks = {}
    for k, g in c.metals().items():
        shapely.prepare(g)
        masks[k] = shapely.contains_xy(g, X, Y)
    return masks, px, (cx - side / 2, cy - side / 2)


def radial_power_spectrum(img: np.ndarray, px: float) -> tuple[np.ndarray, np.ndarray]:
    n = img.shape[0]
    F = np.fft.fftshift(np.fft.fft2(img - img.mean()))
    P = np.abs(F) ** 2 / img.size
    k = np.fft.fftshift(np.fft.fftfreq(n, d=px))
    KX, KY = np.meshgrid(k, k)
    kr = np.hypot(KX, KY).ravel()
    edges = np.geomspace(1.0 / (n * px), 0.5 / px, K_BINS + 1)
    idx = np.digitize(kr, edges) - 1
    ok = (idx >= 0) & (idx < K_BINS)
    s = np.bincount(idx[ok], weights=P.ravel()[ok], minlength=K_BINS)
    cnt = np.bincount(idx[ok], minlength=K_BINS)
    return np.sqrt(edges[1:] * edges[:-1]), np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)


def dirichlet_eigenvalues(mask: np.ndarray, px: float, k: int = N_EIG) -> np.ndarray:
    """Lowest Dirichlet Laplacian eigenvalues of a pixel mask (Shape-DNA), in um^-2."""
    idx = -np.ones(mask.shape, dtype=np.int64)
    ii, jj = np.nonzero(mask)
    n = len(ii)
    if n < 4 * k:
        return np.full(k, np.nan)
    idx[ii, jj] = np.arange(n)
    rows, cols = [np.arange(n)], [np.arange(n)]
    vals = [np.full(n, 4.0)]
    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        ni, nj = ii + di, jj + dj
        inside = (ni >= 0) & (ni < mask.shape[0]) & (nj >= 0) & (nj < mask.shape[1])
        nb = np.full(n, -1)
        nb[inside] = idx[ni[inside], nj[inside]]
        ok = nb >= 0
        rows.append(np.arange(n)[ok])
        cols.append(nb[ok])
        vals.append(-np.ones(ok.sum()))
    L = sparse.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    ev = eigsh(L, k=k, sigma=0, which="LM", return_eigenvectors=False)
    return np.sort(ev) / px**2


def geometry_record(gds_path) -> tuple[dict, dict]:
    return geometry_record_conductors(load_conductors(gds_path))


def geometry_record_conductors(c: Conductors) -> tuple[dict, dict]:
    """Descriptors of a (N, S, G) conductor triple; for multi-conductor queries N/S are the queried pair, G the rest."""
    scalars, arrays = {}, {}
    for name, g in (("N", c.N), ("S", c.S), ("pocket", c.pocket)):
        for k, v in shape_metrics(g).items():
            scalars[f"{name}_{k}"] = float(v)
        arrays[f"fourier_{name}"] = fourier_descriptors(g)
    scalars["gap_area"] = c.gap.area
    scalars["chip_w"], scalars["chip_h"] = c.chip.bounds[2] - c.chip.bounds[0], c.chip.bounds[3] - c.chip.bounds[1]
    scalars["G_area"] = c.G.area

    b = boundaries(c)
    for a, t in (("N", "S"), ("S", "N"), ("N", "G"), ("G", "N"), ("S", "G"), ("G", "S")):
        hist, s = separation_spectrum(b[a], b[t])
        arrays[f"sep_{a}{t}"] = hist
        scalars.update({f"sep_{a}{t}_{k}": v for k, v in s.items()})

    masks, px, origin = raster(c)
    scalars["raster_px_um"] = px
    metal = (masks["N"] | masks["S"] | masks["G"]).astype(np.float64)
    k, P = radial_power_spectrum(metal, px)
    arrays["psd_k"], arrays["psd_metal"] = k, P
    arrays["psd_NS"] = radial_power_spectrum(masks["N"].astype(float) - masks["S"].astype(float), px)[1]
    for name in ("N", "S"):
        arrays[f"eig_{name}"] = dirichlet_eigenvalues(masks[name], px)
        scalars[f"{name}_raster_area_err"] = float(masks[name].sum() * px**2 / getattr(c, name).area - 1)
    arrays["sep_bins_um"] = SEP_BINS_UM
    return scalars, arrays
