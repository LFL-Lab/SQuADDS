"""Generalized multi-terminal star capacitor (Wallraff / Surface-17 style).

Central n-pointed star plus up to n peripheral connector pads. Disabled pad
sectors are filled by the ground plane while keeping an equidistant gap around
the star. Each enabled pad gets a radial CPW stub / pin.
"""

from __future__ import annotations

import numpy as np
from qiskit_metal import Dict, draw
from qiskit_metal.qlibrary.core import QComponent
from shapely import set_precision
from shapely.errors import GEOSException
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union


class GeneralizedCapStar(QComponent):
    """Star capacitor with optional peripheral connector pads.

    Bodies:
      * ``star`` — central n-pointed star (always present)
      * ``pad_i`` — peripheral pads for each enabled connector (i = 0 .. n-1)

    Pads sit in the valleys between fingers: each pad is ``disk(star_outer)``
    minus ``star.buffer(gap_i)`` in that valley, so it matches the gap angle,
    keeps an equidistant clearance to the star, and stops at the finger tips.
    The ground opening is a circle of ``pocket_radius``. Empty valleys are
    either left open to that circle (``valley_ground=circle``) or filled by
    ground with an equidistant ``gap`` to the star (``valley_ground=fill``),
    so tip outer arcs follow the star tip fillets at radius
    ``tip_radius + gap`` via ``star.buffer(gap)``. CPW stubs add a side gap
    only outside the capacitor pocket (``cpw_width`` / ``cpw_gap``), with a
    flat open end. Per-pad star gaps via ``pad_gaps``; omitted => ``gap``.
    """

    default_options = Dict(
        n_points='5',
        star_outer_radius='100um',
        star_inner_radius='45um',
        tip_radius='5um',
        valley_radius='5um',
        star_rotation='90',  # deg; 90 => one finger along +y
        #
        gap='20um',
        # Circular ground opening: 20um past the pad outer ends (star_outer_radius).
        pocket_radius='120um',
        # Per-valley inner radius (how deep each valley dips toward center).
        # Comma-separated lengths; a single value broadcasts to all valleys.
        # Empty / omitted => use star_inner_radius for every valley.
        valley_inner_radii='',
        pad_side_gap='8um',
        #
        # Per-pad gap to the star body (equidistant offset). Comma-separated lengths;
        # a single value broadcasts. Empty / omitted => use ``gap`` for every pad.
        # Disabled (ground-filled) valleys always use ``gap``.
        pad_gaps='',
        # Fillet radii for the three triangle-like corners (0 => sharp).
        pad_apex_radius='5um',       # corner pointing into the valley
        pad_base_cw_radius='5um',    # outer corner on the CW side of the pad axis
        pad_base_ccw_radius='5um',   # outer corner on the CCW side of the pad axis
        #
        # Comma-separated 0/1 flags (avoid a bare '11111' — Metal parses that as a number).
        # Index 0 is the pad in the valley after the first finger (after star_rotation).
        pad_enables='1,1,1,1,1',
        #
        # Per-valley ground in empty (no-pad) valleys: ``circle`` keeps the circular
        # pocket; ``fill`` brings ground in along the finger-edge flats with an
        # equidistant star gap (joins the circle on those flats, not radial tip
        # cuts). Ignored where a pad is enabled. Comma-separated; one token
        # broadcasts. Aliases: c/0=circle, f/1=fill.
        valley_ground='circle',
        #
        corner_arc_points='12',
        #
        cpw_length='25um',
        cpw_width='10um',
        cpw_gap='6um',
    )

    TOOLTIP = """Multi-terminal star capacitor with optional connector pads."""

    @staticmethod
    def _clean_geometry(geom, precision=1e-6):
        if geom is None or geom.is_empty:
            return geom
        geom = geom.buffer(0)
        try:
            geom = set_precision(geom, precision)
        except GEOSException:
            geom = geom.buffer(0)
            geom = set_precision(geom, precision)
        return geom.buffer(0)

    @staticmethod
    def _as_single_polygon(geom, context):
        if geom is None or geom.is_empty:
            raise ValueError(f'{context} is empty')
        geom = geom.buffer(0)
        if geom.geom_type == 'Polygon':
            return geom
        polygons = [
            part for part in getattr(geom, 'geoms', []) if part.geom_type == 'Polygon'
        ]
        if not polygons:
            raise ValueError(f'{context} has no polygon geometry ({geom.geom_type})')
        if len(polygons) == 1:
            return polygons[0]
        merged = unary_union(polygons).buffer(0)
        if merged.geom_type == 'Polygon':
            return merged
        # Soft bridge for near-touching pad + CPW fragments.
        merged = unary_union(polygons).buffer(1e-7).buffer(-1e-7)
        if merged.geom_type == 'MultiPolygon':
            merged = max(merged.geoms, key=lambda g: g.area)
        if merged.geom_type != 'Polygon' or merged.is_empty:
            raise ValueError(f'{context} could not be merged into one polygon')
        return merged

    @staticmethod
    def _parse_pad_enables(raw, n_points):
        if raw is None:
            return [True] * n_points
        if isinstance(raw, (list, tuple)):
            vals = [bool(int(v)) if not isinstance(v, bool) else bool(v) for v in raw]
        else:
            # Metal may parse '11111' as an int; keep digit string form.
            if isinstance(raw, bool):
                vals = [bool(raw)] * n_points
                return vals
            if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                text = f'{int(raw)}'
            else:
                text = str(raw).strip().lower()
            if ',' in text:
                parts = [p.strip() for p in text.split(',') if p.strip() != '']
                vals = []
                for part in parts:
                    vals.append(part in ('1', 'true', 'yes', 'on'))
            else:
                vals = []
                for ch in text:
                    if ch in ('1', 't', 'y'):
                        vals.append(True)
                    elif ch in ('0', 'f', 'n'):
                        vals.append(False)
                    elif ch in (' ', '_', '-', '.'):
                        continue
                    else:
                        raise ValueError(
                            f"pad_enables has invalid character {ch!r}; "
                            "use '1,0,1,0,1' (recommended) or '10101'"
                        )
        if len(vals) == 1 and n_points > 1:
            vals = vals * n_points
        if len(vals) != n_points:
            raise ValueError(
                f'pad_enables length {len(vals)} must equal n_points={n_points}'
            )
        return vals

    def _parse_pad_gaps(self, raw, n_points, default):
        """Parse per-pad star gaps; broadcast a single value; default fills blanks."""
        default = float(default)
        if raw is None:
            return [default] * n_points
        if isinstance(raw, (list, tuple)):
            parts = list(raw)
        else:
            if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                # Metal may already parse a single length into meters.
                return [float(raw)] * n_points
            text = str(raw).strip()
            if text == '':
                return [default] * n_points
            parts = [p.strip() for p in text.split(',')] if ',' in text else [text]
        vals = []
        for part in parts:
            if part is None or part == '':
                vals.append(default)
            elif isinstance(part, (int, float)) and not isinstance(part, bool):
                vals.append(float(part))
            else:
                vals.append(float(self.parse_value(str(part).strip())))
        if len(vals) == 1 and n_points > 1:
            vals = vals * n_points
        if len(vals) != n_points:
            raise ValueError(
                f'pad_gaps length {len(vals)} must equal n_points={n_points}'
            )
        for i, g in enumerate(vals):
            if g <= 0.0:
                raise ValueError(f'pad_gaps[{i}] must be > 0 (got {g})')
        return vals

    def _parse_valley_inner_radii(self, raw, n_points, default):
        """Parse per-valley inner radii; broadcast a single value; default fills blanks."""
        default = float(default)
        if raw is None:
            return [default] * n_points
        if isinstance(raw, (list, tuple)):
            parts = list(raw)
        else:
            if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                return [float(raw)] * n_points
            text = str(raw).strip()
            if text == '':
                return [default] * n_points
            parts = [p.strip() for p in text.split(',')] if ',' in text else [text]
        vals = []
        for part in parts:
            if part is None or part == '':
                vals.append(default)
            elif isinstance(part, (int, float)) and not isinstance(part, bool):
                vals.append(float(part))
            else:
                vals.append(float(self.parse_value(str(part).strip())))
        if len(vals) == 1 and n_points > 1:
            vals = vals * n_points
        if len(vals) != n_points:
            raise ValueError(
                f'valley_inner_radii length {len(vals)} must equal n_points={n_points}'
            )
        for i, r in enumerate(vals):
            if r <= 0.0:
                raise ValueError(f'valley_inner_radii[{i}] must be > 0 (got {r})')
        return vals

    @staticmethod
    def _parse_valley_ground(raw, n_points):
        """Parse per-valley empty-mode: True = fill with ground, False = circular."""
        if raw is None:
            return [False] * n_points

        def one(token):
            if isinstance(token, bool):
                return bool(token)
            if isinstance(token, (int, float)) and not isinstance(token, bool):
                return int(token) != 0
            text = str(token).strip().lower()
            if text in ('fill', 'f', '1', 'ground', 'v', 'true', 'yes', 'on'):
                return True
            if text in ('circle', 'c', '0', 'circular', 'open', 'false', 'no', 'off', ''):
                return False
            raise ValueError(
                f"valley_ground token {token!r} is invalid; use 'circle' or 'fill'"
            )

        if isinstance(raw, (list, tuple)):
            vals = [one(v) for v in raw]
        elif isinstance(raw, (int, float)) and not isinstance(raw, bool):
            text = f'{int(raw)}'
            if len(text) == n_points and all(ch in '01' for ch in text):
                vals = [ch == '1' for ch in text]
            else:
                vals = [one(raw)]
        else:
            text = str(raw).strip()
            if text == '':
                vals = [False]
            elif ',' in text:
                vals = [one(p.strip()) for p in text.split(',')]
            else:
                lower = text.lower()
                if len(text) == n_points and all(ch in '01cf' for ch in lower):
                    vals = [one(ch) for ch in lower]
                else:
                    vals = [one(text)]
        if len(vals) == 1 and n_points > 1:
            vals = vals * n_points
        if len(vals) != n_points:
            raise ValueError(
                f'valley_ground length {len(vals)} must equal n_points={n_points}'
            )
        return vals

    @staticmethod
    def _um(x):
        return float(x) * 1e6

    @staticmethod
    def _fillet_vertex(p_prev, p_vert, p_next, radius, n_arc=12):
        """Return polyline points replacing p_vert with a circular fillet.

        Same idea as ``generalized_ncap_interdigital._get_rounded_corner_coords``:
        move the arc center away from the sharp corner, then trace a circular
        arc between the two tangent points. Radius 0 keeps the sharp vertex.

        Assumes the polygon is traversed CCW (interior to the left).
        """
        if radius < 1e-9:
            return [tuple(map(float, p_vert))]

        a = np.asarray(p_prev, dtype=float)
        b = np.asarray(p_vert, dtype=float)
        c = np.asarray(p_next, dtype=float)

        # Unit directions from the corner along each adjacent edge.
        u = a - b
        v = c - b
        len_u = np.linalg.norm(u)
        len_v = np.linalg.norm(v)
        if len_u < 1e-12 or len_v < 1e-12:
            return [tuple(map(float, b))]
        u = u / len_u
        v = v / len_v

        # Smaller angle between the edge directions (<= pi).
        cosang = float(np.clip(np.dot(u, v), -1.0, 1.0))
        ang = float(np.arccos(cosang))
        if ang < 1e-8 or (np.pi - ang) < 1e-8:
            return [tuple(map(float, b))]

        # Tangent-point distance from the corner along each edge.
        half = ang / 2.0
        dist = radius / np.tan(half)
        max_dist = 0.45 * min(len_u, len_v)
        if dist > max_dist:
            dist = max_dist
            radius = dist * np.tan(half)
        if radius < 1e-9:
            return [tuple(map(float, b))]

        t1 = b + u * dist
        t2 = b + v * dist

        # Move the circle center from the sharp corner along the angle bisector
        # into the small-angle sector (interior for tips, exterior notch for valleys).
        bis = u + v
        bis_norm = np.linalg.norm(bis)
        if bis_norm < 1e-12:
            return [tuple(map(float, b))]
        bis = bis / bis_norm
        center = b + bis * (radius / np.sin(half))

        a1 = np.arctan2(t1[1] - center[1], t1[0] - center[0])
        a2 = np.arctan2(t2[1] - center[1], t2[0] - center[0])
        d_ccw = (a2 - a1) % (2.0 * np.pi)
        d_cw = (a1 - a2) % (2.0 * np.pi)

        # Pick the arc that faces the corner being replaced (midpoint closer to b).
        # Using the traversal cross alone is easy to get backwards for star tips.
        mid_ccw = a1 + 0.5 * d_ccw
        mid_cw = a1 - 0.5 * d_cw
        p_ccw = center + radius * np.array([np.cos(mid_ccw), np.sin(mid_ccw)])
        p_cw = center + radius * np.array([np.cos(mid_cw), np.sin(mid_cw)])
        if np.linalg.norm(p_ccw - b) <= np.linalg.norm(p_cw - b):
            angles = a1 + np.linspace(0.0, d_ccw, max(int(n_arc), 2))
        else:
            angles = a1 - np.linspace(0.0, d_cw, max(int(n_arc), 2))

        return [
            (
                float(center[0] + radius * np.cos(th)),
                float(center[1] + radius * np.sin(th)),
            )
            for th in angles
        ]

    @classmethod
    def _make_rounded_star(
        cls,
        n_points,
        r_outer,
        r_inner,
        tip_radius,
        valley_radius,
        rotation_deg,
        n_arc=12,
    ):
        assert n_points >= 3, 'n_points must be >= 3'
        assert r_outer > 0.0, 'r_outer must be > 0'
        assert tip_radius >= 0.0 and valley_radius >= 0.0, 'corner radii must be >= 0'

        # r_inner may be a scalar (broadcast) or a list of n_points values.
        if hasattr(r_inner, '__len__'):
            r_inner_list = [float(v) for v in r_inner]
        else:
            r_inner_list = [float(r_inner)] * n_points
        assert len(r_inner_list) == n_points, 'r_inner list length must equal n_points'
        assert all(r_outer > r > 0.0 for r in r_inner_list), (
            'need star_outer_radius > every valley_inner_radius > 0'
        )

        rot = np.deg2rad(float(rotation_deg))
        verts = []
        radii = []
        for i in range(n_points):
            a_tip = rot + 2.0 * np.pi * i / n_points
            a_val = a_tip + np.pi / n_points
            verts.append((r_outer * np.cos(a_tip), r_outer * np.sin(a_tip)))
            radii.append(float(tip_radius))
            verts.append((r_inner_list[i] * np.cos(a_val), r_inner_list[i] * np.sin(a_val)))
            radii.append(float(valley_radius))

        n = len(verts)
        outline = []
        for i in range(n):
            p_prev = verts[(i - 1) % n]
            p = verts[i]
            p_next = verts[(i + 1) % n]
            outline.extend(cls._fillet_vertex(p_prev, p, p_next, radii[i], n_arc=n_arc))

        poly = Polygon(outline)
        if not poly.is_valid or poly.is_empty:
            poly = poly.buffer(0)
        # Ensure CCW exterior for consistent buffering / filleting assumptions.
        from shapely.geometry.polygon import orient

        poly = cls._clean_geometry(poly)
        return orient(poly, sign=1.0)

    @staticmethod
    def _sector_polygon(radius, angle0, angle1, n_arc=64):
        """Pie slice from angle0 to angle1 (radians), CCW, out to radius."""
        a0 = float(angle0)
        a1 = float(angle1)
        while a1 <= a0:
            a1 += 2.0 * np.pi
        sweep = a1 - a0
        n = max(int(np.ceil(n_arc * sweep / (2.0 * np.pi))), 8)
        angles = np.linspace(a0, a1, n)
        pts = [(0.0, 0.0)]
        pts.extend((float(radius * np.cos(a)), float(radius * np.sin(a))) for a in angles)
        pts.append((0.0, 0.0))
        return Polygon(pts)

    def _pad_angles(self, index, n_points, rotation_deg):
        """Return (center, start, end) angles for pad sector i (radians).

        Pads sit in the valleys between fingers, so the pad axis is offset by
        half a sector from the finger tips.
        """
        rot = np.deg2rad(float(rotation_deg))
        half = np.pi / n_points
        # Valley between finger ``index`` and finger ``index+1``.
        center = rot + 2.0 * np.pi * index / n_points + half
        return center, center - half, center + half

    @staticmethod
    def _arc_points(center, radius, a_start, a_end, n_arc=12):
        """Sample a circular arc from ``a_start`` to ``a_end`` (shortest directed)."""
        a0 = float(a_start)
        a1 = float(a_end)
        # Choose the shorter signed sweep unless a long sweep was intentionally set.
        d = (a1 - a0 + np.pi) % (2.0 * np.pi) - np.pi
        n = max(int(n_arc), 2)
        angles = a0 + np.linspace(0.0, d, n)
        c = np.asarray(center, dtype=float)
        return [
            (float(c[0] + radius * np.cos(th)), float(c[1] + radius * np.sin(th)))
            for th in angles
        ]

    @classmethod
    def _fillet_circle_line_corner(
        cls,
        r_outer,
        line_point,
        line_dir,
        pad_side_normal,
        fillet_radius,
        n_arc=12,
    ):
        """Fillet between outer circle (|p|=r_outer) and a straight pad side.

        Returns a list of ``(center, t_line, t_circ, alpha)`` candidates, or
        ``None``. The fillet is tangent to the circle and to the line.
        """
        rho = float(fillet_radius)
        R = float(r_outer)
        if rho < 1e-12:
            return None
        if rho >= R - 1e-12:
            return None

        p0 = np.asarray(line_point, dtype=float)
        d = np.asarray(line_dir, dtype=float)
        dn = np.linalg.norm(d)
        if dn < 1e-15:
            return None
        d = d / dn
        n = np.asarray(pad_side_normal, dtype=float)
        nn = np.linalg.norm(n)
        if nn < 1e-15:
            return None
        n = n / nn
        # Ensure n points toward the pad interior (and is perpendicular to d).
        n = n - d * np.dot(n, d)
        nn = np.linalg.norm(n)
        if nn < 1e-15:
            # Fall back to left normal of d.
            n = np.array([-d[1], d[0]])
        else:
            n = n / nn

        # Fillet center: distance R-rho from origin, distance rho from the line.
        # Line equation: (x - p0) · n_line = 0 with n_line = n (pad interior).
        # Centers on the pad side of the line satisfy (c - p0)·n = rho.
        d_fillet = R - rho
        # Solve |c|=d_fillet and c·n = p0·n + rho.
        # Represent c = d_fillet * (cos α, sin α).
        target = float(np.dot(p0, n) + rho)
        if abs(target) > d_fillet + 1e-12:
            return None
        # n = (nx, ny); c·n = d_fillet (nx cosα + ny sinα) = target
        # = d_fillet |n| cos(α - ang_n) with |n|=1 → cos(α - ang_n) = target/d_fillet
        ang_n = float(np.arctan2(n[1], n[0]))
        cos_delta = float(np.clip(target / d_fillet, -1.0, 1.0))
        delta = float(np.arccos(cos_delta))
        # Two candidates; pick the one whose circle-tangent point lies on this corner
        # (closer to the unbounded line's intersection with the outer circle on this side).
        candidates = []
        for sign in (+1.0, -1.0):
            alpha = ang_n + sign * delta
            c = d_fillet * np.array([np.cos(alpha), np.sin(alpha)])
            # Tangent on outer circle: same ray as center.
            t_circ = R * np.array([np.cos(alpha), np.sin(alpha)])
            # Tangent on line: drop perpendicular from c to the line.
            t_line = c - n * rho
            # Keep centers that sit on the pad side and inside the outer disk band.
            if np.dot(c - p0, n) < rho - 1e-9:
                continue
            candidates.append((c, t_line, t_circ, alpha))

        if not candidates:
            return None
        return candidates

    @staticmethod
    def _halfplane_polygon(point, inward_normal, size):
        """Large rectangle covering the half-plane on the ``inward_normal`` side."""
        p0 = np.asarray(point, dtype=float)
        n = np.asarray(inward_normal, dtype=float)
        nn = np.linalg.norm(n)
        if nn < 1e-15:
            raise ValueError('degenerate half-plane normal')
        n = n / nn
        d = np.array([-n[1], n[0]])
        s = float(size)
        pts = [
            p0 + d * s,
            p0 - d * s,
            p0 - d * s + n * s,
            p0 + d * s + n * s,
        ]
        return Polygon([(float(x), float(y)) for x, y in pts])

    @classmethod
    def _finger_edge_offset_line(cls, tip_xy, valley_xy, gap, valley_angle):
        """Offset of the tip→valley finger edge toward the valley by ``gap``.

        Returns ``(point_on_line, direction, inward_normal)``.
        """
        tip = np.asarray(tip_xy, dtype=float)
        valley = np.asarray(valley_xy, dtype=float)
        direction = valley - tip
        length = np.linalg.norm(direction)
        if length < 1e-15:
            raise ValueError('degenerate finger edge')
        direction = direction / length

        # Two unit normals; pick the one pointing toward the valley axis.
        n1 = np.array([-direction[1], direction[0]])
        n2 = -n1
        valley_dir = np.array([np.cos(valley_angle), np.sin(valley_angle)])
        mid = 0.5 * (tip + valley)
        # Prefer the normal that moves the midpoint toward the valley ray.
        if np.dot(mid + n1 * float(gap), valley_dir) >= np.dot(mid + n2 * float(gap), valley_dir):
            inward = n1
        else:
            inward = n2
        # If both are ambiguous, choose by cross with tip→valley vs valley axis.
        if abs(np.dot(inward, valley_dir)) < 1e-9:
            inward = n1 if (direction[0] * valley_dir[1] - direction[1] * valley_dir[0]) > 0 else n2

        point = tip + inward * float(gap)
        return point, direction, inward

    @classmethod
    def _star_corner_offset_disk(
        cls,
        angle,
        r_vertex,
        r_neighbor_a,
        r_neighbor_b,
        neighbor_half,
        corner_radius,
        gap,
    ):
        """Equidistant gap disk around a tip or valley corner of the star.

        Uses the same fillet center as the star corner; outer radius is
        ``corner_radius + gap`` (or ``gap`` if the corner is sharp).
        """
        vert = np.array(
            [float(r_vertex * np.cos(angle)), float(r_vertex * np.sin(angle))],
            dtype=float,
        )
        rho = float(corner_radius)
        g = float(gap)
        if rho < 1e-12:
            return Point(float(vert[0]), float(vert[1])).buffer(g, quad_segs=24)

        a = np.array(
            [
                float(r_neighbor_a * np.cos(angle - neighbor_half)),
                float(r_neighbor_a * np.sin(angle - neighbor_half)),
            ],
            dtype=float,
        )
        b = np.array(
            [
                float(r_neighbor_b * np.cos(angle + neighbor_half)),
                float(r_neighbor_b * np.sin(angle + neighbor_half)),
            ],
            dtype=float,
        )
        u = a - vert
        v = b - vert
        len_u = np.linalg.norm(u)
        len_v = np.linalg.norm(v)
        if len_u < 1e-15 or len_v < 1e-15:
            return Point(float(vert[0]), float(vert[1])).buffer(g, quad_segs=24)
        u = u / len_u
        v = v / len_v
        cosang = float(np.clip(np.dot(u, v), -1.0, 1.0))
        ang = float(np.arccos(cosang))
        if ang < 1e-8 or (np.pi - ang) < 1e-8:
            return Point(float(vert[0]), float(vert[1])).buffer(g, quad_segs=24)
        half_ang = ang / 2.0
        bis = u + v
        bis_n = np.linalg.norm(bis)
        if bis_n < 1e-15:
            return Point(float(vert[0]), float(vert[1])).buffer(g, quad_segs=24)
        bis = bis / bis_n
        center = vert + bis * (rho / np.sin(half_ang))
        return Point(float(center[0]), float(center[1])).buffer(
            rho + g, quad_segs=24
        )

    @classmethod
    def _star_tip_offset_disk(cls, tip_angle, r_outer, r_inner, tip_radius, gap, n_points):
        half = np.pi / float(n_points)
        return cls._star_corner_offset_disk(
            tip_angle, r_outer, r_inner, r_inner, half, tip_radius, gap
        )

    @classmethod
    def _valley_fill_wedge(
        cls,
        valley_angle,
        tip_angle_cw,
        tip_angle_ccw,
        r_outer,
        r_inner,
        gap,
    ):
        """Unbounded wedge for a ground-filled valley.

        Bounded by the gap-offset finger edges (flat sides), not by radial tip
        rays — so the circular pocket joins those flats instead of dropping
        inward along the finger midline.
        """
        tip_cw = (float(r_outer * np.cos(tip_angle_cw)), float(r_outer * np.sin(tip_angle_cw)))
        tip_ccw = (
            float(r_outer * np.cos(tip_angle_ccw)),
            float(r_outer * np.sin(tip_angle_ccw)),
        )
        valley_pt = (
            float(r_inner * np.cos(valley_angle)),
            float(r_inner * np.sin(valley_angle)),
        )
        size = max(float(r_outer), float(gap)) * 8.0
        wedge = None
        for tip_xy in (tip_cw, tip_ccw):
            p_line, _direction, inward = cls._finger_edge_offset_line(
                tip_xy, valley_pt, gap, valley_angle
            )
            hp = cls._halfplane_polygon(p_line, inward, size)
            wedge = hp if wedge is None else wedge.intersection(hp)
        return cls._clean_geometry(wedge)

    @classmethod
    def _make_valley_pad(
        cls,
        star,
        valley_angle,
        tip_angle_cw,
        tip_angle_ccw,
        r_outer,
        r_inner,
        gap,
        side_gap,
        apex_radius,
        base_cw_radius,
        base_ccw_radius,
        n_arc=12,
    ):
        """Pad snug in a star valley: equidistant ``gap``, clipped at finger tips.

        Sides are the gap-offset finger edges (matching the valley angle). The
        outer rim is the ``star_outer_radius`` circle. Outer corners are rounded
        so the fillet is tangent to that circle and to each flat side.
        """
        assert gap > 0.0, 'pad gap must be > 0'
        assert r_outer > 0.0, 'r_outer must be > 0'
        _ = side_gap

        a0 = float(tip_angle_cw)
        a1 = float(tip_angle_ccw)
        while a1 <= a0:
            a1 += 2.0 * np.pi

        disk = Point(0.0, 0.0).buffer(float(r_outer), quad_segs=96)
        keepout = cls._clean_geometry(star.buffer(float(gap)))

        tip_cw = (float(r_outer * np.cos(a0)), float(r_outer * np.sin(a0)))
        tip_ccw = (float(r_outer * np.cos(a1)), float(r_outer * np.sin(a1)))
        valley_pt = (
            float(r_inner * np.cos(valley_angle)),
            float(r_inner * np.sin(valley_angle)),
        )

        hp_size = float(r_outer) * 4.0
        lines = []
        for tip_xy, base_r in (
            (tip_cw, float(base_cw_radius)),
            (tip_ccw, float(base_ccw_radius)),
        ):
            p_line, direction, inward = cls._finger_edge_offset_line(
                tip_xy, valley_pt, gap, valley_angle
            )
            lines.append((p_line, direction, inward, base_r, np.asarray(tip_xy, dtype=float)))

        wedge = disk
        for p_line, _direction, inward, _base_r, _tip in lines:
            wedge = wedge.intersection(cls._halfplane_polygon(p_line, inward, hp_size))
        wedge = cls._clean_geometry(wedge)
        wedge = cls._as_single_polygon(wedge, 'valley pad wedge')

        # Round line/circle corners with an equal-radius offset first (exact
        # tangency to the outer circle and both flats), then grow the larger
        # corner locally if the two base radii differ.
        rho_cw = float(base_cw_radius)
        rho_ccw = float(base_ccw_radius)
        rho_eq = min(rho_cw, rho_ccw)
        if rho_eq > 1e-12:
            eroded = wedge.buffer(-rho_eq, quad_segs=max(int(n_arc), 8))
            if eroded is None or eroded.is_empty:
                raise ValueError('pad base radii too large for this valley')
            wedge = cls._clean_geometry(eroded.buffer(rho_eq, quad_segs=max(int(n_arc), 8)))
            wedge = cls._as_single_polygon(wedge, 'valley pad rounded')

        for p_line, direction, inward, base_r, tip_xy in lines:
            extra = float(base_r) - rho_eq
            if extra < 1e-12:
                continue
            cands = cls._fillet_circle_line_corner(
                r_outer, p_line, direction, inward, base_r, n_arc=n_arc
            )
            if not cands:
                continue
            center, _t_line, _t_circ, _alpha = min(
                cands, key=lambda item: float(np.linalg.norm(item[0] - tip_xy))
            )
            fillet_disk = Point(float(center[0]), float(center[1])).buffer(
                base_r, quad_segs=max(int(n_arc) * 2, 16)
            )
            n = inward / np.linalg.norm(inward)
            ang_n = float(np.arctan2(n[1], n[0]))
            proj = float(np.dot(p_line, n))
            if abs(proj) > r_outer:
                sharp = tip_xy
            else:
                delta = float(np.arccos(np.clip(proj / r_outer, -1.0, 1.0)))
                pts_s = [
                    r_outer * np.array([np.cos(ang_n + s * delta), np.sin(ang_n + s * delta)])
                    for s in (+1.0, -1.0)
                ]
                sharp = min(pts_s, key=lambda p: float(np.linalg.norm(p - tip_xy)))
            cut = Point(float(sharp[0]), float(sharp[1])).buffer(
                max(base_r * 2.0, extra * 3.0), quad_segs=24
            )
            wedge = cls._clean_geometry(wedge.difference(cut.difference(fillet_disk)))
            wedge = cls._clean_geometry(
                unary_union(
                    [
                        wedge,
                        fillet_disk.intersection(disk).intersection(
                            cls._halfplane_polygon(p_line, inward, hp_size)
                        ),
                    ]
                )
            )

        pad = cls._clean_geometry(wedge.difference(keepout))
        pad = cls._as_single_polygon(pad, 'valley pad after keepout')

        if apex_radius > 1e-12:
            coords = list(pad.exterior.coords)[:-1]
            radii = [float(np.hypot(x, y)) for x, y in coords]
            apex_i = int(np.argmin(radii))
            outline = []
            n = len(coords)
            for i in range(n):
                r = float(apex_radius) if i == apex_i else 0.0
                outline.extend(
                    cls._fillet_vertex(
                        coords[(i - 1) % n],
                        coords[i],
                        coords[(i + 1) % n],
                        r,
                        n_arc=n_arc,
                    )
                )
            pad = Polygon(outline)
            if not pad.is_valid or pad.is_empty:
                pad = pad.buffer(0)
            pad = cls._clean_geometry(pad)
            pad = cls._clean_geometry(pad.difference(keepout).intersection(disk))
            pad = cls._as_single_polygon(pad, 'valley pad after apex fillet')

        return pad

    @staticmethod
    def _radius_along_ray(geom, angle, r_max):
        """Farthest intersection radius of a ray with ``geom`` boundary."""
        ray = LineString(
            [
                (0.0, 0.0),
                (float(r_max * np.cos(angle)), float(r_max * np.sin(angle))),
            ]
        )
        hit = ray.intersection(geom.boundary)
        if hit.is_empty:
            return None
        if hit.geom_type == 'Point':
            return float(np.hypot(hit.x, hit.y))
        radii = []
        geoms = getattr(hit, 'geoms', [hit])
        for g in geoms:
            if g.geom_type == 'Point':
                radii.append(float(np.hypot(g.x, g.y)))
            elif g.geom_type == 'MultiPoint':
                for p in g.geoms:
                    radii.append(float(np.hypot(p.x, p.y)))
            elif hasattr(g, 'coords'):
                for x, y in g.coords:
                    radii.append(float(np.hypot(x, y)))
        return max(radii) if radii else None

    @staticmethod
    def _radial_cpw_box(angle, width, r0, r1):
        """Rectangle along +y from r0..r1, then rotated so +y maps to ``angle``."""
        half = float(width) / 2.0
        ymin, ymax = (r0, r1) if r0 <= r1 else (r1, r0)
        local = box(-half, ymin, half, ymax)
        # local is built along +y; angle=0 should point to +x.
        return draw.rotate(local, np.rad2deg(float(angle)) - 90.0, origin=(0, 0))

    def make(self):
        p = self.p
        n_points = int(p.n_points)
        assert n_points >= 3, 'n_points must be >= 3'
        pad_enables = self._parse_pad_enables(
            self.options.get('pad_enables', '1,1,1,1,1'),
            n_points,
        )
        pad_gaps = self._parse_pad_gaps(
            self.options.get('pad_gaps', ''),
            n_points,
            p.gap,
        )
        valley_fill = self._parse_valley_ground(
            self.options.get('valley_ground', 'circle'),
            n_points,
        )
        valley_inner_radii = self._parse_valley_inner_radii(
            self.options.get('valley_inner_radii', ''),
            n_points,
            p.star_inner_radius,
        )
        n_arc = int(p.corner_arc_points)
        assert n_arc >= 2, 'corner_arc_points must be >= 2'

        assert p.gap > 0.0, 'gap must be > 0'
        assert p.pocket_radius > p.star_outer_radius, (
            'pocket_radius must be larger than star_outer_radius '
            '(pads end at star_outer_radius)'
        )
        assert p.pad_side_gap >= 0.0, 'pad_side_gap must be >= 0'
        assert p.cpw_length >= 0.0, 'cpw_length must be >= 0'
        assert p.cpw_width > 0.0, 'cpw_width must be > 0'
        assert p.cpw_gap > 0.0, 'cpw_gap must be > 0'

        star = self._make_rounded_star(
            n_points,
            float(p.star_outer_radius),
            valley_inner_radii,
            float(p.tip_radius),
            float(p.valley_radius),
            float(p.star_rotation),
            n_arc=n_arc,
        )
        assert star is not None and not star.is_empty, 'star geometry is empty'

        r_outer = float(p.star_outer_radius)
        r_circ = float(p.pocket_radius)

        etch_parts = []
        pad_geoms = {}
        pin_specs = []
        fill_valleys = []  # empty valleys where ground hugs the star (equidistant)
        # Normalized tip angle -> fill-valley indices that touch this tip.
        tip_fill_neighbors = {}

        def _norm_ang(a):
            return float(np.mod(float(a), 2.0 * np.pi))

        for i, enabled in enumerate(pad_enables):
            center, a0, a1 = self._pad_angles(i, n_points, p.star_rotation)
            gap_i = float(pad_gaps[i]) if enabled else float(p.gap)
            star_keepout_i = self._clean_geometry(star.buffer(gap_i))

            if enabled:
                pad = self._make_valley_pad(
                    star,
                    center,
                    a0,
                    a1,
                    r_outer,
                    valley_inner_radii[i],
                    gap_i,
                    float(p.pad_side_gap),
                    float(p.pad_apex_radius),
                    float(p.pad_base_cw_radius),
                    float(p.pad_base_ccw_radius),
                    n_arc=n_arc,
                )

                if p.cpw_length > 1e-12:
                    r_pin = r_circ + float(p.cpw_length)
                    # Lead metal starts on the pad and runs to the pin.
                    r_lead0 = max(r_outer * 0.55, r_outer - max(gap_i, float(p.gap)))
                    lead = self._radial_cpw_box(center, p.cpw_width, r_lead0, r_pin)
                    lead = self._clean_geometry(lead.difference(star_keepout_i))
                    pad = self._as_single_polygon(
                        self._clean_geometry(unary_union([pad, lead])),
                        f'pad_{i} with CPW',
                    )
                    # CPW side-gap etch only outside the capacitor pocket so we
                    # do not spawn ground islands inside the main gap. Overlap
                    # slightly into the pocket so the stub union stays attached.
                    etch_w = float(p.cpw_width) + 2.0 * float(p.cpw_gap)
                    r_cpw0 = max(r_outer, r_circ - max(float(p.cpw_gap), 1e-3))
                    etch_parts.append(
                        self._radial_cpw_box(center, etch_w, r_cpw0, r_pin)
                    )
                    pin_specs.append((i, center, r_pin))

                pad_geoms[f'pad_{i}'] = pad
            elif valley_fill[i]:
                fill_valleys.append((center, a0, a1, float(p.gap), valley_inner_radii[i]))
                tip_fill_neighbors.setdefault(_norm_ang(a0), []).append(i)
                tip_fill_neighbors.setdefault(_norm_ang(a1), []).append(i)

        # Ground opening. Default: circle of pocket_radius. Fill valleys bring
        # ground in along finger-edge flats with equidistant star keepout.
        # Keepout = star.buffer(gap), then open with valley_radius so valley
        # spikes (gap > valley_r) become arcs of radius valley_r. Tip outer
        # arcs are tip_r+gap from the star tip centers; wrap ground at tips
        # that border a fill valley so those arcs are not cut into sharp Vs.
        tip_radius = float(p.tip_radius)
        valley_radius = float(p.valley_radius)
        gap0 = float(p.gap)

        def _star_keepout(gap_i):
            return self._clean_geometry(star.buffer(float(gap_i), quad_segs=48))

        all_fill = (not any(pad_enables)) and all(valley_fill)
        if all_fill:
            circle_pocket = _star_keepout(gap0)
        else:
            circle_pocket = Point(0.0, 0.0).buffer(r_circ, quad_segs=96)
            if fill_valleys:
                keepout_default = _star_keepout(gap0)
                for center, a0, a1, gap_i, r_inner_i in fill_valleys:
                    keepout_i = keepout_default
                    if abs(gap_i - gap0) >= 1e-15:
                        keepout_i = _star_keepout(gap_i)
                    wedge = self._valley_fill_wedge(
                        center, a0, a1, r_outer, r_inner_i, gap_i
                    )
                    filled = self._clean_geometry(
                        circle_pocket.intersection(wedge).difference(keepout_i)
                    )
                    if filled is not None and not filled.is_empty:
                        circle_pocket = self._clean_geometry(
                            circle_pocket.difference(filled)
                        )

                # Round sharp inward valley spikes in the etch pocket.
                if valley_radius > 1e-12:
                    opened = circle_pocket.buffer(-valley_radius, quad_segs=16)
                    if opened is not None and not opened.is_empty:
                        circle_pocket = self._clean_geometry(
                            opened.buffer(valley_radius, quad_segs=16)
                        )

                # Tips with fill on BOTH sides: wrap ground around the tip
                # so the outer gap follows the tip_r+gap arc, not a sharp V.
                for tip_ang, neighbors in tip_fill_neighbors.items():
                    if len(set(neighbors)) < 2:
                        continue
                    tip_xy = Point(
                        float(r_outer * np.cos(tip_ang)),
                        float(r_outer * np.sin(tip_ang)),
                    )
                    near = tip_xy.buffer(
                        float(tip_radius + gap0) * 2.5, quad_segs=24
                    )
                    wrap = self._clean_geometry(
                        circle_pocket.intersection(near).difference(keepout_default)
                    )
                    if wrap is not None and not wrap.is_empty:
                        circle_pocket = self._clean_geometry(
                            circle_pocket.difference(wrap)
                        )

                # Preserve the equidistant gap ribbon around the star.
                circle_pocket = self._clean_geometry(
                    unary_union(
                        [
                            circle_pocket,
                            keepout_default.intersection(
                                Point(0.0, 0.0).buffer(r_circ, quad_segs=96)
                            ),
                        ]
                    )
                )

        etch_parts.append(circle_pocket)

        cap_etch = unary_union(etch_parts)
        cap_etch = self._clean_geometry(cap_etch)
        # Soft-merge only; do not drop detached CPW stub etches.
        if cap_etch.geom_type == 'MultiPolygon':
            merged = self._clean_geometry(cap_etch.buffer(1e-6).buffer(-1e-6))
            if merged.geom_type == 'Polygon' and not merged.is_empty:
                cap_etch = merged
            else:
                cap_etch = self._clean_geometry(unary_union(list(cap_etch.geoms)))
        assert not cap_etch.is_empty, 'cap_etch is empty'
        if cap_etch.geom_type == 'MultiPolygon':
            cap_etch = self._clean_geometry(unary_union(list(cap_etch.geoms)))

        # Assemble / place geometries.
        names = ['star'] + list(pad_geoms.keys()) + ['cap_etch']
        geoms = [star] + [pad_geoms[k] for k in pad_geoms.keys()] + [cap_etch]
        geoms = draw.rotate(geoms, p.orientation, origin=(0, 0))
        geoms = draw.translate(geoms, p.pos_x, p.pos_y)
        placed = dict(zip(names, geoms))

        self.add_qgeometry('poly', {'cap_etch': placed['cap_etch']}, layer=p.layer, subtract=True)
        self.add_qgeometry('poly', {'star': placed['star']}, layer=p.layer)
        for name in pad_geoms.keys():
            self.add_qgeometry('poly', {name: placed[name]}, layer=p.layer)

        # Pins at the outer CPW stub ends.
        for i, angle, r_pin in pin_specs:
            half = float(p.cpw_width) / 2.0
            # Local edge at y=r_pin spanning x=±half, then rotate/translate.
            local = LineString([(-half, r_pin), (half, r_pin)])
            edge = draw.translate(
                draw.rotate(
                    draw.rotate(local, np.rad2deg(angle) - 90.0, origin=(0, 0)),
                    p.orientation,
                    origin=(0, 0),
                ),
                p.pos_x,
                p.pos_y,
            )
            pts = np.array(edge.coords)
            self.add_pin(
                f'pad_{i}',
                points=pts,
                width=p.cpw_width,
                gap=p.cpw_gap,
            )
