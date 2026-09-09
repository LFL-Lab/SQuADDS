import numpy as np
from qiskit_metal import Dict, draw
from qiskit_metal.qlibrary.core import QComponent
from shapely import set_precision
from shapely.errors import GEOSException
from shapely.geometry import LineString, box
from shapely.ops import unary_union


class GeneralizedCapConcentric(QComponent):
    """Two-terminal concentric capacitor with elliptical inner and outer bodies.

    Traces (from inside out):
      1. Inner body outer edge (filled pad)
      2. Outer-body inner edge (ring hole)
      3. Outer-body outer edge (ring outline)
      4. Ground-plane inner edge (third ellipse) — gap between outer body and ground

    When ``inner_feed`` is True (default), a south CPW reaches the inner pad through
    a clearance slot in the ring. When False, the outer ring stays continuous.
    A north CPW attaches to the outer ring. ``min_trace_gap`` is enforced between
    consecutive traces and around the inner feed.

    Inspired by the circular pads in Qiskit Metal ``TransmonConcentric``, rewritten
    as a lumped capacitor with independent elliptical traces.
    """

    default_options = Dict(
        inner_radius_x='115um',
        inner_radius_y='115um',
        inner_rotation='0',
        #
        outer_inner_radius_x='150um',
        outer_inner_radius_y='150um',
        outer_inner_rotation='0',
        #
        outer_outer_radius_x='170um',
        outer_outer_radius_y='170um',
        outer_outer_rotation='0',
        #
        # Third trace: inner edge of the ground plane / outer edge of the ground gap.
        ground_inner_radius_x='178um',
        ground_inner_radius_y='178um',
        ground_inner_rotation='0',
        #
        min_trace_gap='6um',
        ellipse_quad_segs='32',
        #
        # True: south CPW feeds the inner pad through a slot in the outer ring.
        # False: outer ring is continuous; no south lead / south pin.
        inner_feed=True,
        #
        north_cpw_length='20um',
        north_cpw_width='10um',
        north_cpw_gap='6um',
        north_cpw_xpos_offset='0um',
        #
        south_cpw_length='20um',
        south_cpw_width='10um',
        south_cpw_gap='6um',
        south_cpw_xpos_offset='0um',
    )

    TOOLTIP = """Elliptical concentric capacitor with independent inner/outer/ground traces."""

    @staticmethod
    def _clean_geometry(geom, precision=1e-6):
        """Repair potentially invalid polygons before/after precision snapping."""
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
    def _um(value_mm):
        return float(value_mm) * 1000.0

    @staticmethod
    def _as_single_polygon(geom, context):
        if geom is None or geom.is_empty:
            raise ValueError(f'{context} is empty')
        if geom.geom_type == 'Polygon':
            return geom
        polygons = [
            part for part in getattr(geom, 'geoms', []) if part.geom_type == 'Polygon'
        ]
        if not polygons:
            raise ValueError(f'{context} has no polygon geometry ({geom.geom_type})')
        polygons = sorted(polygons, key=lambda g: g.area, reverse=True)
        if len(polygons) > 1 and polygons[1].area > 1e-6 * polygons[0].area:
            raise ValueError(
                f'{context} split into multiple pieces; reduce CPW/slot width '
                'or enlarge the outer-inner ellipse'
            )
        return polygons[0]

    def _make_ellipse(self, radius_x, radius_y, rotation_deg, quad_segs):
        assert radius_x > 0.0, 'ellipse radius_x must be > 0'
        assert radius_y > 0.0, 'ellipse radius_y must be > 0'
        circle = draw.Point(0, 0).buffer(1.0, quad_segs=int(quad_segs))
        ellipse = draw.scale(circle, xfact=float(radius_x), yfact=float(radius_y), origin=(0, 0))
        if abs(float(rotation_deg)) > 1e-12:
            ellipse = draw.rotate(ellipse, float(rotation_deg), origin=(0, 0))
        return self._clean_geometry(ellipse)

    def _assert_min_trace_gap(self, inner, outer_inner, outer_outer, ground_inner, min_trace_gap):
        gap_um = self._um(min_trace_gap)
        # Ellipse polygonization can shrink measured boundary gaps by a few nm.
        tol = 1e-5

        assert not inner.is_empty, 'inner ellipse is empty'
        assert not outer_inner.is_empty, 'outer-inner ellipse is empty'
        assert not outer_outer.is_empty, 'outer-outer ellipse is empty'
        assert not ground_inner.is_empty, 'ground-inner ellipse is empty'

        assert outer_inner.covers(inner), (
            'inner ellipse must lie fully inside the outer ring inner trace'
        )
        inner_to_hole = inner.boundary.distance(outer_inner.boundary)
        assert inner_to_hole + tol >= min_trace_gap, (
            f'distance between inner pad and outer-ring inner trace is '
            f'{self._um(inner_to_hole):.3f} um, which is below min_trace_gap={gap_um:.3f} um'
        )

        assert outer_outer.covers(outer_inner), (
            'outer-ring inner trace must lie fully inside the outer-ring outer trace'
        )
        ring_width = outer_inner.boundary.distance(outer_outer.boundary)
        assert ring_width + tol >= min_trace_gap, (
            f'outer-ring width between its inner and outer traces is '
            f'{self._um(ring_width):.3f} um, which is below min_trace_gap={gap_um:.3f} um'
        )

        assert ground_inner.covers(outer_outer), (
            'outer-ring outer trace must lie fully inside the ground-inner trace'
        )
        ground_gap = outer_outer.boundary.distance(ground_inner.boundary)
        assert ground_gap + tol >= min_trace_gap, (
            f'gap between outer-ring outer trace and ground-inner trace is '
            f'{self._um(ground_gap):.3f} um, which is below min_trace_gap={gap_um:.3f} um'
        )

    def _cpw_box(self, xpos_offset, width, y0, y1):
        half = float(width) / 2.0
        x0 = float(xpos_offset) - half
        x1 = float(xpos_offset) + half
        ymin, ymax = (y0, y1) if y0 <= y1 else (y1, y0)
        return box(x0, ymin, x1, ymax)

    def _use_inner_feed(self):
        """South CPW through a ring slot to the inner pad (False => continuous ring)."""
        p = self.p
        raw = getattr(p, 'inner_feed', True)
        if isinstance(raw, str):
            enabled = raw.strip().lower() in ('1', 'true', 'yes', 'on')
        else:
            enabled = bool(raw)
        return enabled and float(p.south_cpw_length) > 1e-12

    def _make_inner_body(self, inner, south_pin_y):
        p = self.p
        if not self._use_inner_feed():
            return inner
        feed = self._cpw_box(p.south_cpw_xpos_offset, p.south_cpw_width, south_pin_y, 0.0)
        return self._clean_geometry(unary_union([inner, feed]))

    def _make_outer_body(self, outer_inner, outer_outer, inner_body, south_pin_y, north_pin_y):
        p = self.p
        outer_ring = self._clean_geometry(outer_outer.difference(outer_inner))
        assert not outer_ring.is_empty, 'outer ring is empty; increase outer_outer radii'

        if self._use_inner_feed():
            clearance = max(float(p.min_trace_gap), float(p.south_cpw_gap))
            slot_width = float(p.south_cpw_width) + 2.0 * clearance
            slot = self._cpw_box(p.south_cpw_xpos_offset, slot_width, south_pin_y, 0.0)
            outer_ring = self._as_single_polygon(
                self._clean_geometry(outer_ring.difference(slot)),
                'outer ring after inner-feed slot',
            )

        if p.north_cpw_length > 1e-12:
            north_lead = self._cpw_box(
                p.north_cpw_xpos_offset, p.north_cpw_width, 0.0, north_pin_y
            )
            north_lead = self._clean_geometry(north_lead.difference(outer_inner))
            outer_ring = self._clean_geometry(unary_union([outer_ring, north_lead]))

        keepout = inner_body.buffer(max(float(p.min_trace_gap) - 1e-9, 0.0))
        outer_ring = self._as_single_polygon(
            self._clean_geometry(outer_ring.difference(keepout)),
            'outer ring after min_trace_gap keepout',
        )
        return outer_ring

    def _make_cap_etch(self, ground_inner, south_pin_y, north_pin_y):
        """Pocket subtract: interior of the ground-inner ellipse plus CPW gap corridors."""
        p = self.p
        parts = [ground_inner]

        if self._use_inner_feed():
            south_etch_width = float(p.south_cpw_width) + 2.0 * float(p.south_cpw_gap)
            parts.append(
                self._cpw_box(p.south_cpw_xpos_offset, south_etch_width, south_pin_y, 0.0)
            )
        if p.north_cpw_length > 1e-12:
            north_etch_width = float(p.north_cpw_width) + 2.0 * float(p.north_cpw_gap)
            parts.append(
                self._cpw_box(p.north_cpw_xpos_offset, north_etch_width, 0.0, north_pin_y)
            )

        return self._clean_geometry(unary_union(parts))

    def make(self):
        p = self.p
        quad_segs = int(p.ellipse_quad_segs)
        assert quad_segs >= 4, 'ellipse_quad_segs must be >= 4'
        assert p.min_trace_gap > 0.0, 'min_trace_gap must be > 0'

        inner = self._make_ellipse(p.inner_radius_x, p.inner_radius_y, p.inner_rotation, quad_segs)
        outer_inner = self._make_ellipse(
            p.outer_inner_radius_x, p.outer_inner_radius_y, p.outer_inner_rotation, quad_segs
        )
        outer_outer = self._make_ellipse(
            p.outer_outer_radius_x, p.outer_outer_radius_y, p.outer_outer_rotation, quad_segs
        )
        ground_inner = self._make_ellipse(
            p.ground_inner_radius_x, p.ground_inner_radius_y, p.ground_inner_rotation, quad_segs
        )
        self._assert_min_trace_gap(inner, outer_inner, outer_outer, ground_inner, p.min_trace_gap)

        _, ground_miny, _, ground_maxy = ground_inner.bounds
        use_inner_feed = self._use_inner_feed()
        south_pin_y = ground_miny - float(p.south_cpw_length) if use_inner_feed else ground_miny
        north_pin_y = ground_maxy + float(p.north_cpw_length)

        cap_inner_body = self._make_inner_body(inner, south_pin_y)
        cap_outer_body = self._make_outer_body(
            outer_inner, outer_outer, cap_inner_body, south_pin_y, north_pin_y
        )

        body_gap = cap_inner_body.distance(cap_outer_body)
        assert body_gap + 1e-9 >= p.min_trace_gap, (
            f'after CPW/slot construction, inner and outer traces are '
            f'{self._um(body_gap):.3f} um apart, below min_trace_gap='
            f'{self._um(p.min_trace_gap):.3f} um'
        )

        cap_etch = self._make_cap_etch(ground_inner, south_pin_y, north_pin_y)

        c_items = [cap_inner_body, cap_outer_body, cap_etch]
        c_items = draw.rotate(c_items, p.orientation, origin=(0, 0))
        c_items = draw.translate(c_items, p.pos_x, p.pos_y)
        [cap_inner_body, cap_outer_body, cap_etch] = c_items

        self.add_qgeometry('poly', {'cap_etch': cap_etch}, layer=p.layer, subtract=True)
        self.add_qgeometry('poly', {'cap_inner_body': cap_inner_body}, layer=p.layer)
        self.add_qgeometry('poly', {'cap_outer_body': cap_outer_body}, layer=p.layer)

        if use_inner_feed:
            south_cpw_edge_coords = np.array(
                [
                    (p.south_cpw_xpos_offset - p.south_cpw_width / 2, south_pin_y),
                    (p.south_cpw_xpos_offset + p.south_cpw_width / 2, south_pin_y),
                ]
            )
            south_line = draw.translate(
                draw.rotate(LineString(south_cpw_edge_coords), p.orientation, origin=(0, 0)),
                p.pos_x,
                p.pos_y,
            )
            south_cpw_edge_coords = np.array(south_line.coords)
            self.add_pin(
                'south_end',
                points=south_cpw_edge_coords[::-1],
                width=p.south_cpw_width,
                gap=p.south_cpw_gap,
            )

        if p.north_cpw_length > 1e-12:
            north_cpw_edge_coords = np.array(
                [
                    (p.north_cpw_xpos_offset - p.north_cpw_width / 2, north_pin_y),
                    (p.north_cpw_xpos_offset + p.north_cpw_width / 2, north_pin_y),
                ]
            )
            north_line = draw.translate(
                draw.rotate(LineString(north_cpw_edge_coords), p.orientation, origin=(0, 0)),
                p.pos_x,
                p.pos_y,
            )
            north_cpw_edge_coords = np.array(north_line.coords)
            self.add_pin(
                'north_end',
                points=north_cpw_edge_coords,
                width=p.north_cpw_width,
                gap=p.north_cpw_gap,
            )
