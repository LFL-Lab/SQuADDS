import numpy as np
import shapely
from qiskit_metal import Dict, draw
from qiskit_metal.qlibrary.core import QComponent
from shapely import affinity
from shapely.geometry import Point, box


def _parse_list(raw) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return [str(v).strip() for v in raw]
    text = str(raw).strip()
    return [t.strip() for t in text.split(",") if t.strip()] if text else []


def _parse_flag(token) -> bool:
    if isinstance(token, bool):
        return token
    return str(token).strip().lower() in ("1", "true", "yes", "on", "g", "grounded")


class ShieldedCapConcentric(QComponent):
    """Concentric capacitor with guard rings between the inner disk and the outer ring.

    From the centre out: an inner disk (one conductor), one or more guard rings, an outer ring fed by a north CPW
    (another conductor) and the circular ground opening. Each guard ring is either floating (its own conductor) or
    grounded, in which case a radial strap joins it to the ground plane through slots in every ring outside it, so the
    outer ring becomes a split ring. All outlines are circles stretched by ``aspect`` along x.

    Options (lengths with units; lists comma-separated, one entry per guard ring from the inside out):
        * inner_radius: radius of the inner disk
        * aspect: x/y stretch of every outline
        * guard_gaps, guard_widths: gap to the previous body and width of each guard ring
        * guard_grounded: 1 (grounded through a strap) or 0 (floating) per guard ring
        * ring_gap, ring_width: gap between the last guard and the outer ring, and the outer ring width
        * ground_gap: gap between the outer ring and the ground opening
        * strap_width, strap_gap: width of the grounding strap and the clearance of the slots it passes through
        * cpw_length, cpw_width, cpw_gap: the north feed line beyond the ground opening
        * circle_quad_segs: segments per quarter circle

    Qgeometry names: ``inner`` (the disk), ``outer_ring`` (with its feed), ``guard_<k>`` for each floating guard
    (k counts floating guards from 1), and the subtractive ``cap_etch``. This is the component of the tl-squadds
    shielded concentric family (exp15).
    """

    default_options = Dict(
        inner_radius="60um",
        aspect="1.0",
        guard_gaps="12um",
        guard_widths="10um",
        guard_grounded="0",
        ring_gap="12um",
        ring_width="30um",
        ground_gap="20um",
        strap_width="10um",
        strap_gap="6um",
        cpw_length="20um",
        cpw_width="10um",
        cpw_gap="6um",
        circle_quad_segs="128",
    )

    TOOLTIP = """Concentric capacitor with floating or grounded guard rings."""

    def _guards(self):
        gaps = _parse_list(self.options.get("guard_gaps"))
        widths = _parse_list(self.options.get("guard_widths"))
        grounded = _parse_list(self.options.get("guard_grounded"))
        if len(widths) == 1 and len(gaps) > 1:
            widths = widths * len(gaps)
        if len(grounded) == 1 and len(gaps) > 1:
            grounded = grounded * len(gaps)
        assert len(gaps) == len(widths) == len(grounded), "guard_gaps, guard_widths and guard_grounded differ in length"
        return [
            (self._um(self.parse_value(g)), self._um(self.parse_value(w)), _parse_flag(f))
            for g, w, f in zip(gaps, widths, grounded)
        ]

    @staticmethod
    def _um(value_mm) -> float:
        return float(value_mm) * 1e3

    def geometry_um(self) -> dict:
        """Shapes at the origin in micrometres: metal per qgeometry name, the etch and the feed pin height."""
        p = self.p
        a = float(p.aspect)
        segs = int(p.circle_quad_segs)
        cpw_w, cpw_g = self._um(p.cpw_width), self._um(p.cpw_gap)
        strap_w, strap_gap = self._um(p.strap_width), self._um(p.strap_gap)
        gap_n, w_n, gap_g = self._um(p.ring_gap), self._um(p.ring_width), self._um(p.ground_gap)

        def ring(r_in, r_out):
            return affinity.scale(Point(0, 0).buffer(r_out, segs).difference(Point(0, 0).buffer(r_in, segs)), a, 1.0)

        r = self._um(p.inner_radius)
        metal = {"inner": affinity.scale(Point(0, 0).buffer(r, segs), a, 1.0)}
        rings, n_float = [], 0
        for k, (gap, width, grounded) in enumerate(self._guards(), 1):
            r0 = r + gap
            r = r0 + width
            if not grounded:
                n_float += 1
            rings.append((f"guard_{n_float}" if not grounded else f"grounded_{k}", ring(r0, r), grounded, r0))
        rn0 = r + gap_n
        rn1 = rn0 + w_n
        r_pocket = rn1 + gap_g
        n_ring = ring(rn0, rn1)
        straps, slot_half = [], None
        for _, _, grounded, r0 in rings:
            if grounded:
                straps.append(box(-strap_w / 2, -r_pocket - 1, strap_w / 2, -r0 + 1e-3))
                slot_half = strap_w / 2 + strap_gap
        shapes = {}
        for name, shape, grounded, r0 in rings:
            if slot_half is not None and not grounded and any(g and r0 > rr0 for _, _, g, rr0 in rings):
                shape = shape.difference(box(-slot_half, -r_pocket - 1, slot_half, 0))
            shapes[name] = (shape, grounded)
        if slot_half is not None:
            n_ring = n_ring.difference(box(-slot_half, -r_pocket - 1, slot_half, 0))
        y_pin = r_pocket + self._um(p.cpw_length)
        trace = box(-cpw_w / 2, rn1 - w_n / 2, cpw_w / 2, y_pin)
        metal["outer_ring"] = shapely.union_all([n_ring, trace])
        for name, (shape, grounded) in shapes.items():
            if not grounded:
                metal[name] = shape
        grounded_metal = shapely.union_all([s for s, g in shapes.values() if g] + straps) if straps else None
        pocket = affinity.scale(Point(0, 0).buffer(r_pocket, segs), a, 1.0).union(
            box(-cpw_w / 2 - cpw_g, 0, cpw_w / 2 + cpw_g, y_pin)
        )
        etch = pocket.difference(grounded_metal) if grounded_metal is not None else pocket
        return {"metal": metal, "etch": etch, "pin_y": y_pin, "r_pocket": r_pocket}

    def make(self):
        p = self.p
        g = self.geometry_um()

        def place(shape):
            shape = affinity.scale(shape, 1e-3, 1e-3, origin=(0, 0))
            return draw.translate(draw.rotate(shape, p.orientation, origin=(0, 0)), p.pos_x, p.pos_y)

        self.add_qgeometry("poly", {"cap_etch": place(g["etch"])}, subtract=True, layer=p.layer)
        for name, shape in g["metal"].items():
            self.add_qgeometry("poly", {name: place(shape)}, layer=p.layer)
        y, half = g["pin_y"] * 1e-3, p.cpw_width / 2
        pin = draw.translate(
            draw.rotate(draw.LineString([(-half, y), (half, y)]), p.orientation, origin=(0, 0)), p.pos_x, p.pos_y
        )
        self.add_pin("readout", points=np.array(pin.coords), width=p.cpw_width, gap=p.cpw_gap)


def shielded_options_from_generator(params: dict) -> dict:
    """Map a tl-squadds exp15 generator record (lengths as floats in micrometres) to component options."""

    def um(v):
        return f"{float(v)!r}um"

    guards = params["guards"]
    return {
        "inner_radius": um(params["r_s"]),
        "aspect": repr(float(params["aspect"])),
        "guard_gaps": ",".join(um(g["gap"]) for g in guards),
        "guard_widths": ",".join(um(g["width"]) for g in guards),
        "guard_grounded": ",".join("1" if g["grounded"] else "0" for g in guards),
        "ring_gap": um(params["gap_n"]),
        "ring_width": um(params["w_n"]),
        "ground_gap": um(params["gap_g"]),
        "strap_width": um(params["strap_w"]),
        "strap_gap": um(params["strap_gap"]),
        "cpw_length": um(params.get("cpw_len", 20.0)),
        "cpw_width": "10um",
        "cpw_gap": "6um",
    }
