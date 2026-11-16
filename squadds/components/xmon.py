from qiskit_metal import Dict, draw
from qiskit_metal.qlibrary.qubits.transmon_cross import TransmonCross as MetalTransmonCross


class Xmon(MetalTransmonCross):
    """Xmon capacitor: a four-arm cross in a ground pocket with one or two claw couplers.

    The geometry is quantum-metal's ``TransmonCross`` (cross, cross etch and claw connectors, pins included)
    without the junction, so the layout is the capacitive network of an Xmon: the cross is one conductor and
    every claw another. The ``readout`` claw sits on the north arm by default; add a second claw with another
    entry in ``connection_pads`` (for example ``bus`` with ``connector_location`` ``"0"`` (west) or ``"180"`` (east); as in the dataset, any angle above 135 is the east arm).

    Pins are quantum-metal's: the claw pin line runs from ``-claw_cpw_width/2`` to ``+claw_width/2`` across the
    claw feed, so its middle is offset from the feed axis when the two widths differ. The dataset's port stubs and
    ground end slots were cut at these pins.

    This is the component of the tl-squadds Xmon family (exp13). It differs from
    :class:`squadds.components.qubits.TransmonCross`, which draws the SQuADDS qubit with its junction options.
    """

    default_options = Dict(
        cross_width="20um",
        cross_length="200um",
        cross_gap="20um",
        chip="main",
        connection_pads=Dict(
            readout=Dict(
                connector_location="90",
                connector_type="0",
                claw_length="60um",
                claw_width="10um",
                claw_gap="6um",
                ground_spacing="8um",
                claw_cpw_length="40um",
                claw_cpw_width="10um",
            )
        ),
        _default_connection_pads=Dict(
            connector_type="0",
            claw_length="30um",
            ground_spacing="5um",
            claw_width="10um",
            claw_gap="6um",
            claw_cpw_length="40um",
            claw_cpw_width="10um",
            connector_location="0",
        ),
    )

    component_metadata = Dict(short_name="Xmon", _qgeometry_table_poly="True")

    TOOLTIP = """Xmon cross with one or two claw couplers (no junction)."""

    def make_pocket(self):
        """Cross and its etch, as in quantum-metal's TransmonCross, without the junction."""
        p = self.p
        cross_line = draw.shapely.ops.unary_union(
            [
                draw.LineString([(0, p.cross_length), (0, -p.cross_length)]),
                draw.LineString([(p.cross_length, 0), (-p.cross_length, 0)]),
            ]
        )
        cross = cross_line.buffer(p.cross_width / 2, cap_style=2)
        cross_etch = cross.buffer(p.cross_gap, cap_style=3, join_style=2)
        cross, cross_etch = draw.translate(
            draw.rotate([cross, cross_etch], p.orientation, origin=(0, 0)), p.pos_x, p.pos_y
        )
        self.add_qgeometry("poly", dict(cross=cross), chip=p.chip)
        self.add_qgeometry("poly", dict(cross_etch=cross_etch), subtract=True, chip=p.chip)

    def make_connection_pad(self, name: str):
        """Claw connector as drawn for the dataset (quantum-metal 0.5): ``connector_location`` above 135 is the east
        arm, above 45 the north arm, otherwise the west arm. quantum-metal 0.9 maps 270 to the south arm instead."""
        p = self.p
        pc = p.connection_pads[name]
        c_g, c_l, c_w = pc.claw_gap, pc.claw_length, pc.claw_width
        c_c_w, c_c_l, g_s = pc.claw_cpw_width, pc.claw_cpw_length, pc.ground_spacing
        con_loc = pc.connector_location

        claw_cpw = draw.box(-c_w, -c_c_w / 2, -c_c_l - c_w, c_c_w / 2)
        if pc.connector_type == 0:
            t_claw_height = 2 * c_g + 2 * c_w + 2 * g_s + 2 * p.cross_gap + p.cross_width
            claw_base = draw.box(-c_w, -t_claw_height / 2, c_l, t_claw_height / 2)
            claw_subtract = draw.box(0, -t_claw_height / 2 + c_w, c_l, t_claw_height / 2 - c_w)
            connector_arm = draw.shapely.ops.unary_union([claw_base.difference(claw_subtract), claw_cpw])
        else:
            connector_arm = draw.box(0, -c_w / 2, -4 * c_w, c_w / 2)
        connector_etcher = draw.buffer(connector_arm, c_g)
        port_line = draw.LineString([(-c_c_l - c_w, -c_c_w / 2), (-c_c_l - c_w, c_w / 2)])

        claw_rotate = 180 if con_loc > 135 else (-90 if con_loc > 45 else 0)
        polys = draw.translate(
            [connector_arm, connector_etcher, port_line], -(p.cross_length + p.cross_gap + g_s + c_g), 0
        )
        polys = draw.rotate(polys, claw_rotate, origin=(0, 0))
        polys = draw.rotate(polys, p.orientation, origin=(0, 0))
        connector_arm, connector_etcher, port_line = draw.translate(polys, p.pos_x, p.pos_y)

        self.add_qgeometry("poly", {f"{name}_connector_arm": connector_arm}, chip=p.chip)
        self.add_qgeometry("poly", {f"{name}_connector_etcher": connector_etcher}, subtract=True, chip=p.chip)
        self.add_pin(name, port_line.coords, c_c_w)
