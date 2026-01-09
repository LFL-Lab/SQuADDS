import shapely
from qiskit_metal import Dict, draw
from qiskit_metal.qlibrary.core.base import QComponent

#!TODO: make JJ_length a parameter


class JjDolan(QComponent):
    """
    The base "JjDolan" inherits the "QComponent" class.

    NOTE TO USER: Please be aware that when designing with this
    qcomponent, one must take care in accounting for the junction
    qgeometry when exporting to to GDS and/or to a simulator. This
    qcomponent should not be rendered for EM simulation.

    This creates a "Dolan"-style Josephson Junction consisting
    a single junction slightly separated, adhearing to LFL's prefererances.

    Default Options:
        * bridge_length: '28um' -- lenght between pads
        * JJ_width:   '0.188um' -- Josephson Junction width
        * bridge_layer:       1 -- GDS layer for bridge, make sure it's different from qubit + JJs
        * JJ_layer:           2 -- GDS layer for JJ, make sure it's different from qubit + bridge
    """

    default_options = Dict(bridge_length="28um", JJ_width="0.188um", bridge_layer=20, JJ_layer=60)
    """Default options."""

    component_metadata = Dict(short_name="Cross", _qgeometry_table_poly="True", _qgeometry_table_junction="True")
    """Component metadata"""

    TOOLTIP = """Josephson Junctions used at LFL"""

    def make(self):
        pad_length = 0.007  # mm
        pad_width = 0.010  # mm
        triangle_width = 0.006  # mm
        triangle_height = 0.003  # mm
        bridge_width = 0.002  # mm

        # Here's our user defined variables
        p = self.parse_options()

        ### Bridge Layer
        # Bottom pad
        pad = draw.rectangle(pad_width, pad_length, 0, 0)
        triangle = draw.Polygon([(-triangle_width / 2, 0), (triangle_width / 2, 0), (0, triangle_height)])
        triangle = draw.translate(triangle, 0, pad_length / 2)
        pad = draw.union(pad, triangle)
        pad = draw.translate(pad, 0, -p.bridge_length / 2 - pad_length / 2)

        # Top pad
        t_pad = draw.rotate(pad, 180, origin=(0, 0))

        # Bridge
        bridge = draw.rectangle(bridge_width, p.bridge_length, 0, 0)

        # Final bridge
        bridge = draw.union(bridge, pad, t_pad)

        ### JJ Layer
        jjl2 = 0.002  # 2.0um
        finger_length = 0.00136  # 1.36um
        JJ_taper = jjl2 - finger_length  # 0.5um
        jj1_gap = 0.000140  # 140nm

        # Make cutout on the bridge
        JJ = draw.Polygon(
            [
                (-bridge_width / 2, 0),
                (bridge_width / 2, 0),
                (p.JJ_width / 2, JJ_taper),
                (p.JJ_width / 2, JJ_taper + finger_length - jj1_gap),
                (-p.JJ_width / 2, JJ_taper + finger_length - jj1_gap),
                (-p.JJ_width / 2, JJ_taper),
            ]
        )
        cutout = draw.Polygon(
            [(-bridge_width / 2, 0), (bridge_width / 2, 0), (bridge_width / 2, jjl2), (-bridge_width / 2, jjl2)]
        )

        cutout = draw.subtract(cutout, JJ)
        bridge = draw.subtract(bridge, cutout)

        # Now you can draw your JJ
        JJ = draw.Polygon(
            [
                (-bridge_width / 2, 0),
                (bridge_width / 2, 0),
                (p.JJ_width / 2, JJ_taper),
                (p.JJ_width / 2, JJ_taper + finger_length),
                (-p.JJ_width / 2, JJ_taper + finger_length),
                (-p.JJ_width / 2, JJ_taper),
            ]
        )

        polys = [bridge, JJ]
        polys = draw.rotate(polys, p.orientation, origin=(0, 0))
        polys = draw.translate(polys, p.pos_x, p.pos_y)

        bridge, JJ = polys

        ### Add everything to QGeometry
        self.add_qgeometry("poly", {"bridge": bridge}, layer=p.bridge_layer, subtract=False)
        self.add_qgeometry("poly", {"JJ": JJ}, layer=p.JJ_layer, subtract=False)

        # Add a LineString for the junction
        coords = list(JJ.exterior.coords)
        # Use the midpoints of the two short sides (indices 0-1 and 3-4 for this polygon)
        mid1 = ((coords[0][0] + coords[1][0]) / 2, (coords[0][1] + coords[1][1]) / 2)
        mid2 = ((coords[3][0] + coords[4][0]) / 2, (coords[3][1] + coords[4][1]) / 2)
        ls = shapely.geometry.LineString([mid1, mid2])
        self.add_qgeometry("junction", {"design": ls}, width=p.JJ_width, layer=p.JJ_layer)


class SquidLoopDolan(QComponent):
    """
    The base "SquidLoopDolan" inherits the "QComponent" class.

    NOTE TO USER: Please be aware that when designing with this
    qcomponent, one must take care in accounting for the junction
    qgeometry when exporting to to GDS and/or to a simulator. This
    qcomponent should not be rendered for EM simulation.

    This creates a "Dolan"-style SQUID loop consisting
    of a slightly separated loop with 2 symmetrical JJs on each side.

    Default Options:
        * SQUID_width: '63um' -- inner width (x-axis) of SQUID loop
        * SQUID_length: '9um' -- inner length (y-axis) of SQUID loop
        * stem1_length: '7um' -- length between bottom pad and loop
        * stem2_length: '7um' -- length between side pad and loop
        * stem1_offset: '0um' -- aligns stem1 some distance away from center
        * stem2_offset: '0um' -- aligns stem2 some distance away from center
        * JJ_width:   '0.3um' -- correlated to setting the Lj of your SQUID
        * JJ_offset:    '0um' -- aligns the JJs some distance away from center
        * mirror:           0 -- input 0 or 1, want to mirror about the x-axis?
        * SQUID_layer:      1 -- GDS layer for loop, make sure it's different from qubit + JJs
        * JJ_layer:         2 -- GDS layer for JJs, make sure it's different from qubit + loop


    """

    # Default drawing options
    default_options = Dict(
        SQUID_width="63um",
        SQUID_length="9um",
        SQUID_thickness="2um",
        stem1_length="7um",
        stem2_length="7um",
        stem1_offset="0um",
        stem2_offset="0um",
        JJ_width="0.3um",
        JJ_offset="0um",
        JJ_flip=False,
        mirror=0,
        SQUID_layer=20,
        JJ_layer=60,
    )
    """Default drawing options"""

    # Name prefix of component, if user doesn't provide name
    component_metadata = Dict(short_name="component")
    """Component metadata"""

    def make(self):
        self.make_loop()
        self.make_junction()

    def make_loop(self):
        """
        Makes the SQUID loop structure
        """
        p = self.parse_options()

        ### Make first pad
        pad = draw.Polygon([(0, 0), (0.010, 0), (0.010, 0.016), (0, 0.016)])
        pad = draw.translate(pad, -0.005, 0)

        ### Make first tappered structure
        tapper = draw.Polygon([(-0.0062 / 2, 0), (0.0062 / 2, 0), (0, 0.003125)])

        stem1 = draw.Polygon(
            [
                (-p.SQUID_thickness / 2, 0),
                (p.SQUID_thickness / 2, 0),
                (p.SQUID_thickness / 2, p.stem1_length),
                (-p.SQUID_thickness / 2, p.stem1_length),
            ]
        )
        tapper = draw.translate(tapper, 0, 0.016)
        stem1 = draw.translate(stem1, 0, 0.016)
        tappered_stem = draw.union(stem1, tapper)

        checkpoint = draw.union(tappered_stem, pad)

        ### make loop by subtracting inner rectangle from outter rectangle
        outter = draw.Polygon(
            [
                (-p.SQUID_width / 2 - p.SQUID_thickness, -p.SQUID_length / 2 - p.SQUID_thickness),
                (-p.SQUID_width / 2 - p.SQUID_thickness, p.SQUID_length / 2 + p.SQUID_thickness),
                (p.SQUID_width / 2 + p.SQUID_thickness, p.SQUID_length / 2 + p.SQUID_thickness),
                (p.SQUID_width / 2 + p.SQUID_thickness, -p.SQUID_length / 2 - p.SQUID_thickness),
            ]
        )
        inner = draw.Polygon(
            [
                (-p.SQUID_width / 2, -p.SQUID_length / 2),
                (-p.SQUID_width / 2, p.SQUID_length / 2),
                (p.SQUID_width / 2, p.SQUID_length / 2),
                (p.SQUID_width / 2, -p.SQUID_length / 2),
            ]
        )

        loop = draw.subtract(outter, inner)
        loop = draw.translate(loop, 0, p.SQUID_length / 2 + p.SQUID_thickness + 0.016 + p.stem1_length)

        ### Checkpoint, let's prep to make the 2nd and final stem

        total = [checkpoint, loop]
        total = draw.rotate(total, 90, origin=(0, 0))
        total = draw.translate(
            total,
            0.016 + p.stem1_length + p.SQUID_length / 2 + 2 * p.SQUID_thickness / 2,
            p.SQUID_width / 2 + p.SQUID_thickness,
        )

        checkpoint, loop = total

        ### Make the 2nd stemp

        stem2 = draw.Polygon(
            [
                (-p.SQUID_thickness / 2, 0),
                (p.SQUID_thickness / 2, 0),
                (p.SQUID_thickness / 2, p.stem2_length),
                (-p.SQUID_thickness / 2, p.stem2_length),
            ]
        )
        stem2 = draw.translate(stem2, -(p.SQUID_length / 2 + p.SQUID_thickness / 2), 0.016)
        tapper = draw.translate(tapper, -(p.SQUID_length / 2 + p.SQUID_thickness / 2), 0)
        pad = draw.translate(pad, -(p.SQUID_length / 2 + p.SQUID_thickness / 2), 0)

        checkpoint2 = draw.union(stem2, tapper, pad)
        checkpoint2 = draw.translate(checkpoint2, 0, -p.stem2_length - 0.016)

        ### Choose offsets for pads

        checkpoint = draw.translate(checkpoint, 0, p.stem1_offset)

        checkpoint2 = draw.translate(checkpoint2, p.stem2_offset, 0)

        ### We the general structure, we just need to remove
        ### part of the loop for the JJs. Centering middle of loop.

        final_loop = [checkpoint, checkpoint2, loop]
        final_loop = draw.translate(
            final_loop,
            -(0.016 + p.stem1_length + p.SQUID_length / 2 + 2 * p.SQUID_thickness / 2),
            -(p.SQUID_width / 2 + p.SQUID_thickness),
        )
        final_loop = draw.rotate(final_loop, -90, origin=(0, 0))

        checkpoint, checkpoint2, loop = final_loop

        if p.mirror == 1:
            checkpoint2 = draw.scale(checkpoint2, -1, 1, origin=(0, 0))

        final_loop = draw.union(checkpoint, checkpoint2, loop)

        final_loop = draw.translate(final_loop, 0, -(0.016 + p.stem1_length + p.SQUID_thickness + p.SQUID_length / 2))

        ### Subtract pocket for JJs

        JJ_taper = 0.002 - 0.00136  # 0.5um
        finger_length = 0.00136  # 1.36um
        x_gap = 0.00015  # 0.14um

        JJ = draw.Polygon(
            [
                (-p.SQUID_thickness / 2, 0),
                (p.SQUID_thickness / 2, 0),
                (p.JJ_width / 2, JJ_taper),
                (p.JJ_width / 2, JJ_taper + finger_length - x_gap),
                (-p.JJ_width / 2, JJ_taper + finger_length - x_gap),
                (-p.JJ_width / 2, JJ_taper),
            ]
        )

        # I added this x_gap as buffer to fully subtract part of 'final_loop',
        # I noticed that when i set 'JJ_flip = True', it would leave some residual loop
        # Scuffed fix, but will have to suffice for now
        pocket = draw.Polygon(
            [
                (-p.SQUID_thickness / 2 - x_gap, 0),
                (p.SQUID_thickness / 2 + x_gap, 0),
                (p.SQUID_thickness / 2 + x_gap, JJ_taper + finger_length),
                (-p.SQUID_thickness / 2 - x_gap, JJ_taper + finger_length),
            ]
        )

        pocket = draw.subtract(pocket, JJ)

        if p.JJ_flip:
            pocket = draw.rotate(pocket, 180, origin=(0, (JJ_taper + finger_length - x_gap) / 2))

        pocket2 = draw.translate(pocket, p.SQUID_width / 2 + p.SQUID_thickness / 2, -(JJ_taper + finger_length) / 2)
        pocket = draw.translate(pocket, -(p.SQUID_width / 2 + p.SQUID_thickness / 2), -(JJ_taper + finger_length) / 2)

        final_loop = draw.subtract(final_loop, pocket2)
        final_loop = draw.subtract(final_loop, pocket)

        final_loop = draw.rotate(final_loop, p.orientation, origin=(0, 0))
        final_loop = draw.translate(final_loop, p.pos_x, p.pos_y)

        ### Add to QGeometry!
        self.add_qgeometry("poly", {"final_loop": final_loop}, layer=p.SQUID_layer, subtract=False)

    def make_junction(self):
        """Makes the JJs"""
        p = self.parse_options()
        # These are some parameters given by LFL's design prefererances
        JJ_taper = 0.002 - 0.00136  # 0.5um
        finger_length = 0.00136  # 1.36um
        x_gap = 0.00015  # 0.14um

        JJ = draw.Polygon(
            [
                (-p.SQUID_thickness / 2, 0),
                (p.SQUID_thickness / 2, 0),
                (p.JJ_width / 2, JJ_taper),
                (p.JJ_width / 2, JJ_taper + finger_length),
                (-p.JJ_width / 2, JJ_taper + finger_length),
                (-p.JJ_width / 2, JJ_taper),
            ]
        )

        if p.JJ_flip:
            JJ = draw.rotate(JJ, 180, origin=(0, (JJ_taper + finger_length - x_gap) / 2))

        JJ2 = draw.translate(JJ, p.SQUID_width / 2 + p.SQUID_thickness / 2, -(JJ_taper + finger_length) / 2)
        JJ = draw.translate(JJ, -(p.SQUID_width / 2 + p.SQUID_thickness / 2), -(JJ_taper + finger_length) / 2)

        final_design = [JJ, JJ2]

        final_design = draw.rotate(final_design, p.orientation, origin=(0, 0))
        final_design = draw.translate(final_design, p.pos_x, p.pos_y)

        JJ, JJ2 = final_design

        ### Add to QGeometry
        self.add_qgeometry("poly", {"JJ": JJ}, layer=p.JJ_layer, subtract=False)
        self.add_qgeometry("poly", {"JJ2": JJ2}, layer=p.JJ_layer, subtract=False)

        # Add LineStrings for both junctions
        coords1 = list(JJ.exterior.coords)
        mid1a = ((coords1[0][0] + coords1[1][0]) / 2, (coords1[0][1] + coords1[1][1]) / 2)
        mid1b = ((coords1[3][0] + coords1[4][0]) / 2, (coords1[3][1] + coords1[4][1]) / 2)
        ls1 = shapely.geometry.LineString([mid1a, mid1b])
        self.add_qgeometry("junction", {"JJ": ls1}, width=p.JJ_width, layer=p.JJ_layer)

        coords2 = list(JJ2.exterior.coords)
        mid2a = ((coords2[0][0] + coords2[1][0]) / 2, (coords2[0][1] + coords2[1][1]) / 2)
        mid2b = ((coords2[3][0] + coords2[4][0]) / 2, (coords2[3][1] + coords2[4][1]) / 2)
        ls2 = shapely.geometry.LineString([mid2a, mid2b])
        self.add_qgeometry("junction", {"JJ2": ls2}, width=p.JJ_width, layer=p.JJ_layer)
