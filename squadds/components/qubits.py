import numpy as np
from qiskit_metal import Dict, draw
from qiskit_metal.qlibrary.core import BaseQubit

from .jjs import JjDolan


class TransmonCross(BaseQubit):  # pylint: disable=invalid-name
    """The base `TransmonCross` class.

    Inherits `BaseQubit` class.

    Simple Metal Transmon Cross object. Creates the X cross-shaped island,
    the "junction" on the south end, and up to 3 connectors on the remaining arms
    (claw or gap).

    'claw_width' and 'claw_gap' define the width/gap of the CPW line that
    makes up the connector. Note, DC SQUID currently represented by single
    inductance sheet

    Add connectors to it using the `connection_pads` dictionary. See BaseQubit for more
    information.

    Sketch:
        Below is a sketch of the qubit
        ::

                                        claw_length
            Claw:       _________                    Gap:
                        |   ________________             _________    ____________
                  ______|  |                             _________|  |____________
                        |  |________________
                        |_________


    .. image::
        transmon_cross.png

    .. meta::
        Transmon Cross

    BaseQubit Default Options:
        * connection_pads: Empty Dict -- The dictionary which contains all active connection lines for the qubit.
        * _default_connection_pads: empty Dict -- The default values for the (if any) connection lines of the qubit.

    Default Options:
        * cross_width: '20um' -- Width of the CPW center trace making up the Crossmon
        * cross_length: '200um' -- Length of one Crossmon arm (from center)
        * cross_gap: '20um' -- Width of the CPW gap making up the Crossmon
        * _default_connection_pads: Dict
            * connector_type: '0' -- 0 = Claw type, 1 = gap type
            * claw_length: '30um' -- Length of the claw 'arms', measured from the connector center trace
            * ground_spacing: '5um' -- Amount of ground plane between the connector and Crossmon arm (minimum should be based on fabrication capabilities)
            * claw_width: '10um' -- The width of the CPW center trace making up the claw/gap connector
            * claw_gap: '6um' -- The gap of the CPW center trace making up the claw/gap connector
            * connector_location: '0' -- 0 => 'west' arm, 90 => 'north' arm, 180 => 'east' arm
       * style: 'default' -- Change the Josephson Junction style.
                             NOTE TO USER: Non-default choices shouldn't be rendered for simulation. They are for GDS rendering.
                             Choose from the following:
                             * 'default' -- Used for simulation in Ansys
                             * 'SQUID_LOOP_Dolan' -- Makes a SQUID loop in Dolan style. Use 'SQUID_LOOP' for options.
                             * 'Dolan_JJ' -- Makes a Josephson Junction in Dolan style. Use 'Dolan_JJ' for options.
       * SQUID_LOOP: Dict
           * SQUID_width: '40um' -- Interior width (x-axis) of the SQUID LOOP
           * SQUID_length: '5um' -- Interior length (y-axis) of the SQUID LOOP
           * SQUID_offset: '5um' -- Distance between the edge of the bottom pad and the grounding plane across of the cross
           * JJ_width: '150nm' -- Josephson Junction width. Governs Lj.
           * JJ_flip: False -- Switch the direction of the Josephson Junctions.
       * Dolan_JJ: Dict
           * JJ_width: '150nm' -- Josephson Junction width. Governs Lj.
           * JJ_flip: False -- Switch the direction of the Josephson Junctions.

    """

    default_options = Dict(
        cross_width="20um",
        cross_length="200um",
        cross_gap="20um",
        chip="main",
        _default_connection_pads=Dict(
            connector_type="0",  # 0 = Claw type, 1 = gap type
            claw_length="30um",
            ground_spacing="5um",
            claw_width="10um",
            claw_gap="6um",
            claw_cpw_length="40um",
            claw_cpw_width="10um",
            connector_location="0",  # 0 => 'west' arm, 90 => 'north' arm, 180 => 'east' arm
        ),
        SQUID_LOOP=Dict(SQUID_width="40um", SQUID_length="5um", SQUID_offset="5um", JJ_width="150nm", JJ_flip=False),
        Dolan_JJ=Dict(JJ_width="150nm", JJ_flip=False),
        style="default",
        layer="5",
    )
    """Default options."""

    component_metadata = Dict(short_name="Cross", _qgeometry_table_poly="True", _qgeometry_table_junction="True")
    """Component metadata"""

    TOOLTIP = """Simple Metal Transmon Cross used at LFL"""

    ##############################################MAKE######################################################

    def make(self):
        """This is executed by the GUI/user to generate the qgeometry for the
        component."""
        self.make_pocket()
        self.make_connection_pads()

    ###################################TRANSMON#############################################################

    def make_pocket(self):
        """Makes a basic Crossmon, 4 arm cross."""

        # self.p allows us to directly access parsed values (string -> numbers) form the user option
        p = self.p

        cross_width = p.cross_width
        cross_length = p.cross_length
        cross_gap = p.cross_gap

        # access to chip name
        chip = p.chip

        # Creates the cross and the etch equivalent.
        cross_line = draw.shapely.ops.unary_union(
            [
                draw.LineString([(0, cross_length), (0, -cross_length)]),
                draw.LineString([(cross_length, 0), (-cross_length, 0)]),
            ]
        )

        cross = cross_line.buffer(cross_width / 2, cap_style=2)
        cross_etch = cross.buffer(cross_gap, cap_style=3, join_style=2)

        ### Choose JJ / SQUID Style ###

        # If we're working with a default style
        if p.style == "default":
            jj = draw.LineString([(0, -cross_length), (0, -cross_length - cross_gap)])

            rect_jj = draw.rotate(jj, p.orientation, origin=(0, 0))
            rect_jj = draw.translate(jj, p.pos_x, p.pos_y)

            self.add_qgeometry("junction", dict(rect_jj=rect_jj), width=cross_width, chip=chip)

        elif (p.style == "SQUID_LOOP_Dolan") or (p.style == "JJ_Dolan"):
            # LFL Standard Parameters
            pad_width = 0.010  # mm
            SQUID_thickness = 0.002  # mm
            jj1_pad04 = 0.0005  # Pad should extend 5um past base

            # Cut out pin structures from qubit
            T_pin = draw.Polygon(
                [
                    (0, 0),
                    (0.001, 0),
                    (0.001, 0.002),
                    (0.003, 0.002),
                    (0.003, 0.004),
                    (-0.003, 0.004),
                    (-0.003, 0.002),
                    (-0.001, 0.002),
                    (-0.001, 0),
                ]
            )
            T_pin = draw.translate(T_pin, 0, -cross_length)
            cross = draw.subtract(cross, T_pin)

            ## If we're working w/ SQUID_LOOP_Dolan style
            if p.style == "SQUID_LOOP_Dolan":
                # SQUID Parameters
                sl = p.SQUID_LOOP

                # Make the other T pin, goes on the outside
                T_pin2 = draw.rotate(T_pin, -90, origin=(0, -cross_length))
                T_pin2 = draw.translate(
                    T_pin2, cross_gap + cross_width / 2, -cross_gap + pad_width / 2 + sl.SQUID_offset
                )

                # Rotate and translate the JJ
                jjy = (
                    -p.cross_length
                    - p.cross_gap
                    + sl.SQUID_length / 2
                    + SQUID_thickness / 2
                    + pad_width / 2
                    + sl.SQUID_offset
                )

                jjxprime = -np.sin(np.radians(p.orientation)) * jjy
                jjyprime = np.cos(np.radians(p.orientation)) * jjy

                # Make the JJ auto generate to the TransmonCross
                jj_options = dict(
                    pos_x=p.pos_x + jjxprime,
                    pos_y=p.pos_y + jjyprime,
                    orientation=180 + p.orientation,
                    SQUID_length=sl.SQUID_length,
                    SQUID_width=sl.SQUID_width,
                    stem1_length=cross_gap
                    - sl.SQUID_offset
                    - pad_width / 2
                    - 1.5 * SQUID_thickness
                    - sl.SQUID_length
                    - jj1_pad04,
                    stem2_length=cross_width / 2 + cross_gap - jj1_pad04 - SQUID_thickness - sl.SQUID_width / 2,
                    JJ_flip=sl.JJ_flip,
                    JJ_width=sl.JJ_width,
                )

                SQUID_LOOP_Dolan(self.design, self.name + "_SQUID_LOOP_Dolan", options=jj_options)

                ### Add to QGeometry
                polys = T_pin2
                polys = draw.rotate(polys, p.orientation, origin=(0, 0))
                polys = draw.translate(polys, p.pos_x, p.pos_y)
                T_pin2 = polys

                self.add_qgeometry("poly", dict(T_pin2=T_pin2), subtract=True, chip=chip)

            ## If we're working w/ a JJ_Dolan Style
            else:
                # JJ Parameters
                pj = p.Dolan_JJ

                # Make the 2nd T shaped pin
                T_pin2 = draw.rotate(T_pin, 180, origin=(0, -(cross_length + cross_gap / 2)))

                # Rotate and translate the JJ
                jjy = -cross_gap / 2 - cross_length

                jjxprime = -np.sin(np.radians(p.orientation)) * jjy
                jjyprime = np.cos(np.radians(p.orientation)) * jjy

                # Make the JJ auto generate to the TransmonCross
                jj_options = Dict(
                    pos_x=p.pos_x + jjxprime,
                    pos_y=p.pos_y + jjyprime,
                    orientation=p.orientation,
                    bridge_length=cross_gap - 2 * jj1_pad04,
                    JJ_width=pj.JJ_width,
                )

                if pj.JJ_flip:
                    jj_options.orientation = p.orientation + 180

                JjDolan(self.design, self.name + "_JJ_Dolan", options=jj_options)

                ### Add QGeometries
                polys = T_pin2
                polys = draw.rotate(polys, p.orientation, origin=(0, 0))
                polys = draw.translate(polys, p.pos_x, p.pos_y)
                T_pin2 = polys

                self.add_qgeometry("poly", dict(T_pin2=T_pin2), subtract=True, chip=chip)

        # Handling for future styles??!
        else:
            raise ValueError(
                "You entered an invalid 'options.style'. Choose from 'default', 'SQUID_LOOP_Dolan', or 'JJ_Dolan'"
            )

        # rotate and translate
        polys = [cross, cross_etch]
        polys = draw.rotate(polys, p.orientation, origin=(0, 0))
        polys = draw.translate(polys, p.pos_x, p.pos_y)

        [cross, cross_etch] = polys

        # generate qgeometry
        self.add_qgeometry("poly", dict(cross=cross), chip=chip)
        self.add_qgeometry("poly", dict(cross_etch=cross_etch), subtract=True, chip=chip)

    ############################CONNECTORS##################################################################################################

    def make_connection_pads(self):
        """Goes through connector pads and makes each one."""
        for name in self.options.connection_pads:
            self.make_connection_pad(name)

    def make_connection_pad(self, name: str):
        """Makes individual connector pad.

        Args:
            name (str) : Name of the connector pad
        """

        # self.p allows us to directly access parsed values (string -> numbers) form the user option
        p = self.p
        cross_width = p.cross_width
        cross_length = p.cross_length
        cross_gap = p.cross_gap

        # access to chip name
        chip = p.chip

        pc = self.p.connection_pads[name]  # parser on connector options
        c_g = pc.claw_gap
        c_l = pc.claw_length
        c_w = pc.claw_width
        c_c_w = pc.claw_cpw_width
        c_c_l = pc.claw_cpw_length
        g_s = pc.ground_spacing
        con_loc = pc.connector_location

        claw_cpw = draw.box(-c_w, -c_c_w / 2, -c_c_l - c_w, c_c_w / 2)

        if pc.connector_type == 0:  # Claw connector
            t_claw_height = 2 * c_g + 2 * c_w + 2 * g_s + 2 * cross_gap + cross_width  # temp value

            claw_base = draw.box(-c_w, -(t_claw_height) / 2, c_l, t_claw_height / 2)
            claw_subtract = draw.box(0, -t_claw_height / 2 + c_w, c_l, t_claw_height / 2 - c_w)
            claw_base = claw_base.difference(claw_subtract)

            connector_arm = draw.shapely.ops.unary_union([claw_base, claw_cpw])
            connector_etcher = draw.buffer(connector_arm, c_g)
        else:
            connector_arm = draw.box(0, -c_w / 2, -4 * c_w, c_w / 2)
            connector_etcher = draw.buffer(connector_arm, c_g)

        # Making the pin for  tracking (for easy connect functions).
        # Done here so as to have the same translations and rotations as the connector. Could
        # extract from the connector later, but since allowing different connector types,
        # this seems more straightforward.
        port_line = draw.LineString([(-c_c_l - c_w, -c_c_w / 2), (-c_c_l - c_w, c_c_w / 2)])

        claw_rotate = 0
        if con_loc > 135:
            claw_rotate = 180
        elif con_loc > 45:
            claw_rotate = -90

        # Rotates and translates the connector polygons (and temporary port_line)
        polys = [connector_arm, connector_etcher, port_line]
        polys = draw.translate(polys, -(cross_length + cross_gap + g_s + c_g), 0)
        polys = draw.rotate(polys, claw_rotate, origin=(0, 0))
        polys = draw.rotate(polys, p.orientation, origin=(0, 0))
        polys = draw.translate(polys, p.pos_x, p.pos_y)
        [connector_arm, connector_etcher, port_line] = polys

        # Generates qgeometry for the connector pads
        self.add_qgeometry("poly", {f"{name}_connector_arm": connector_arm}, layer=p.layer, chip=chip)
        self.add_qgeometry("poly", {f"{name}_connector_etcher": connector_etcher}, subtract=True, chip=chip)

        self.add_pin(name, port_line.coords, c_c_w)


class TransmonCrossFL(TransmonCross):  # pylint: disable=invalid-name
    """The base `TransmonCrossFL` class.
    Inherits `TransmonCross` class
    Description:
        Simple Metal Transmon Cross object. Creates the X cross-shaped island,
        the "junction" on the south end, and up to 3 connectors on the remaining arms
        (claw or gap).
        'claw_width' and 'claw_gap' define the width/gap of the CPW line that
        makes up the connector. Note, DC SQUID currently represented by single
        inductance sheet
        Add connectors to it using the `connection_pads` dictonary. See BaseQubit for more
        information.
        Flux line is added by default to the 'south' arm where the DC SQUID is located,
        default is a symmetric T style
        Default Options:
        Convention: Values (unless noted) are strings with units included, (e.g., '30um')
        * make_fl -         (Boolean) If True, adds a flux line
        * fl_style -        (String) Choose a style to construct the flux line
            * "none" -          Qiskit's default.
                * t_top -                 length of the flux line for mutual inductance to the SQUID
                * t_inductive_gap -       amount of metallization between the flux line and SQUID
                * t_offset -              degree by which the tail of the T is offset from the center
                * t_width -               width of the flux line's transmission line center trace
                * t_gap -                 dielectric gap of the flux line's transmission line
            * "tapered" -       Tapered / trapazoidal flux line.
                * t_length                length of the flux line
                * t_width_i               width of side further away from the qubit
                * t_width_f               width of side closer to the qubit
                * t_gap_i                 dielectric gap of flux line's tranmission line, further away from qubit
                * t_gap_f                 dielectric gap of flux line's tranmission line, closer to qubit
                * t_punch_through         length of how much the flux line punches through ground plane
                * t_hanger_para_length    length of hanger which moves along the x-axis
                * t_hanger_para_width     width of  hanger which moves along the x-axis
                * t_hanger_para_gap       dielectric gap of the x-axis hanger
                * t_hanger_perp_length    length of hanger which moves along the y-axis
                * t_hanger_perp_width     width of hanger which moves along the y-axis
                * t_hanger_perp_gap       dielectric gap of the y-axis hanger

    .. image::
        transmon_cross_fl.png
    .. meta::
        Transmon Cross Flux Line
    """

    component_metadata = Dict(short_name="Q", _qgeometry_table_poly="True", _qgeometry_table_path="True")
    """Component metadata"""

    default_options = Dict(
        make_fl=True,
        fl_style="default",
        fl_options=Dict(t_top="15um", t_offset="0um", t_inductive_gap="3um", t_width="5um", t_gap="3um"),
        LFL_options=Dict(
            t_width_i="10um",
            t_width_f="4.25um",
            t_length="90um",
            t_gap_i="6um",
            t_gap_f="2.5um",
            t_punch_through="5um",
            t_hanger_para_length="40um",
            t_hanger_para_width="2.5um",
            t_hanger_para_gap="2.5um",
            t_hanger_perp_length="55um",
            t_hanger_perp_width="2.7um",
            t_hanger_perp_gap="3.2um",
        ),
    )
    """Default drawing options"""

    TOOLTIP = """The base `TransmonCrossFL` class."""

    def make(self):
        """Define the way the options are turned into QGeometry."""
        super().make()

        if self.options.make_fl:
            if (self.options.fl_style == "default") or (self.options.fl_style is None):
                self.make_flux_line()
            elif self.options.fl_style == "tapered":
                self.make_flux_line_tapered()
            else:
                pass

    #####################################################################

    def make_flux_line(self):
        """Creates the charge line if the user has charge line option to
        TRUE.

        This is Qiskit Metal's default.
        """

        # Grab option values
        pf = self.p.fl_options
        p = self.p
        # Make the T flux line
        h_line = draw.LineString([(-pf.t_top / 2, 0), (pf.t_top / 2, 0)])
        v_line = draw.LineString([(pf.t_offset, 0), (pf.t_offset, -0.03)])

        parts = [h_line, v_line]

        # Move the flux line down to the SQUID
        parts = draw.translate(
            parts, 0, -(p.cross_length + p.cross_gap + pf.t_inductive_gap + pf.t_width / 2 + pf.t_gap)
        )

        # Rotate and translate based on crossmon location
        parts = draw.rotate(parts, p.orientation, origin=(0, 0))
        parts = draw.translate(parts, p.pos_x, p.pos_y)

        [h_line, v_line] = parts

        # Adding to qgeometry table
        self.add_qgeometry("path", {"h_line": h_line, "v_line": v_line}, width=pf.t_width, layer=p.layer)

        self.add_qgeometry(
            "path",
            {"h_line_sub": h_line, "v_line_sub": v_line},
            width=pf.t_width + 2 * pf.t_gap,
            subtract=True,
            layer=p.layer,
        )

        # Generating pin
        pin_line = v_line.coords
        self.add_pin("flux_line", points=pin_line, width=pf.t_width, gap=pf.t_gap, input_as_norm=True)

    #####################################################################

    def make_flux_line_tapered(self):
        """
        Creates the charge line which is taper w/ a hanging bar.

        Activates when all of these conditions are met:
        - self.options.make_fl == True
        - self.options.fl_options.lfl_style == "tapered"
        """
        # Grab option values
        pf = self.p.LFL_options
        p = self.p

        ##### Fast Flux Line Element #####
        # Make tappered FF Line
        tapper = draw.Polygon(
            [
                (pf.t_width_i / 2, 0),
                (-pf.t_width_i / 2, 0),
                (-pf.t_width_f / 2, pf.t_length),
                (-pf.t_width_f / 2 + pf.t_hanger_para_length, pf.t_length),
                (-pf.t_width_f / 2 + pf.t_hanger_para_length, pf.t_length - pf.t_hanger_perp_length),
                (
                    -pf.t_width_f / 2 + pf.t_hanger_para_length - pf.t_hanger_perp_width,
                    pf.t_length - pf.t_hanger_perp_length,
                ),
                (
                    -pf.t_width_f / 2 + pf.t_hanger_para_length - pf.t_hanger_perp_width,
                    pf.t_length - pf.t_hanger_para_width,
                ),
                (
                    (pf.t_length - pf.t_hanger_para_width) * (pf.t_width_f - pf.t_width_i) / (2 * pf.t_length)
                    + pf.t_width_i / 2,
                    pf.t_length - pf.t_hanger_para_width,
                ),
            ]
        )

        # Make subtraction
        sub_tapper = draw.Polygon(
            [
                (pf.t_width_i / 2 + pf.t_gap_i, 0),
                (-pf.t_width_i / 2 - pf.t_gap_i, 0),
                (-pf.t_width_f / 2 - pf.t_gap_f, pf.t_length + pf.t_hanger_para_gap),
                (
                    -pf.t_width_f / 2 + pf.t_hanger_para_length + pf.t_hanger_perp_gap,
                    pf.t_length + pf.t_hanger_para_gap,
                ),
                (
                    -pf.t_width_f / 2 + pf.t_hanger_para_length + pf.t_hanger_perp_gap,
                    pf.t_length - pf.t_hanger_perp_length,
                ),
                (
                    -pf.t_width_f / 2 + pf.t_hanger_para_length - pf.t_hanger_perp_width - pf.t_hanger_perp_gap,
                    pf.t_length - pf.t_hanger_perp_length,
                ),
                (
                    -pf.t_width_f / 2 + pf.t_hanger_para_length - pf.t_hanger_perp_width - pf.t_hanger_perp_gap,
                    pf.t_length - pf.t_hanger_para_width - pf.t_hanger_para_gap,
                ),
                (
                    (pf.t_length - pf.t_hanger_para_width - pf.t_hanger_para_gap)
                    / (
                        (pf.t_length - pf.t_hanger_para_gap)
                        / ((pf.t_width_f - pf.t_width_i) / 2 + pf.t_gap_f - pf.t_gap_i)
                    )
                    + pf.t_width_i / 2
                    + pf.t_gap_i,
                    pf.t_length - pf.t_hanger_para_width - pf.t_hanger_para_gap,
                ),
            ]
        )

        # Translate all the parts to the edge of the
        parts = [tapper, sub_tapper]

        parts = draw.translate(parts, 0, -(p.cross_length + p.cross_gap + pf.t_length - pf.t_punch_through))
        parts = draw.rotate(parts, p.orientation, origin=(0, 0))
        parts = draw.translate(parts, p.pos_x, p.pos_y)

        tapper, sub_tapper = parts

        ##### Adding to qgeometry table #####
        self.add_qgeometry("poly", dict(tapper=tapper), layer=p.layer)
        self.add_qgeometry("poly", dict(sub_tapper=sub_tapper), subtract=True, layer=p.layer)
