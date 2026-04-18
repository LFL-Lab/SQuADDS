import warnings
from collections import OrderedDict

from qiskit_metal import Dict
from qiskit_metal.qlibrary.core import QComponent
from qiskit_metal.qlibrary.qubits.transmon_cross import TransmonCross


class QubitCavity(QComponent):
    """
    QubitCavity class represents a coupled qubit-cavity system.
    It contains methods to create the qubit, cavity, coupler, and CPWs.
    """

    default_options = Dict(
        chip="main",
        cavity_claw_options=Dict(
            coupler_type="CLT",
            coupler_options=Dict(orientation="180", coupling_length="200um"),
            cpw_opts=Dict(
                total_length="4000um",
                left_options=Dict(
                    lead=Dict(start_straight=0, end_straight=0, start_jogged_extension=0, end_jogged_extension=0),
                    fillet="49.9um",
                ),
                right_options=Dict(
                    meander=Dict(spacing="100um", asymmetry="0um"),
                    fillet="49.9um",
                ),
            ),
        ),
        qubit_options=Dict(connection_pads=Dict(), claw_cpw_length="50um", orientation="180", pos_y="1500um"),
    )

    component_metadata = Dict(short_name="qubitcavity")
    """Component metadata"""

    """Default options"""

    def copier(self, d, u):
        for k, v in u.items():
            if (
                not isinstance(v, str)
                and not isinstance(v, float)
                and not isinstance(v, bool)
                and not isinstance(v, int)
                and v is not None
                and not isinstance(v, list)
            ):
                d[k] = self.copier(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    def make(self):
        self.make_qubit()
        self.make_cavity()
        self.make_pins()
        warnings.warn(
            "QubitCavity is a convenience wrapper for quick visualization only. "
            "For production designs, build components individually (TransmonCross, "
            "CoupledLineTee, RouteMeander) to control trace widths, fillets, and "
            "lead straights. See SQuADDS Tutorial 5 or the MCP layout-guide resource. "
            "If you see CPW kinks, increase 'lead.start_straight' / 'lead.end_straight' "
            "on the RouteMeander (try 75um / 50um).",
            UserWarning,
            stacklevel=2,
        )

    def make_qubit(self):
        """
        Creates a qubit based on the specified qubit options.

        Returns:
            None
        """
        p = self.p

        qubit_opts = Dict()
        self.copier(qubit_opts, p.qubit_options)
        qubit_opts["pos_y"] = 0
        try:
            qubit_opts["pos_x"] = "-1500um" if p.cavity_claw_options["cpw_opts"].total_length > 2.500 else "-1000um"
        except:
            qubit_opts["pos_x"] = "-1500um" if p.cavity_claw_options["cpw_options"].total_length > 2.500 else "-1000um"
        self.qubit = TransmonCross(self.design, f"{self.name}_xmon", options=qubit_opts)

    def make_cavity(self):
        """
        This method is used to create a cavity in the coupled system.
        It calls the make_coupler() and make_cpws() methods to create the necessary components.
        """
        self.make_coupler()
        self.make_cpws()

    def make_coupler(self):
        """
        Creates a coupler based on the specified coupling type in the cavity options.

        Returns:
            None
        """
        p = self.p

        temp_opts = Dict()
        try:
            self.copier(temp_opts, p.cavity_claw_options["coupler_options"])
        except:
            self.copier(temp_opts, p.cavity_claw_options["cplr_opts"])

        if p.cavity_claw_options["coupler_type"].upper() == "CLT":
            from qiskit_metal.qlibrary.couplers.coupled_line_tee import CoupledLineTee

            self.coupler = CoupledLineTee(self.design, f"{self.name}_CLT_coupler", options=temp_opts)
        elif (
            p.cavity_claw_options["coupler_type"].lower() == "capn"
            or p.cavity_claw_options["coupler_type"].lower() == "ncap"
            or p.cavity_claw_options["coupler_type"].lower() == "capninterdigitaltee"
        ):
            from qiskit_metal.qlibrary.couplers.cap_n_interdigital_tee import CapNInterdigitalTee

            self.coupler = CapNInterdigitalTee(self.design, f"{self.name}_capn_coupler", options=temp_opts)

    def make_cpws(self):
        """
        Creates the CPWs (Coplanar Waveguides) for the coupled systems.

        Returns:
            None
        """
        from qiskit_metal.qlibrary.tlines.meandered import RouteMeander

        p = self.p
        try:
            p.cpw_opts = p.cavity_claw_options["cpw_opts"]
        except:
            p.cpw_opts = p.cavity_claw_options["cpw_options"]

        left_opts = Dict()
        left_opts.update(
            {
                "total_length": (
                    p.cpw_opts.total_length
                    if p.cavity_claw_options["coupler_type"] == "capacitive"
                    else p.cpw_opts.total_length / 2
                )
            }
        )
        left_opts.update({"pin_inputs": Dict(start_pin=Dict(component="", pin=""), end_pin=Dict(component="", pin=""))})
        self.copier(left_opts, p.cpw_opts.left_options)

        if p.cavity_claw_options["coupler_type"].lower() == "clt" and self.coupler.options["coupling_length"] > 0.150:
            adj_distance = self.coupler.options["coupling_length"]
        else:
            adj_distance = 0
        jogs = OrderedDict()
        jogs[0] = ["R90", f"{adj_distance / (1.5)}um"]

        # Compute a safe fillet from the meander spacing to avoid self-intersecting curves.
        meander_spacing = 0.100  # default 100um in mm
        if hasattr(p, "cpw_opts") and hasattr(p.cpw_opts, "left_options"):
            fillet_val = p.cpw_opts.left_options.get("fillet", 0.0499)
            if isinstance(fillet_val, str):
                fillet_val = float(fillet_val.replace("um", "")) / 1000
            min(fillet_val, meander_spacing / 2.1)
        else:
            min(0.0499, meander_spacing / 2.1)

        left_opts.update({"lead": Dict(start_straight="75um", end_straight="50um", start_jogged_extension=jogs)})
        left_opts.update(
            {
                "pin_inputs": Dict(
                    start_pin=Dict(component=self.coupler.name, pin="second_end"),
                    end_pin=Dict(component=self.qubit.name, pin=list(self.qubit.options["connection_pads"].keys())[0]),
                )
            }
        )
        left_opts.update(
            {
                "meander": Dict(
                    spacing="100um",
                    asymmetry=f"{adj_distance / (3)}um",  # need this to make CPW asymmetry half of the coupling length
                )
            }
        )  # if not, sharp kinks occur in CPW :(
        # cpw = RouteMeander(design, 'cpw', options = opts)

        left_opts["pin_inputs"]["start_pin"].update({"component": self.qubit.name})
        left_opts["pin_inputs"]["start_pin"].update({"pin": list(self.qubit.options["connection_pads"].keys())[0]})

        left_opts["pin_inputs"]["end_pin"].update({"component": self.coupler.name})
        left_opts["pin_inputs"]["end_pin"].update({"pin": "second_end"})

        self.LeftMeander = RouteMeander(self.design, f"{self.name}_left_cpw", options=left_opts)

        # --- Trace width matching ---
        # Ensure the RouteMeander trace_width matches the qubit claw's CPW width
        # to avoid impedance discontinuities at the connection point.
        qubit_conn_name = list(self.qubit.options["connection_pads"].keys())[0]
        qubit_conn_opts = self.qubit.options["connection_pads"][qubit_conn_name]
        claw_cpw_w = qubit_conn_opts.get("claw_cpw_width", None)
        meander_trace_w = self.LeftMeander.options.get("trace_width", None)
        if claw_cpw_w is not None and meander_trace_w is not None:
            if str(claw_cpw_w) != str(meander_trace_w):
                warnings.warn(
                    f"Trace width mismatch: qubit claw_cpw_width={claw_cpw_w} "
                    f"but RouteMeander trace_width={meander_trace_w}. "
                    f"Setting RouteMeander trace_width to {claw_cpw_w} to match. "
                    f"Override by explicitly setting trace_width in cpw_opts.",
                    UserWarning,
                    stacklevel=2,
                )
                self.LeftMeander.options["trace_width"] = str(claw_cpw_w)

        if p.cavity_claw_options["coupler_type"] == "inductive":
            right_opts = Dict()
            right_opts.update({"total_length": p.cpw_opts.total_length / 2})
            right_opts.update(
                {"pin_inputs": Dict(start_pin=Dict(component="", pin=""), end_pin=Dict(component="", pin=""))}
            )
            right_opts["pin_inputs"]["end_pin"].update({"component": p.cpw_opts.pin_inputs.end_pin.component})
            right_opts["pin_inputs"]["end_pin"].update({"pin": p.cpw_opts.pin_inputs.end_pin.pin})

            right_opts["pin_inputs"]["start_pin"].update({"component": self.coupler.name})
            right_opts["pin_inputs"]["start_pin"].update({"pin": "second_start"})

            self.copier(right_opts, p.cpw_opts.right_options)

            self.RightMeander = RouteMeander(self.design, f"{self.name}_right_cpw", options=right_opts)
            self.add_qgeometry("path", self.RightMeander.qgeometry_dict("path"), chip=p.chip)

    def make_pins(self):
        """
        Adds pins to the coupled system.

        Retrieves pin information from the coupler and adds the pins to the system.

        Args:
            None

        Returns:
            None
        """
        start_dict = self.coupler.get_pin("prime_start")
        end_dict = self.coupler.get_pin("prime_end")
        self.add_pin("prime_start", start_dict["points"], start_dict["width"])
        self.add_pin("prime_end", end_dict["points"], end_dict["width"])

    def make_wirebond_pads(self):
        from qiskit_metal.qlibrary.terminations.launchpad_wb import LaunchpadWirebond
        from qiskit_metal.qlibrary.tlines.straight_path import RouteStraight

        p = self.p
        options = Dict(
            orientation=-90,
            pos_y=(float)(self.coupler.options.pos_y[:-2]) + 2.75,
            trace_width=p.cavity_claw_options.cpw_opts.left_options.trace_width,
            trace_gap=p.cavity_claw_options.cpw_opts.left_options.trace_gap,
        )
        LaunchpadWirebond(self.design, "wb_top", options=options)

        options = Dict(
            orientation=90,
            pos_y=(float)(self.coupler.options.pos_y[:-2]) - 2.75,
            trace_width=p.cavity_claw_options.cpw_opts.left_options.trace_width,
            trace_gap=p.cavity_claw_options.cpw_opts.left_options.trace_gap,
        )
        LaunchpadWirebond(self.design, "wb_bottom", options=options)

        feedline_opts = Dict(
            pin_inputs=Dict(
                start_pin=Dict(component="wb_top", pin="tie"), end_pin=Dict(component="wb_bottom", pin="tie")
            ),
            trace_width=p.cavity_claw_options.cpw_opts.left_options.trace_width,
            trace_gap=p.cavity_claw_options.cpw_opts.left_options.trace_gap,
        )
        RouteStraight(self.design, "feedline", options=feedline_opts)

    def show(self, gui, include_wirebond_pads=False, **kwargs):
        if include_wirebond_pads:
            self.make_wirebond_pads()
        else:
            if "feedline" in self.design.components:
                self.design.delete_component("wb_top")
                self.design.delete_component("wb_bottom")
                self.design.delete_component("feedline")

        gui.rebuild()
        gui.autoscale()
        figure_name = kwargs.get("figure_name", "QubitCavity.png")
        gui.screenshot(figure_name)

    def to_gds(self, filename, include_wirebond_pads=False):
        if include_wirebond_pads:
            self.make_wirebond_pads()
        else:
            if "feedline" in self.design.components:
                self.design.delete_component("wb_top")
                self.design.delete_component("wb_bottom")
                self.design.delete_component("feedline")

        a_gds = self.design.renderers.gds
        a_gds.options["cheese"]["view_in_file"]["main"][1] = False
        a_gds.options["no_cheese"]["view_in_file"]["main"][1] = False

        a_gds.export_to_gds(f"{filename}.gds")
