from collections import OrderedDict

import numpy as np
from qiskit_metal import Dict
from qiskit_metal.qlibrary.core import QComponent, QRoute, QRoutePoint
from qiskit_metal.qlibrary.qubits.transmon_cross import TransmonCross

from squadds.components.claw_coupler import TransmonClaw


class CavityClaw(QComponent):
    """
    QubitCavity class represents a coupled qubit-cavity system.
    It contains methods to create the qubit, cavity, coupler, and CPWs.
    """
    
    default_options = Dict(
        chip = 'main',
        cavity_claw_options = Dict(
            coupler_type = 'CLT',
            coupler_options = Dict(
                orientation = '180',
                coupling_length = '200um'
            ),
            cpw_opts = Dict(
                total_length = '4000um',
                left_options = Dict(
                    lead = Dict(
                        start_straight = 0,
                        end_straight = 0,
                        start_jogged_extension = 0,
                        end_jogged_extension = 0
                    ),
                    fillet = '49.9um',
                ),
                right_options = Dict(
                    meander=Dict(spacing='100um', asymmetry='0um'),
                    fillet = '49.9um',
                )
            )
        ),
        qubit_options = Dict(
            connection_pads=Dict(),
            claw_cpw_length = '50um',
            orientation = '180',
            pos_y = '1500um'
        )
    )
    
    component_metadata = Dict(short_name='cavityclaw')
    """Component metadata"""

    """Default options"""

    def copier(self, d, u):
        for k, v in u.items():
            if not isinstance(v, str) and not isinstance(v, float) and not isinstance(v, bool) and not isinstance(v, int) and v is not None and not isinstance(v, list):
                d[k] = self.copier(d.get(k, {}), v)
            else:
                d[k] = v
        return d
    
    def make(self):
        p = self.p
        self.make_qubit()
        self.make_cavity()
        self.make_pins()
        
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
        qubit_opts["pos_x"] = "-1500um" if p.cavity_claw_options['cpw_opts'].total_length > 2.500 else "-1000um"
        self.qubit = TransmonClaw(self.design, "{}_xmon".format(self.name), options = qubit_opts)

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
        self.copier(temp_opts, p.cavity_claw_options['coupler_options'])

        if(p.cavity_claw_options['coupler_type'].upper() == "CLT"):
            from qiskit_metal.qlibrary.couplers.coupled_line_tee import \
                CoupledLineTee
            self.coupler = CoupledLineTee(self.design, "{}_CLT_coupler".format(self.name), options=temp_opts)
        elif(p.cavity_claw_options['coupler_type'].lower() == 'capn' or p.cavity_claw_options['coupler_type'].lower() == 'ncap' or p.cavity_claw_options['coupler_type'].lower() == 'capninterdigitaltee'):
            from qiskit_metal.qlibrary.couplers.cap_n_interdigital_tee import \
                CapNInterdigitalTee
            self.coupler = CapNInterdigitalTee(self.design, '{}_capn_coupler'.format(self.name), options=temp_opts)

    def make_cpws(self):
        """
        Creates the CPWs (Coplanar Waveguides) for the coupled systems.

        Returns:
            None
        """
        from qiskit_metal.qlibrary.tlines.meandered import RouteMeander

        p = self.p
        p.cpw_opts = p.cavity_claw_options['cpw_opts']
        
        left_opts = Dict()
        left_opts.update({'total_length': (p.cpw_opts.total_length if p.cavity_claw_options['coupler_type'] == 'capacitive' else p.cpw_opts.total_length/2) })
        left_opts.update({'pin_inputs':Dict(
                                            start_pin = Dict(
                                                component = '',
                                                pin = ''
                                            ),
                                            end_pin = Dict(
                                                component = '',
                                                pin = ''
                                            )
                                            )})
        self.copier(left_opts, p.cpw_opts.left_options)

        if p.cavity_claw_options["coupler_type"].lower() == "clt" and self.coupler.options["coupling_length"] > 0.150:
            adj_distance = self.coupler.options["coupling_length"]
        else:
            adj_distance = 0
        jogs = OrderedDict()
        jogs[0] = ["R90", f'{adj_distance/(1.5)}um']
        left_opts.update({"lead" : Dict(
                                start_straight = "100um",
                                end_straight = "50um",
                                
                                start_jogged_extension = jogs
                                )})
        left_opts.update({"pin_inputs" : Dict(start_pin = Dict(component = self.coupler.name,
                                                        pin = 'second_end'),
                                    end_pin = Dict(component = self.qubit.name,
                                                    pin = list(self.qubit.options["connection_pads"].keys())[0]))})
        left_opts.update({"meander" : Dict(
                                    spacing = "100um",
                                    asymmetry = f'{adj_distance/(3)}um' # need this to make CPW asymmetry half of the coupling length
                                    )})                                 # if not, sharp kinks occur in CPW :(
        # cpw = RouteMeander(design, 'cpw', options = opts)

        left_opts['pin_inputs']['start_pin'].update({'component':self.qubit.name})
        left_opts['pin_inputs']['start_pin'].update({'pin':list(self.qubit.options["connection_pads"].keys())[0]})

        left_opts['pin_inputs']['end_pin'].update({'component':self.coupler.name})
        left_opts['pin_inputs']['end_pin'].update({'pin':'second_end'})

        self.LeftMeander = RouteMeander(self.design, "{}_left_cpw".format(self.name), options = left_opts)

        if(p.cavity_claw_options['coupler_type'] == 'inductive'):
            right_opts = Dict()
            right_opts.update({'total_length':p.cpw_opts.total_length/2})
            right_opts.update({'pin_inputs':Dict(
                                                start_pin = Dict(
                                                    component = '',
                                                    pin = ''
                                                ),
                                                end_pin = Dict(
                                                    component = '',
                                                    pin = ''
                                                )
                                                )})
            right_opts['pin_inputs']['end_pin'].update({'component':p.cpw_opts.pin_inputs.end_pin.component})
            right_opts['pin_inputs']['end_pin'].update({'pin':p.cpw_opts.pin_inputs.end_pin.pin})

            right_opts['pin_inputs']['start_pin'].update({'component':self.coupler.name})
            right_opts['pin_inputs']['start_pin'].update({'pin':'second_start'})

            self.copier(right_opts, p.cpw_opts.right_options)

            self.RightMeander = RouteMeander(self.design, "{}_right_cpw".format(self.name), options = right_opts)
            self.add_qgeometry('path', self.RightMeander.qgeometry_dict('path'), chip = p.chip)

        
    def make_pins(self):
        """
        Adds pins to the coupled system.

        Retrieves pin information from the coupler and adds the pins to the system.

        Args:
            None

        Returns:
            None
        """
        start_dict = self.coupler.get_pin('prime_start')
        end_dict = self.coupler.get_pin('prime_end')
        self.add_pin('prime_start', start_dict['points'], start_dict['width'])
        self.add_pin('prime_end', end_dict['points'], end_dict['width'])

    def make_wirebond_pads(self):
        from qiskit_metal.qlibrary.terminations.launchpad_wb import \
            LaunchpadWirebond
        from qiskit_metal.qlibrary.tlines.straight_path import RouteStraight
        p = self.p
        print(self.coupler.options)        
        options = Dict(
            orientation = -90,
            pos_y = (float)(self.coupler.options.pos_y[:-2]) + 0.75,
            trace_width = p.cavity_claw_options.cpw_opts.left_options.trace_width,
            trace_gap = p.cavity_claw_options.cpw_opts.left_options.trace_gap
        )
        wb1 = LaunchpadWirebond(self.design, 'wb_top', options=options)

        options = Dict(
            orientation = 90,
            pos_y = (float)(self.coupler.options.pos_y[:-2]) - 0.75,
            trace_width = p.cavity_claw_options.cpw_opts.left_options.trace_width,
            trace_gap = p.cavity_claw_options.cpw_opts.left_options.trace_gap
        )
        wb2 = LaunchpadWirebond(self.design, 'wb_bottom', options=options)

        feedline_opts = Dict(pin_inputs = Dict(start_pin = Dict(component = 'wb_top',
                                                                pin = 'tie'),
                                            end_pin = Dict(component = 'wb_bottom',
                                                            pin = 'tie')),
                            trace_width = p.cavity_claw_options.cpw_opts.left_options.trace_width,
                            trace_gap = p.cavity_claw_options.cpw_opts.left_options.trace_gap)
        feedline = RouteStraight(self.design, 'feedline', options = feedline_opts)

    def show(self, gui, include_wirebond_pads=False):
        if include_wirebond_pads:
            self.make_wirebond_pads()
        gui.rebuild()
        gui.autoscale()
        gui.screenshot()

    def to_gds(self, options, include_wirebond_pads=False):
        raise NotImplementedError("This method is not implemented in the base class.")