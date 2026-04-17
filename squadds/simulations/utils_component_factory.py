from qiskit_metal import Dict
from qiskit_metal.qlibrary.couplers.cap_n_interdigital_tee import CapNInterdigitalTee
from qiskit_metal.qlibrary.couplers.coupled_line_tee import CoupledLineTee
from qiskit_metal.qlibrary.tlines.meandered import RouteMeander

from squadds.components.claw_coupler import TransmonClaw
from squadds.components.coupled_systems import QubitCavity


def create_qubitcavity(opts, design):
    return QubitCavity(design, "qubitcavity", options=opts)


def create_claw(opts, cpw_length, design):
    opts["orientation"] = "-90"
    opts["pos_x"] = "-1500um" if cpw_length > 2500 else "-1000um"
    return TransmonClaw(design, "claw", options=opts)


def create_ncap_coupler(opts, design):
    opts["orientation"] = "-90"
    return CapNInterdigitalTee(design, "cplr", options=opts)


def create_clt_coupler(opts, design):
    opts["orientation"] = "-90"
    return CoupledLineTee(design, "cplr", options=opts)


def create_cpw(opts, cplr, design):
    if "finger_count" not in cplr.options:
        (
            int("".join(filter(str.isdigit, cplr.options["coupling_length"])))
            if int("".join(filter(str.isdigit, cplr.options["coupling_length"]))) > 150
            else 0
        )

    opts.update({"lead": Dict(start_straight="50um", end_straight="50um")})
    opts.update(
        {
            "pin_inputs": Dict(
                start_pin=Dict(component=cplr.name, pin="second_end"),
                end_pin=Dict(component="claw", pin="readout"),
            )
        }
    )
    opts.update({"meander": Dict(spacing="100um")})
    return RouteMeander(design, "cpw", options=opts)
