from qiskit_metal import Dict, draw
from qiskit_metal.qlibrary.core.base import QComponent


class Airbridge(QComponent):
    """
    The base "Airbridge" inherits the "QComponent" class.

    NOTE TO USER: Please be aware that when designing with this
    qcomponent, one must take care in accounting for the junction
    qgeometry when exporting to to GDS and/or to a simulator. This
    qcomponent should not be rendered for EM simulation.

    Default Options:
        * crossover_length: '20um' -- Distance between the two outter squares.
                                      This should be the same length as (cpw_width + 2 * cpw_gap)
        * RR_layer:              3 -- GDS layer of inner squares
        * BR_layer:              4 -- GDS layer of bridge
    """

    # Default drawing options
    default_options = Dict(pos_x=0, pos_y=0, orientation=0, crossover_length="20um", RR_layer=30, BR_layer=31)
    """Default drawing options"""

    # Name prefix of component, if user doesn't provide name
    component_metadata = Dict(short_name="component")
    """Component metadata"""

    def make(self):
        """Convert self.options into QGeometry."""

        p = self.parse_options()
        l = p.crossover_length

        # These parameters define the dimensions of the airbridge components
        bridge_pad_gap = 0.0015  # 1.5um distance between edge of bridge and pad
        if l >= 0.005 and l <= 0.016:
            bridge_width = 0.005
            pad_length = 0.008
            pad_width = 0.008
        elif l > 0.016 and l <= 0.027:
            bridge_width = 0.0075
            pad_length = 0.010
            pad_width = 0.0105
        elif l > 0.027 and l <= 0.032:
            bridge_width = 0.010
            pad_length = 0.014
            pad_width = 0.013

        # Make the pad Structure
        left_inside = draw.rectangle(pad_length, pad_width, 0, 0)
        right_inside = draw.translate(left_inside, l / 2 + pad_length / 2 + bridge_pad_gap, 0)
        left_inside = draw.translate(left_inside, -(l / 2 + pad_length / 2 + bridge_pad_gap), 0)

        # Make the bridge Structure
        bridge = draw.rectangle(l, bridge_width, 0, 0)
        left_outside = draw.rectangle(pad_length + 2 * bridge_pad_gap, pad_width + 2 * bridge_pad_gap, 0, 0)
        right_outside = draw.translate(left_outside, l / 2 + pad_length / 2 + bridge_pad_gap, 0)
        left_outside = draw.translate(left_outside, -(l / 2 + pad_length / 2 + bridge_pad_gap), 0)

        bridge_struct = draw.union(bridge, left_outside, right_outside)

        ### Final adjustments to allow repositioning
        final_design = [bridge_struct, left_inside, right_inside]

        final_design = draw.rotate(final_design, p.orientation, origin=(0, 0))
        final_design = draw.translate(final_design, p.pos_x, p.pos_y)

        bridge_struct, left_inside, right_inside = final_design

        ### Add everything as a QGeometry
        self.add_qgeometry("poly", {"bridge_struct": bridge_struct}, layer=p.BR_layer, subtract=False)
        self.add_qgeometry("poly", {"left_inside": left_inside}, layer=p.RR_layer, subtract=False)
        self.add_qgeometry("poly", {"right_inside": right_inside}, layer=p.RR_layer, subtract=False)
