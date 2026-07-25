import numpy as np
from qiskit_metal import Dict, draw
from qiskit_metal.qlibrary.core import QComponent
from shapely import set_precision
from shapely.affinity import rotate, translate
from shapely.errors import GEOSException
from shapely.geometry import LineString, Polygon


class GeneralizedCapNInterdigital(QComponent):
    @staticmethod
    def _clean_geometry(geom, precision=1e-6):
        """Repair potentially invalid polygons before/after precision snapping."""
        if geom is None:
            return geom

        # First pass: resolve obvious self-intersections.
        geom = geom.buffer(0)
        try:
            geom = set_precision(geom, precision)
        except GEOSException:
            # Fallback path for borderline-topology cases that fail on snap.
            geom = geom.buffer(0)
            geom = set_precision(geom, precision)
        return geom.buffer(0)

    default_options = Dict(
        north_cpw_length="20um",
        north_cpw_width="10um",
        north_cpw_gap="6um",
        north_cpw_radius="0um",
        north_cpw_etch_radius="0um",
        north_cpw_xpos_offset="0um",
        north_cpw_maintain_gap=True,
        #
        south_cpw_length="20um",
        south_cpw_width="10um",
        south_cpw_gap="6um",
        south_cpw_radius="0um",
        south_cpw_etch_radius="0um",
        south_cpw_xpos_offset="0um",
        south_cpw_maintain_gap=True,
        #
        north_spine_width="10um",
        north_west_spine_radius="0um",
        north_east_spine_radius="0um",
        north_west_spine_etch_radius="0um",
        north_east_spine_etch_radius="0um",
        #
        south_spine_width="10um",
        south_west_spine_radius="0um",
        south_east_spine_radius="0um",
        south_west_spine_etch_radius="0um",
        south_east_spine_etch_radius="0um",
        #
        north_gap_ground="6um",
        south_gap_ground="6um",
        east_gap_ground="6um",
        west_gap_ground="6um",
        #
        finger_length="20um",
        finger_width="10um",
        finger_gap_north_south="6um",
        finger_gap_east_west="6um",
        finger_radius="0um",
        finger_etch_radius="0um",
        finger_edge_gap_radius="0um",
        finger_count="5",
    )

    def make(self):
        # self._install_hfss_scalar_safe_polyline_patch()
        p = self.p

        # north body is rotated 180 degrees, so for building it, the x-directions need to be flipped
        p.north_cpw_xpos_offset = -p.north_cpw_xpos_offset
        self._initialize_calculated_variables()

        num_fingers = int(p.finger_count)
        is_finger_count_even = num_fingers % 2 == 0

        if is_finger_count_even:
            finger_count_south = finger_count_north = num_fingers // 2
        else:
            finger_count_south = num_fingers // 2 + 1
            finger_count_north = num_fingers // 2

        assert p.finger_length >= 0.0, "finger_length must be greater than or equal to 0"

        if p.finger_length > 0.0:
            assert p.finger_count > 0, "finger_count must be greater than 0 if finger_length is greater than 0"

            assert p.finger_radius <= p.finger_width / 2, (
                f"finger_radius must be <= half the finger_width (= {p.finger_width / 2 * 1000} um)"
            )
            assert (p.finger_length - (p.finger_radius + p.finger_etch_radius)) >= -1e-6, (
                "Must have finger_length >= finger_radius + finger_etch_radius in order to render"
            )

        assert (p.north_cpw_length - p.north_cpw_radius) >= 0.0, (
            "north_cpw_length must be larger than or equal to north_cpw_radius in order to render"
        )
        assert (p.south_cpw_length - p.south_cpw_radius) >= 0.0, (
            "south_cpw_length must be larger than or equal to south_cpw_radius in order to render"
        )

        y_translation_offset = -self.cap_height - p.north_cpw_length

        cap_south_body = self._make_cap_body(is_body_south=True, finger_count=finger_count_south)
        cap_south_body = translate(cap_south_body, 0, y_translation_offset)
        cap_south_body = self._clean_geometry(cap_south_body)

        cap_north_body = self._make_cap_body(
            is_body_south=False, finger_count=finger_count_north, gap_first=False if is_finger_count_even else True
        )
        cap_north_body = rotate(cap_north_body, 180, origin=(0, 0))
        cap_north_body = translate(cap_north_body, 0, self.cap_height + y_translation_offset)
        cap_north_body = self._clean_geometry(cap_north_body)

        # switch it back for etch
        p.north_cpw_xpos_offset = -p.north_cpw_xpos_offset
        self._initialize_calculated_variables()

        cap_etch = self._make_cap_etch()
        cap_etch = translate(
            cap_etch,
            self.etch_width / 2 - p.west_gap_ground - self.cap_width / 2,
            -p.south_gap_ground + y_translation_offset,
        )
        cap_etch = self._clean_geometry(cap_etch)

        c_items = [cap_north_body, cap_south_body, cap_etch]
        c_items = draw.rotate(c_items, p.orientation, origin=(0, 0))
        c_items = draw.translate(c_items, p.pos_x, p.pos_y)
        [cap_north_body, cap_south_body, cap_etch] = c_items

        self.add_qgeometry("poly", {"cap_etch": cap_etch}, layer=p.layer, subtract=True)
        self.add_qgeometry("poly", {"cap_north_body": cap_north_body}, layer=p.layer)
        self.add_qgeometry("poly", {"cap_south_body": cap_south_body}, layer=p.layer)

        south_cpw_edge_coords = np.array(
            [
                (self.south_cpw_left_edge_x, -p.south_cpw_length + y_translation_offset),
                (self.south_cpw_right_edge_x, -p.south_cpw_length + y_translation_offset),
            ]
        )
        north_cpw_edge_coords = np.array(
            [
                (self.north_cpw_left_edge_x, self.cap_height + p.north_cpw_length + y_translation_offset),
                (self.north_cpw_right_edge_x, self.cap_height + p.north_cpw_length + y_translation_offset),
            ]
        )
        cpw_edge_lines = [
            LineString(south_cpw_edge_coords),
            LineString(north_cpw_edge_coords),
        ]
        cpw_edge_lines = draw.rotate(cpw_edge_lines, p.orientation, origin=(0, 0))
        cpw_edge_lines = draw.translate(cpw_edge_lines, p.pos_x, p.pos_y)
        south_cpw_edge_coords = np.array(cpw_edge_lines[0].coords)
        north_cpw_edge_coords = np.array(cpw_edge_lines[1].coords)

        self.add_pin(
            "south_end",
            points=south_cpw_edge_coords[::-1],
            width=p.south_cpw_width,
            gap=p.south_cpw_gap,
        )
        self.add_pin(
            "north_end",
            points=north_cpw_edge_coords,
            width=p.north_cpw_width,
            gap=p.north_cpw_gap,
        )

    def _initialize_calculated_variables(self):
        p = self.p

        self.cap_width = p.finger_count * p.finger_width + (p.finger_count - 1) * p.finger_gap_east_west
        self.cap_height = p.south_spine_width + p.finger_length + p.finger_gap_north_south + p.north_spine_width

        self.south_cpw_left_edge_x = p.south_cpw_xpos_offset - p.south_cpw_width / 2
        self.south_cpw_right_edge_x = p.south_cpw_xpos_offset + p.south_cpw_width / 2
        self.north_cpw_left_edge_x = p.north_cpw_xpos_offset - p.north_cpw_width / 2
        self.north_cpw_right_edge_x = p.north_cpw_xpos_offset + p.north_cpw_width / 2

        self.etch_width = self.cap_width + p.west_gap_ground + p.east_gap_ground
        self.etch_height = self.cap_height + p.north_gap_ground + p.south_gap_ground
        self.etch_left_edge_x = -self.cap_width / 2 - p.west_gap_ground
        self.etch_right_edge_x = self.cap_width / 2 + p.east_gap_ground

        self.finger_gap_width = p.finger_width + 2 * p.finger_gap_east_west

    def _get_rounded_corner_coords(
        self, r, quadrant, corner_coords, clockwise=False, start_angle=None, end_angle=None, num_points=16
    ):
        if r < 1e-6:
            return [corner_coords]

        x = corner_coords[0]
        y = corner_coords[1]

        if quadrant == "ne":
            angle_start = 0
            angle_end = np.pi / 2
            if clockwise:
                angle_start, angle_end = angle_end, angle_start
            x = x - r
            y = y - r

        elif quadrant == "nw":
            angle_start = np.pi / 2
            angle_end = np.pi
            if clockwise:
                angle_start, angle_end = angle_end, angle_start
            x = x + r
            y = y - r

        elif quadrant == "sw":
            angle_start = -np.pi
            angle_end = -np.pi / 2
            if clockwise:
                angle_start, angle_end = angle_end, angle_start
            x = x + r
            y = y + r

        elif quadrant == "se":
            angle_start = -np.pi / 2
            angle_end = 0
            if clockwise:
                angle_start, angle_end = angle_end, angle_start
            x = x - r
            y = y + r

        if start_angle is None:
            start_angle = angle_start
        if end_angle is None:
            end_angle = angle_end

        theta = np.linspace(start_angle, end_angle, num_points)
        x_vals = r * np.cos(theta) + x
        y_vals = r * np.sin(theta) + y
        return [(float(xv), float(yv)) for xv, yv in zip(x_vals, y_vals)]

    def _get_partial_corner_info(self, cpw_center, spine_center, cpw_radius, spine_radius, y_int_val=0.0):
        x1, y1 = cpw_center
        x2, y2 = spine_center
        R1 = cpw_radius
        R2 = spine_radius

        dx, dy = x2 - x1, y2 - y1
        d = np.sqrt(dx**2 + dy**2)

        c = (R1 + R2) / d
        s = np.sqrt(d**2 - (R1 + R2) ** 2) / d

        nx1 = c * (dx / d) - s * (dy / d)
        ny1 = c * (dy / d) + s * (dx / d)

        nx2 = c * (dx / d) + s * (dy / d)
        ny2 = c * (dy / d) - s * (dx / d)

        pt1x_circ1, pt1y_circ1 = x1 + R1 * nx1, y1 + R1 * ny1
        pt1x_circ2, pt1y_circ2 = x2 - R2 * nx1, y2 - R2 * ny1

        pt2x_circ1, pt2y_circ1 = x1 + R1 * nx2, y1 + R1 * ny2
        pt2x_circ2, pt2y_circ2 = x2 - R2 * nx2, y2 - R2 * ny2

        if abs(pt1y_circ1 - y_int_val) < 1e-6:
            pt_circ1 = np.array([pt2x_circ1, pt2y_circ1])
            pt_circ2 = np.array([pt2x_circ2, pt2y_circ2])
        else:
            pt_circ1 = np.array([pt1x_circ1, pt1y_circ1])
            pt_circ2 = np.array([pt1x_circ2, pt1y_circ2])

        numerator = (pt_circ1[0] * pt_circ2[1] - pt_circ2[0] * pt_circ1[1]) + y_int_val * (pt_circ2[0] - pt_circ1[0])
        denominator = pt_circ2[1] - pt_circ1[1]
        int_point = np.array([numerator / denominator, y_int_val])

        xc1, yc1 = cpw_center
        xt1, yt1 = pt_circ1
        dx = xt1 - xc1
        dy = yt1 - yc1
        angle1 = np.arctan2(dy, dx)

        xc2, yc2 = spine_center
        xt2, yt2 = pt_circ2
        dx = xt2 - xc2
        dy = yt2 - yc2
        angle2 = np.arctan2(dy, dx)

        info_dict = {
            "int_point": int_point.tolist(),
            "pt_circ1": pt_circ1.tolist(),
            "pt_circ2": pt_circ2.tolist(),
            "cpw_angle": angle1,
            "spine_angle": angle2,
        }
        return info_dict

    def _get_finger_coords(self, start_coord, exclude_left_ledge=False, exclude_right_ledge=False):
        p = self.p
        length = p.finger_length
        width = p.finger_width
        rf = p.finger_radius
        re = p.finger_etch_radius
        gap_width = self.finger_gap_width
        start_x = start_coord[0]
        start_y = start_coord[1]

        coords = []
        x = start_x
        y = start_y

        if exclude_left_ledge:
            y = y + length
            coords.extend(self._get_rounded_corner_coords(rf, "nw", (x, y), clockwise=True))
        else:
            x = x + gap_width / 2
            coords.extend(self._get_rounded_corner_coords(re, "se", (x, y)))

            y = y + length
            coords.extend(self._get_rounded_corner_coords(rf, "nw", (x, y), clockwise=True))

        x = x + width
        coords.extend(self._get_rounded_corner_coords(rf, "ne", (x, y), clockwise=True))

        if not exclude_right_ledge:
            y = y - length
            coords.extend(self._get_rounded_corner_coords(re, "sw", (x, y)))

        return coords, x, y

    def _spine_ledge_geometry(self, side: str, y: float) -> dict:
        """Circle/fillet centers for the spine ledge at the outer finger gap."""
        p = self.p
        if side == "left":
            circ_center_x = -self.cap_width / 2 - p.finger_gap_east_west + p.finger_etch_radius
            fillet_center_x = -self.cap_width / 2 + p.finger_edge_gap_radius
        else:
            circ_center_x = self.cap_width / 2 + p.finger_gap_east_west - p.finger_etch_radius
            fillet_center_x = self.cap_width / 2 - p.finger_edge_gap_radius
        circ_center_y = y + p.finger_etch_radius
        radicand = (p.finger_etch_radius + p.finger_edge_gap_radius) ** 2 - (fillet_center_x - circ_center_x) ** 2
        fillet_center_y = None
        fillet_upper_y = None
        circ_lower_y = circ_center_y - p.finger_etch_radius
        if radicand >= 0:
            fillet_center_y = circ_center_y - np.sqrt(radicand)
            fillet_upper_y = fillet_center_y + p.finger_edge_gap_radius
        return {
            "circ_center_x": circ_center_x,
            "circ_center_y": circ_center_y,
            "fillet_center_x": fillet_center_x,
            "fillet_center_y": fillet_center_y,
            "fillet_upper_y": fillet_upper_y,
            "circ_lower_y": circ_lower_y,
            "radicand": radicand,
        }

    def _spine_ledge_use_simplified(self, side: str, y: float) -> bool:
        """Use simplified spine-ledge corners when fillet and etch circles overlap.

        Simplified path when the fillet top is at or above the etch-circle bottom
        (within 1e-6 mm). Otherwise use tangent fillet + etch-arc construction.
        """
        geom = self._spine_ledge_geometry(side, y)
        if geom["radicand"] < 0:
            return True
        return (geom["circ_lower_y"] - geom["fillet_upper_y"]) <= 1e-6

    def _append_spine_ledge_left_simplified(self, cap_body_coords: list, y: float) -> None:
        cap_body_coords.extend(
            self._get_rounded_corner_coords(
                self.p.finger_edge_gap_radius,
                "nw",
                (-self.cap_width / 2, y),
                start_angle=np.pi,
                end_angle=np.pi / 2,
                clockwise=True,
            )
        )

    def _append_spine_ledge_right_simplified(self, cap_body_coords: list, y: float) -> None:
        cap_body_coords.extend(
            self._get_rounded_corner_coords(
                self.p.finger_edge_gap_radius,
                "ne",
                (self.cap_width / 2, y),
                start_angle=np.pi / 2,
                end_angle=0,
                clockwise=True,
            )
        )

    def _append_spine_ledge_left_tangent(self, cap_body_coords: list, y: float) -> None:
        p = self.p
        geom = self._spine_ledge_geometry("left", y)
        circ_center_x = geom["circ_center_x"]
        circ_center_y = geom["circ_center_y"]
        fillet_center_x = geom["fillet_center_x"]
        fillet_center_y = geom["fillet_center_y"]
        fillet_angle_end = np.arctan2(circ_center_y - fillet_center_y, circ_center_x - fillet_center_x)
        theta_start = np.arctan2(fillet_center_y - circ_center_y, fillet_center_x - circ_center_x)
        cap_body_coords.extend(
            self._get_rounded_corner_coords(
                p.finger_edge_gap_radius,
                "nw",
                (-self.cap_width / 2, fillet_center_y + p.finger_edge_gap_radius),
                start_angle=np.pi,
                end_angle=fillet_angle_end,
            )
        )
        cap_body_coords.extend(
            self._get_rounded_corner_coords(
                p.finger_etch_radius,
                "sw",
                (circ_center_x - p.finger_etch_radius, y),
                start_angle=theta_start,
                end_angle=-np.pi / 2,
            )
        )

    def _append_spine_ledge_right_tangent(self, cap_body_coords: list, y: float) -> None:
        p = self.p
        geom = self._spine_ledge_geometry("right", y)
        circ_center_x = geom["circ_center_x"]
        circ_center_y = geom["circ_center_y"]
        fillet_center_x = geom["fillet_center_x"]
        fillet_center_y = geom["fillet_center_y"]
        fillet_angle_start = np.arctan2(circ_center_y - fillet_center_y, circ_center_x - fillet_center_x)
        theta_end = np.arctan2(fillet_center_y - circ_center_y, fillet_center_x - circ_center_x)
        cap_body_coords.extend(
            self._get_rounded_corner_coords(
                p.finger_etch_radius,
                "se",
                (circ_center_x + p.finger_etch_radius, y),
                start_angle=-np.pi / 2,
                end_angle=theta_end,
            )
        )
        cap_body_coords.extend(
            self._get_rounded_corner_coords(
                p.finger_edge_gap_radius,
                "ne",
                (self.cap_width / 2, fillet_center_y + p.finger_edge_gap_radius),
                start_angle=fillet_angle_start,
                end_angle=0,
            )
        )

    def _make_cap_body(self, is_body_south, finger_count, gap_first=False):
        p = self.p
        if is_body_south:
            spine_width = p.south_spine_width
            west_spine_radius = p.south_west_spine_radius
            east_spine_radius = p.south_east_spine_radius
            cpw_width = p.south_cpw_width
            cpw_length = p.south_cpw_length
            cpw_radius = p.south_cpw_radius
            cpw_xpos = p.south_cpw_xpos_offset
            cpw_left_edge_x = self.south_cpw_left_edge_x
            cpw_right_edge_x = self.south_cpw_right_edge_x
            dir_text = "south_"
        else:
            spine_width = p.north_spine_width
            west_spine_radius = (
                p.north_east_spine_radius
            )  # Direction flipped since north body will be rotated 180 degrees
            east_spine_radius = p.north_west_spine_radius
            cpw_width = p.north_cpw_width
            cpw_length = p.north_cpw_length
            cpw_radius = p.north_cpw_radius
            cpw_xpos = p.north_cpw_xpos_offset
            cpw_left_edge_x = self.north_cpw_left_edge_x
            cpw_right_edge_x = self.north_cpw_right_edge_x
            dir_text = "north_"

        assert cpw_length >= cpw_radius, (
            f"{dir_text}cpw_length must be larger than or equal to {dir_text}cpw_radius in order to render"
        )

        cap_body_coords = []

        # (0,0) is south spine bottom edge horizontally center

        # LEFT: CPW AND SPINE
        if cpw_length > 0.0:
            x = cpw_xpos
            y = -cpw_length

            # cpw left corner
            x = x - cpw_width / 2
            cap_body_coords.append((x, y))

            # cpw spine left connection curve
            left_edge_x = -self.cap_width / 2
            y = 0

            wr_partial_valid = (left_edge_x < cpw_left_edge_x) and (
                (cpw_left_edge_x - cpw_radius) < (left_edge_x + west_spine_radius)
            )
            wl_partial_valid = (left_edge_x > cpw_left_edge_x) and (
                (cpw_left_edge_x + cpw_radius) > (left_edge_x - west_spine_radius)
            )
            w_equality_valid = np.abs(left_edge_x - cpw_left_edge_x) < 1e-6

            if not any([wr_partial_valid, wl_partial_valid, w_equality_valid]):
                # full pi/2 rotations for both curves
                if left_edge_x < cpw_left_edge_x:  # cpw fully connected to spine
                    cap_body_coords.extend(self._get_rounded_corner_coords(cpw_radius, "ne", (x, y)))
                    x = left_edge_x
                    cap_body_coords.extend(
                        self._get_rounded_corner_coords(west_spine_radius, "sw", (x, y), clockwise=True)
                    )
                else:  # cpw sticking out to the left of the cap body
                    cap_body_coords.extend(self._get_rounded_corner_coords(cpw_radius, "nw", (x, y), clockwise=True))
                    x = left_edge_x
                    cap_body_coords.extend(self._get_rounded_corner_coords(west_spine_radius, "se", (x, y)))

            else:
                if w_equality_valid:  # straight connection from cpw to spine
                    cap_body_coords.append((x, y))

                elif wr_partial_valid:  # ne cpw curve -> tilted line -> sw spine curve
                    spine_circ_center = np.array([left_edge_x + west_spine_radius, west_spine_radius])
                    cpw_circ_center = np.array([cpw_left_edge_x - cpw_radius, -cpw_radius])
                    partial_info = self._get_partial_corner_info(
                        cpw_circ_center, spine_circ_center, cpw_radius, west_spine_radius
                    )

                    cpw_end_angle = partial_info["cpw_angle"]
                    assert cpw_end_angle >= 0.0, "DEBUG"
                    cap_body_coords.extend(
                        self._get_rounded_corner_coords(
                            cpw_radius, "ne", (x, y), start_angle=0, end_angle=cpw_end_angle
                        )
                    )

                    cap_body_coords.append(partial_info["int_point"])

                    x = left_edge_x
                    spine_start_angle = partial_info["spine_angle"]
                    assert spine_start_angle <= 0.0, "DEBUG"
                    cap_body_coords.extend(
                        self._get_rounded_corner_coords(
                            west_spine_radius,
                            "sw",
                            (x, y),
                            clockwise=True,
                            start_angle=spine_start_angle,
                            end_angle=-np.pi,
                        )
                    )

                elif wl_partial_valid:  # nw cpw curve -> tilted line -> se spine curve
                    assert left_edge_x < cpw_right_edge_x, (
                        f"{dir_text}cpw must be connected to capacitor (exceeds {left_edge_x} mm). Reduce offset magnitude"
                    )

                    spine_circ_center = np.array([left_edge_x - west_spine_radius, west_spine_radius])
                    cpw_circ_center = np.array([cpw_left_edge_x + cpw_radius, -cpw_radius])
                    partial_info = self._get_partial_corner_info(
                        cpw_circ_center, spine_circ_center, cpw_radius, west_spine_radius
                    )

                    cpw_end_angle = partial_info["cpw_angle"]
                    assert cpw_end_angle >= 0.0, "DEBUG"
                    cap_body_coords.extend(
                        self._get_rounded_corner_coords(
                            cpw_radius, "nw", (x, y), start_angle=np.pi, end_angle=cpw_end_angle
                        )
                    )

                    cap_body_coords.append(partial_info["int_point"])

                    x = -self.cap_width / 2
                    spine_start_angle = partial_info["spine_angle"]
                    assert spine_start_angle <= 0.0, "DEBUG"
                    cap_body_coords.extend(
                        self._get_rounded_corner_coords(
                            west_spine_radius, "se", (x, y), clockwise=True, start_angle=spine_start_angle, end_angle=0
                        )
                    )

                else:
                    raise RuntimeError("Should not reach this point. Code logic issue")

        else:
            # no cpw, just spine
            x = -self.cap_width / 2
            y = 0
            cap_body_coords.extend(self._get_rounded_corner_coords(west_spine_radius, "sw", (x, y), clockwise=True))

        # FINGERS
        y = y + spine_width

        if p.finger_length > 0.0:
            if gap_first:
                if self._spine_ledge_use_simplified("left", y):
                    self._append_spine_ledge_left_simplified(cap_body_coords, y)
                else:
                    self._append_spine_ledge_left_tangent(cap_body_coords, y)
                x = x + p.finger_width + p.finger_gap_east_west - self.finger_gap_width / 2

            for i in range(finger_count):
                coords, x, y = self._get_finger_coords(
                    [x, y],
                    exclude_left_ledge=True if i == 0 and not gap_first else False,
                    exclude_right_ledge=True
                    if i == finger_count - 1 and not gap_first and p.finger_count % 2 == 1
                    else False,
                )
                if i != finger_count - 1:
                    x = x + self.finger_gap_width / 2
                cap_body_coords.extend(coords)

            if p.finger_count % 2 == 0 or gap_first:
                if self._spine_ledge_use_simplified("right", y):
                    self._append_spine_ledge_right_simplified(cap_body_coords, y)
                else:
                    self._append_spine_ledge_right_tangent(cap_body_coords, y)
        else:
            cap_body_coords.extend(
                self._get_rounded_corner_coords(
                    p.finger_edge_gap_radius,
                    "nw",
                    (x, y),
                    clockwise=True,
                )
            )
            x = x + self.cap_width
            cap_body_coords.extend(
                self._get_rounded_corner_coords(
                    p.finger_edge_gap_radius,
                    "ne",
                    (x, y),
                    clockwise=True,
                )
            )

        # RIGHT: CPW AND SPINE
        right_cpw_spine_coords = []
        if cpw_length > 0.0:
            x = cpw_xpos
            y = -cpw_length

            # cpw left corner
            x = x + cpw_width / 2
            right_cpw_spine_coords.append((x, y))

            # cpw spine left connection curve
            right_edge_x = self.cap_width / 2
            y = 0

            er_partial_valid = (right_edge_x < cpw_right_edge_x) and (
                (cpw_right_edge_x - cpw_radius) < (right_edge_x + east_spine_radius)
            )
            el_partial_valid = (right_edge_x > cpw_right_edge_x) and (
                (cpw_right_edge_x + cpw_radius) > (right_edge_x - east_spine_radius)
            )
            e_equality_valid = np.abs(right_edge_x - cpw_right_edge_x) < 1e-6

            if not any([er_partial_valid, el_partial_valid, e_equality_valid]):
                # full pi/2 rotations for both curves
                if right_edge_x > cpw_right_edge_x:  # cpw fully connected to spine
                    right_cpw_spine_coords.extend(
                        self._get_rounded_corner_coords(cpw_radius, "nw", (x, y), clockwise=True)
                    )
                    x = right_edge_x
                    right_cpw_spine_coords.extend(self._get_rounded_corner_coords(east_spine_radius, "se", (x, y)))
                else:  # cpw sticking out to the left of the cap body
                    right_cpw_spine_coords.extend(self._get_rounded_corner_coords(cpw_radius, "ne", (x, y)))
                    x = right_edge_x
                    right_cpw_spine_coords.extend(
                        self._get_rounded_corner_coords(east_spine_radius, "sw", (x, y), clockwise=True)
                    )

            else:
                if e_equality_valid:  # straight connection from cpw to spine
                    right_cpw_spine_coords.append((x, y))

                elif el_partial_valid:  # nw cpw curve -> tilted line -> se spine curve
                    spine_circ_center = np.array([right_edge_x - east_spine_radius, east_spine_radius])
                    cpw_circ_center = np.array([cpw_right_edge_x + cpw_radius, -cpw_radius])
                    partial_info = self._get_partial_corner_info(
                        cpw_circ_center, spine_circ_center, cpw_radius, east_spine_radius
                    )

                    cpw_end_angle = partial_info["cpw_angle"]
                    assert cpw_end_angle >= 0.0, "DEBUG"
                    right_cpw_spine_coords.extend(
                        self._get_rounded_corner_coords(
                            cpw_radius, "nw", (x, y), start_angle=np.pi, end_angle=cpw_end_angle, clockwise=True
                        )
                    )

                    right_cpw_spine_coords.append(partial_info["int_point"])

                    x = right_edge_x
                    spine_start_angle = partial_info["spine_angle"]
                    assert spine_start_angle <= 0.0, "DEBUG"
                    right_cpw_spine_coords.extend(
                        self._get_rounded_corner_coords(
                            east_spine_radius, "se", (x, y), start_angle=spine_start_angle, end_angle=0
                        )
                    )

                elif er_partial_valid:  # ne cpw curve -> tilted line -> sw spine curve
                    assert right_edge_x > cpw_left_edge_x, (
                        f"{dir_text}cpw must be connected to capacitor (exceeds {right_edge_x} mm). Reduce offset magnitude"
                    )

                    spine_circ_center = np.array([right_edge_x + east_spine_radius, east_spine_radius])
                    cpw_circ_center = np.array([cpw_right_edge_x - cpw_radius, -cpw_radius])
                    partial_info = self._get_partial_corner_info(
                        cpw_circ_center, spine_circ_center, cpw_radius, east_spine_radius
                    )

                    cpw_end_angle = partial_info["cpw_angle"]
                    assert cpw_end_angle >= 0.0, "DEBUG"
                    right_cpw_spine_coords.extend(
                        self._get_rounded_corner_coords(
                            cpw_radius, "ne", (x, y), start_angle=0, end_angle=cpw_end_angle
                        )
                    )

                    right_cpw_spine_coords.append(partial_info["int_point"])

                    x = right_edge_x
                    spine_start_angle = partial_info["spine_angle"]
                    assert spine_start_angle <= 0.0, "DEBUG"
                    right_cpw_spine_coords.extend(
                        self._get_rounded_corner_coords(
                            east_spine_radius,
                            "sw",
                            (x, y),
                            clockwise=True,
                            start_angle=spine_start_angle,
                            end_angle=-np.pi,
                        )
                    )

                else:
                    raise RuntimeError("Should not reach this point. Code logic issue")

        else:
            # no cpw, just spine
            x = self.cap_width / 2
            y = 0
            right_cpw_spine_coords.extend(self._get_rounded_corner_coords(cpw_radius, "nw", (x, y)))

        cap_body_coords.extend(right_cpw_spine_coords[::-1])

        cap_body = Polygon(cap_body_coords)
        return cap_body

    def _make_cap_etch(self):
        p = self.p

        cap_etch_coords = []

        south_cpw_etch_left_edge_x = self.south_cpw_left_edge_x - p.south_cpw_gap
        south_cpw_etch_right_edge_x = self.south_cpw_right_edge_x + p.south_cpw_gap
        north_cpw_etch_left_edge_x = self.north_cpw_left_edge_x - p.north_cpw_gap
        north_cpw_etch_right_edge_x = self.north_cpw_right_edge_x + p.north_cpw_gap

        south_cpw_etch_length = p.south_cpw_length - p.south_gap_ground
        north_cpw_etch_length = p.north_cpw_length - p.north_gap_ground

        # SOUTH WEST CORNER
        if p.south_cpw_length > p.south_gap_ground:
            x = south_cpw_etch_left_edge_x
            y = -south_cpw_etch_length
            cap_etch_coords.append((x, y))

            y = 0

            swr_partial_valid = (self.etch_left_edge_x < south_cpw_etch_left_edge_x) and (
                (south_cpw_etch_left_edge_x - p.south_cpw_etch_radius)
                < (self.etch_left_edge_x + p.south_west_spine_etch_radius)
            )
            swl_partial_valid = (self.etch_left_edge_x > south_cpw_etch_left_edge_x) and (
                (south_cpw_etch_left_edge_x + p.south_cpw_etch_radius)
                > (self.etch_left_edge_x - p.south_west_spine_etch_radius)
            )
            sw_equality_valid = np.abs(self.etch_left_edge_x - south_cpw_etch_left_edge_x) < 1e-6

            if not any([swr_partial_valid, swl_partial_valid, sw_equality_valid]):
                # fill pi/2 rotations for both curves
                if self.etch_left_edge_x < south_cpw_etch_left_edge_x:  # cpw fully connected to spine
                    cap_etch_coords.extend(self._get_rounded_corner_coords(p.south_cpw_etch_radius, "ne", (x, y)))
                    x = self.etch_left_edge_x
                    cap_etch_coords.extend(
                        self._get_rounded_corner_coords(p.south_west_spine_etch_radius, "sw", (x, y), clockwise=True)
                    )
                else:  # cpw sticking out to the left of the cap body
                    cap_etch_coords.extend(
                        self._get_rounded_corner_coords(p.south_cpw_etch_radius, "nw", (x, y), clockwise=True)
                    )
                    x = self.etch_left_edge_x
                    cap_etch_coords.extend(
                        self._get_rounded_corner_coords(p.south_west_spine_etch_radius, "se", (x, y))
                    )

            else:
                if sw_equality_valid:  # straight connection from cpw to spine
                    cap_etch_coords.append((x, y))

                elif swr_partial_valid:  # ne cpw curve -> tilted line -> sw spine curve
                    spine_circ_center = np.array(
                        [self.etch_left_edge_x + p.south_west_spine_etch_radius, p.south_west_spine_etch_radius]
                    )
                    cpw_circ_center = np.array(
                        [south_cpw_etch_left_edge_x - p.south_cpw_etch_radius, -p.south_cpw_etch_radius]
                    )
                    partial_info = self._get_partial_corner_info(
                        cpw_circ_center, spine_circ_center, p.south_cpw_etch_radius, p.south_west_spine_etch_radius
                    )

                    cpw_end_angle = partial_info["cpw_angle"]
                    assert cpw_end_angle >= 0.0, "DEBUG"
                    cap_etch_coords.extend(
                        self._get_rounded_corner_coords(
                            p.south_cpw_etch_radius, "ne", (x, y), start_angle=0, end_angle=cpw_end_angle
                        )
                    )

                    cap_etch_coords.append(partial_info["int_point"])

                    x = self.etch_left_edge_x
                    spine_start_angle = partial_info["spine_angle"]
                    assert spine_start_angle <= 0.0, "DEBUG"
                    cap_etch_coords.extend(
                        self._get_rounded_corner_coords(
                            p.south_west_spine_etch_radius,
                            "sw",
                            (x, y),
                            clockwise=True,
                            start_angle=spine_start_angle,
                            end_angle=-np.pi,
                        )
                    )

                elif swl_partial_valid:  # nw cpw curve -> tilted line -> se spine curve
                    assert self.etch_left_edge_x < south_cpw_etch_right_edge_x, (
                        f"south_cpw must be connected to capacitor (exceeds {self.etch_left_edge_x} mm). Reduce offset magnitude"
                    )

                    spine_circ_center = np.array(
                        [self.etch_left_edge_x - p.south_west_spine_etch_radius, p.south_west_spine_etch_radius]
                    )
                    cpw_circ_center = np.array(
                        [south_cpw_etch_left_edge_x + p.south_cpw_etch_radius, -p.south_cpw_etch_radius]
                    )
                    partial_info = self._get_partial_corner_info(
                        cpw_circ_center, spine_circ_center, p.south_cpw_etch_radius, p.south_west_spine_etch_radius
                    )

                    cpw_end_angle = partial_info["cpw_angle"]
                    assert cpw_end_angle >= 0.0, "DEBUG"
                    cap_etch_coords.extend(
                        self._get_rounded_corner_coords(
                            p.south_cpw_etch_radius, "nw", (x, y), start_angle=np.pi, end_angle=cpw_end_angle
                        )
                    )

                    cap_etch_coords.append(partial_info["int_point"])

                    x = -self.cap_width / 2
                    spine_start_angle = partial_info["spine_angle"]
                    assert spine_start_angle <= 0.0, "DEBUG"
                    cap_etch_coords.extend(
                        self._get_rounded_corner_coords(
                            p.south_west_spine_etch_radius,
                            "se",
                            (x, y),
                            clockwise=True,
                            start_angle=spine_start_angle,
                            end_angle=0,
                        )
                    )

                else:
                    raise RuntimeError("Should not reach this point. Code logic issue")

        elif p.south_cpw_length > 0.0 and (p.south_cpw_length < p.south_gap_ground):
            south_cpw_etch_left_edge_x = self.south_cpw_left_edge_x - p.south_cpw_gap

            if south_cpw_etch_left_edge_x < self.etch_width / 2:
                raise NotImplementedError

        elif p.south_cpw_length < 1e-6:
            raise NotImplementedError

        # NORTH WEST CORNER
        if p.south_cpw_length > p.south_gap_ground:
            x = self.etch_left_edge_x
            y = self.etch_height

            nwr_partial_valid = (self.etch_left_edge_x < north_cpw_etch_left_edge_x) and (
                (north_cpw_etch_left_edge_x - p.north_cpw_etch_radius)
                < (self.etch_left_edge_x + p.north_west_spine_etch_radius)
            )
            nwl_partial_valid = (self.etch_left_edge_x > north_cpw_etch_left_edge_x) and (
                (north_cpw_etch_left_edge_x + p.north_cpw_etch_radius)
                > (self.etch_left_edge_x - p.north_west_spine_etch_radius)
            )
            nw_equality_valid = np.abs(self.etch_left_edge_x - north_cpw_etch_left_edge_x) < 1e-6

            if not any([nwr_partial_valid, nwl_partial_valid, nw_equality_valid]):
                # fill pi/2 rotations for both curves
                if self.etch_right_edge_x > south_cpw_etch_right_edge_x:  # cpw fully connected to spine
                    cap_etch_coords.extend(
                        self._get_rounded_corner_coords(p.north_west_spine_etch_radius, "nw", (x, y), clockwise=True)
                    )
                    x = north_cpw_etch_left_edge_x
                    cap_etch_coords.extend(self._get_rounded_corner_coords(p.north_cpw_etch_radius, "se", (x, y)))
                else:  # cpw sticking out to the left of the cap body
                    cap_etch_coords.extend(
                        self._get_rounded_corner_coords(p.north_west_spine_etch_radius, "ne", (x, y))
                    )
                    x = north_cpw_etch_left_edge_x
                    cap_etch_coords.extend(
                        self._get_rounded_corner_coords(p.north_cpw_etch_radius, "sw", (x, y), clockwise=True)
                    )

            else:
                if nw_equality_valid:  # straight connection from cpw to spine
                    cap_etch_coords.append((x, y))

                elif nwr_partial_valid:  # nw spine curve -> tilted line -> se cpw curve
                    raise NotImplementedError
                    spine_circ_center = np.array(
                        [
                            self.etch_left_edge_x + p.north_west_spine_etch_radius,
                            self.etch_height - p.north_west_spine_etch_radius,
                        ]
                    )
                    cpw_circ_center = np.array(
                        [
                            north_cpw_etch_left_edge_x - p.north_cpw_etch_radius,
                            self.etch_height + p.north_cpw_etch_radius,
                        ]
                    )
                    partial_info = self._get_partial_corner_info(
                        cpw_circ_center,
                        spine_circ_center,
                        p.north_cpw_etch_radius,
                        p.north_west_spine_etch_radius,
                        y_int_val=self.etch_height,
                    )

                    spine_end_angle = partial_info["spine_angle"]
                    assert spine_end_angle >= 0.0, "DEBUG"
                    cap_etch_coords.extend(
                        self._get_rounded_corner_coords(
                            p.north_west_spine_etch_radius,
                            "nw",
                            (x, y),
                            clockwise=True,
                            start_angle=np.pi,
                            end_angle=spine_end_angle,
                        )
                    )

                    cap_etch_coords.append(partial_info["int_point"])

                    x = north_cpw_etch_left_edge_x
                    cpw_start_angle = partial_info["cpw_angle"]
                    assert cpw_start_angle <= 0.0, "DEBUG"
                    cap_etch_coords.extend(
                        self._get_rounded_corner_coords(
                            p.north_cpw_etch_radius, "se", (x, y), start_angle=cpw_start_angle, end_angle=0
                        )
                    )

                elif nwl_partial_valid:  # ne spine curve -> tilted line -> sw cpw curve
                    raise NotImplementedError
                    assert self.etch_left_edge_x < south_cpw_etch_right_edge_x, (
                        f"south_cpw must be connected to capacitor (exceeds {self.etch_left_edge_x} mm). Reduce offset magnitude"
                    )

                    y_int_val = self.etch_height - p.north_gap_ground

                    spine_circ_center = np.array(
                        [self.etch_left_edge_x - p.north_west_spine_etch_radius, y_int_val - p.north_west_spine_radius]
                    )
                    cpw_circ_center = np.array(
                        [north_cpw_etch_left_edge_x + p.north_cpw_etch_radius, y_int_val + p.north_cpw_radius]
                    )
                    partial_info = self._get_partial_corner_info(
                        cpw_circ_center,
                        spine_circ_center,
                        p.north_cpw_etch_radius,
                        p.north_west_spine_etch_radius,
                        y_int_val=y_int_val,
                    )

                    y = y_int_val
                    spine_end_angle = partial_info["spine_angle"]
                    assert spine_end_angle >= 0.0, "DEBUG"
                    cap_etch_coords.extend(
                        self._get_rounded_corner_coords(
                            p.north_west_spine_etch_radius, "ne", (x, y), start_angle=0, end_angle=spine_end_angle
                        )
                    )

                    cap_etch_coords.append(partial_info["int_point"])

                    x = north_cpw_etch_left_edge_x
                    cpw_start_angle = partial_info["cpw_angle"]
                    assert cpw_start_angle <= 0.0, "DEBUG"
                    cap_etch_coords.extend(
                        self._get_rounded_corner_coords(
                            p.north_cpw_etch_radius,
                            "sw",
                            (x, y),
                            start_angle=cpw_start_angle,
                            end_angle=-np.pi,
                            clockwise=True,
                        )
                    )

                else:
                    raise RuntimeError("Should not reach this point. Code logic issue")

            x = north_cpw_etch_left_edge_x
            y = self.etch_height + north_cpw_etch_length
            cap_etch_coords.append((x, y))

        elif p.south_cpw_length > 0.0 and (p.south_cpw_length < p.south_gap_ground):
            south_cpw_etch_left_edge_x = self.south_cpw_left_edge_x - p.south_cpw_gap

            if south_cpw_etch_left_edge_x < self.etch_width / 2:
                raise NotImplementedError

        elif p.south_cpw_length < 1e-6:
            raise NotImplementedError

        # NORTH EAST CORNER
        if p.north_cpw_length > p.north_gap_ground:
            x = north_cpw_etch_right_edge_x
            y = self.etch_height + north_cpw_etch_length
            cap_etch_coords.append((x, y))

            y = self.etch_height
            ner_partial_valid = (self.etch_right_edge_x < north_cpw_etch_right_edge_x) and (
                (north_cpw_etch_right_edge_x - p.north_cpw_etch_radius)
                > (self.etch_left_edge_x + p.north_east_spine_etch_radius)
            )
            nel_partial_valid = (self.etch_right_edge_x > north_cpw_etch_right_edge_x) and (
                (north_cpw_etch_right_edge_x + p.north_cpw_etch_radius)
                < (self.etch_left_edge_x - p.north_east_spine_etch_radius)
            )
            ne_equality_valid = np.abs(self.etch_right_edge_x - north_cpw_etch_right_edge_x) < 1e-6

            if not any([ner_partial_valid, nel_partial_valid, ne_equality_valid]):
                # fill pi/2 rotations for both curves
                if self.etch_right_edge_x > north_cpw_etch_right_edge_x:  # cpw fully connected to spine
                    cap_etch_coords.extend(self._get_rounded_corner_coords(p.north_cpw_etch_radius, "sw", (x, y)))
                    x = self.etch_right_edge_x
                    cap_etch_coords.extend(
                        self._get_rounded_corner_coords(p.north_east_spine_etch_radius, "ne", (x, y), clockwise=True)
                    )
                else:  # cpw sticking out to the left of the cap body
                    cap_etch_coords.extend(
                        self._get_rounded_corner_coords(p.north_cpw_etch_radius, "se", (x, y), clockwise=True)
                    )
                    x = self.etch_right_edge_x
                    cap_etch_coords.extend(
                        self._get_rounded_corner_coords(p.north_east_spine_etch_radius, "nw", (x, y))
                    )

            else:
                if ne_equality_valid:  # straight connection from cpw to spine
                    cap_etch_coords.append((x, y))

                elif nel_partial_valid:  # sw cpw curve -> tilted line -> ne spine curve
                    raise NotImplementedError
                    spine_circ_center = np.array(
                        [
                            self.etch_right_edge_x - p.north_east_spine_etch_radius,
                            self.etch_height - p.north_east_spine_etch_radius,
                        ]
                    )
                    cpw_circ_center = np.array(
                        [
                            north_cpw_etch_right_edge_x + p.north_cpw_etch_radius,
                            self.etch_height + p.north_cpw_etch_radius,
                        ]
                    )
                    partial_info = self._get_partial_corner_info(
                        cpw_circ_center, spine_circ_center, p.north_cpw_etch_radius, p.north_east_spine_etch_radius
                    )

                    x = north_cpw_etch_right_edge_x
                    cpw_end_angle = partial_info["cpw_angle"]
                    assert cpw_end_angle <= 0.0, "DEBUG"
                    cap_etch_coords.extend(
                        self._get_rounded_corner_coords(
                            p.north_cpw_etch_radius, "se", (x, y), start_angle=cpw_start_angle, end_angle=0
                        )
                    )

                    cap_etch_coords.append(partial_info["int_point"])

                    spine_end_angle = partial_info["spine_angle"]
                    assert spine_end_angle >= 0.0, "DEBUG"
                    cap_etch_coords.extend(
                        self._get_rounded_corner_coords(
                            p.north_east_spine_etch_radius,
                            "nw",
                            (x, y),
                            clockwise=True,
                            start_angle=np.pi,
                            end_angle=spine_end_angle,
                        )
                    )

                elif ner_partial_valid:  # ne spine curve -> tilted line -> sw cpw curve
                    raise NotImplementedError
                    assert self.etch_right_edge_x < south_cpw_etch_right_edge_x, (
                        f"south_cpw must be connected to capacitor (exceeds {self.etch_left_edge_x} mm). Reduce offset magnitude"
                    )

                    spine_circ_center = np.array(
                        [self.etch_right_edge_x - p.north_east_spine_etch_radius, p.north_east_spine_etch_radius]
                    )
                    cpw_circ_center = np.array(
                        [north_cpw_etch_right_edge_x + p.north_cpw_etch_radius, -p.north_cpw_etch_radius]
                    )
                    partial_info = self._get_partial_corner_info(
                        cpw_circ_center, spine_circ_center, p.north_cpw_etch_radius, p.north_east_spine_etch_radius
                    )

                    spine_end_angle = partial_info["spine_angle"]
                    assert spine_end_angle >= 0.0, "DEBUG"
                    cap_etch_coords.extend(
                        self._get_rounded_corner_coords(
                            p.north_east_spine_etch_radius,
                            "ne",
                            (x, y),
                            clockwise=True,
                            start_angle=0,
                            end_angle=spine_end_angle,
                        )
                    )

                    cap_etch_coords.append(partial_info["int_point"])

                    cpw_start_angle = partial_info["cpw_angle"]
                    assert cpw_start_angle <= 0.0, "DEBUG"
                    cap_etch_coords.extend(
                        self._get_rounded_corner_coords(
                            p.north_cpw_etch_radius, "sw", (x, y), start_angle=cpw_start_angle, end_angle=-np.pi
                        )
                    )

                else:
                    raise RuntimeError("Should not reach this point. Code logic issue")

        elif p.north_cpw_length > 0.0 and (p.north_cpw_length < p.north_gap_ground):
            north_cpw_etch_left_edge_x = self.north_cpw_left_edge_x - p.north_cpw_gap

            if north_cpw_etch_left_edge_x < self.etch_width / 2:
                raise NotImplementedError

        elif p.north_cpw_length < 1e-6:
            raise NotImplementedError

        # SOUTH EAST CORNER
        if p.north_cpw_length > p.north_gap_ground:
            x = self.etch_right_edge_x
            y = 0

            ser_partial_valid = (self.etch_right_edge_x < south_cpw_etch_right_edge_x) and (
                (south_cpw_etch_right_edge_x - p.south_cpw_etch_radius)
                < (self.etch_right_edge_x + p.south_east_spine_etch_radius)
            )
            sel_partial_valid = (self.etch_right_edge_x > south_cpw_etch_right_edge_x) and (
                (south_cpw_etch_right_edge_x + p.south_cpw_etch_radius)
                > (self.etch_right_edge_x - p.south_east_spine_etch_radius)
            )
            se_equality_valid = np.abs(self.etch_right_edge_x - south_cpw_etch_right_edge_x) < 1e-6

            if not any([ser_partial_valid, sel_partial_valid, se_equality_valid]):
                # fill pi/2 rotations for both curves
                if self.etch_left_edge_x < north_cpw_etch_left_edge_x:  # cpw fully connected to spine
                    cap_etch_coords.extend(
                        self._get_rounded_corner_coords(p.south_east_spine_etch_radius, "se", (x, y), clockwise=True)
                    )
                    x = south_cpw_etch_right_edge_x
                    cap_etch_coords.extend(self._get_rounded_corner_coords(p.south_cpw_etch_radius, "nw", (x, y)))
                else:  # cpw sticking out to the left of the cap body
                    cap_etch_coords.extend(
                        self._get_rounded_corner_coords(p.south_east_spine_etch_radius, "sw", (x, y))
                    )
                    x = south_cpw_etch_right_edge_x
                    cap_etch_coords.extend(
                        self._get_rounded_corner_coords(p.south_cpw_etch_radius, "ne", (x, y), clockwise=True)
                    )

            else:
                if se_equality_valid:  # straight connection from cpw to spine
                    cap_etch_coords.append((x, y))

                elif ser_partial_valid:  # nw spine curve -> tilted line -> se cpw curve
                    raise NotImplementedError
                    spine_circ_center = np.array(
                        [
                            self.etch_left_edge_x + p.north_west_spine_etch_radius,
                            self.etch_height - p.north_west_spine_etch_radius,
                        ]
                    )
                    cpw_circ_center = np.array(
                        [
                            north_cpw_etch_left_edge_x - p.north_cpw_etch_radius,
                            self.etch_height + p.north_cpw_etch_radius,
                        ]
                    )
                    partial_info = self._get_partial_corner_info(
                        cpw_circ_center,
                        spine_circ_center,
                        p.north_cpw_etch_radius,
                        p.north_west_spine_etch_radius,
                        y_int_val=self.etch_height,
                    )

                    spine_end_angle = partial_info["spine_angle"]
                    assert spine_end_angle >= 0.0, "DEBUG"
                    cap_etch_coords.extend(
                        self._get_rounded_corner_coords(
                            p.north_west_spine_etch_radius,
                            "nw",
                            (x, y),
                            clockwise=True,
                            start_angle=np.pi,
                            end_angle=spine_end_angle,
                        )
                    )

                    cap_etch_coords.append(partial_info["int_point"])

                    x = north_cpw_etch_left_edge_x
                    cpw_start_angle = partial_info["cpw_angle"]
                    assert cpw_start_angle <= 0.0, "DEBUG"
                    cap_etch_coords.extend(
                        self._get_rounded_corner_coords(
                            p.north_cpw_etch_radius, "se", (x, y), start_angle=cpw_start_angle, end_angle=0
                        )
                    )

                elif sel_partial_valid:  # ne spine curve -> tilted line -> sw cpw curve
                    raise NotImplementedError
                    assert self.etch_left_edge_x < south_cpw_etch_right_edge_x, (
                        f"south_cpw must be connected to capacitor (exceeds {self.etch_left_edge_x} mm). Reduce offset magnitude"
                    )

                    y_int_val = self.etch_height - p.north_gap_ground

                    spine_circ_center = np.array(
                        [self.etch_left_edge_x - p.north_west_spine_etch_radius, y_int_val - p.north_west_spine_radius]
                    )
                    cpw_circ_center = np.array(
                        [north_cpw_etch_left_edge_x + p.north_cpw_etch_radius, y_int_val + p.north_cpw_radius]
                    )
                    partial_info = self._get_partial_corner_info(
                        cpw_circ_center,
                        spine_circ_center,
                        p.north_cpw_etch_radius,
                        p.north_west_spine_etch_radius,
                        y_int_val=y_int_val,
                    )

                    y = y_int_val
                    spine_end_angle = partial_info["spine_angle"]
                    assert spine_end_angle >= 0.0, "DEBUG"
                    cap_etch_coords.extend(
                        self._get_rounded_corner_coords(
                            p.north_west_spine_etch_radius, "ne", (x, y), start_angle=0, end_angle=spine_end_angle
                        )
                    )

                    cap_etch_coords.append(partial_info["int_point"])

                    x = north_cpw_etch_left_edge_x
                    cpw_start_angle = partial_info["cpw_angle"]
                    assert cpw_start_angle <= 0.0, "DEBUG"
                    cap_etch_coords.extend(
                        self._get_rounded_corner_coords(
                            p.north_cpw_etch_radius,
                            "sw",
                            (x, y),
                            start_angle=cpw_start_angle,
                            end_angle=-np.pi,
                            clockwise=True,
                        )
                    )

                else:
                    raise RuntimeError("Should not reach this point. Code logic issue")

            x = south_cpw_etch_right_edge_x
            y = -south_cpw_etch_length
            cap_etch_coords.append((x, y))

        elif p.north_cpw_length > 0.0 and (p.north_cpw_length < p.north_gap_ground):
            north_cpw_etch_left_edge_x = self.north_cpw_left_edge_x - p.north_cpw_gap

            if north_cpw_etch_left_edge_x < self.etch_width / 2:
                raise NotImplementedError

        elif p.north_cpw_length < 1e-6:
            raise NotImplementedError

        cap_etch = Polygon(cap_etch_coords)
        return cap_etch
