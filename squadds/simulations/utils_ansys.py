from __future__ import annotations

from typing import Any

from ansys.aedt.core import Hfss

from squadds.simulations.utils_geometry import calculate_center_and_dimensions


def getMeshScreenshot(projectname, designname, solutiontype="Eigenmode"):
    raise NotImplementedError()


def _cryo_silicon_args(permittivity: float, loss_tangent: float) -> list[Any]:
    return [
        "NAME:silicon",
        "CoordinateSystemType:=",
        "Cartesian",
        "BulkOrSurfaceType:=",
        1,
        [
            "NAME:PhysicsTypes",
            "set:=",
            ["Electromagnetic", "Thermal", "Structural"],
        ],
        [
            "NAME:AttachedData",
            [
                "NAME:MatAppearanceData",
                "property_data:=",
                "appearance_data",
                "Red:=",
                210,
                "Green:=",
                105,
                "Blue:=",
                30,
                "Transparency:=",
                0,
            ],
        ],
        "permittivity:=",
        str(permittivity),
        "permeability:=",
        "1.0",
        "conductivity:=",
        "0",
        "dielectric_loss_tangent:=",
        str(loss_tangent),
        "magnetic_loss_tangent:=",
        "0",
        "thermal_conductivity:=",
        "0.01",
        "mass_density:=",
        "0",
        "specific_heat:=",
        "0",
        "thermal_expansion_coefficient:=",
        "0",
        "youngs_modulus:=",
        "0",
        "poissons_ratio:=",
        "0",
        "diffusivity:=",
        "0.8",
        "molecular_mass:=",
        "0",
        "viscosity:=",
        "0",
    ]


class _LiveMaterialApp:
    def __init__(self, source_renderer: Any):
        live_pinfo = getattr(source_renderer, "pinfo", None)
        live_project = getattr(live_pinfo, "project", None)
        live_design = getattr(live_pinfo, "design", None)
        live_desktop = getattr(live_pinfo, "desktop", None)

        self._oproject = getattr(live_project, "_project", None)
        self._odesign = getattr(live_design, "_design", None)
        self._desktop = getattr(live_desktop, "_desktop", None)
        if self._desktop is None:
            self._desktop = getattr(getattr(live_project, "parent", None), "_desktop", None)
        self.design_type = "HFSS"

    @property
    def odesktop(self):
        return self._desktop

    @property
    def oproject(self):
        return self._oproject

    @property
    def odesign(self):
        return self._odesign

    @property
    def odefinition_manager(self):
        if self._oproject is None:
            return None
        return self._oproject.GetDefinitionManager()


def _resolve_material(materials: Any, material_name: str):
    candidate = None
    if hasattr(materials, "checkifmaterialexists"):
        try:
            candidate = materials.checkifmaterialexists(material_name)
        except Exception:
            candidate = None
        if hasattr(candidate, "permittivity") and hasattr(candidate, "dielectric_loss_tangent"):
            return candidate

    if hasattr(materials, "exists_material"):
        try:
            candidate = materials.exists_material(material_name)
        except Exception:
            candidate = None
        if hasattr(candidate, "permittivity") and hasattr(candidate, "dielectric_loss_tangent"):
            return candidate

    return None


def _apply_live_material_properties(
    renderer: Any,
    *,
    permittivity: float,
    loss_tangent: float,
    materials_factory: Any | None = None,
) -> bool:
    live_app = _LiveMaterialApp(renderer)
    definition_manager = live_app.odefinition_manager
    if definition_manager is not None and hasattr(definition_manager, "EditMaterial"):
        definition_manager.EditMaterial("silicon", _cryo_silicon_args(permittivity, loss_tangent))
        return True

    if materials_factory is None:
        try:
            from ansys.aedt.core.modules.material_lib import Materials
        except Exception:
            Materials = None
        materials_factory = Materials

    if materials_factory is None or (live_app.oproject is None and live_app.odesign is None):
        return False

    try:
        live_materials = materials_factory(live_app)
        silicon = _resolve_material(live_materials, "silicon")
        if not silicon:
            return False
        silicon.permittivity = permittivity
        silicon.dielectric_loss_tangent = loss_tangent
        return True
    except Exception:
        return False


def setMaterialProperties(
    projectname=None,
    designname=None,
    solutiontype="Eigenmode",
    *,
    renderer: Any | None = None,
    permittivity: float = 11.45,
    loss_tangent: float = 1e-7,
    hfss_factory: Any | None = None,
    materials_factory: Any | None = None,
):
    if renderer is not None:
        pinfo = getattr(renderer, "pinfo", None)
        if projectname is None:
            projectname = getattr(pinfo, "project_name", None)
        if designname is None:
            designname = getattr(pinfo, "design_name", None)
        if _apply_live_material_properties(
            renderer,
            permittivity=permittivity,
            loss_tangent=loss_tangent,
            materials_factory=materials_factory,
        ):
            return

    if projectname is None or designname is None:
        raise ValueError("Project and design names are required when no live renderer session is available.")

    if hfss_factory is None:
        hfss_factory = Hfss

    aedt = hfss_factory(
        projectname=projectname,
        designname=designname,
        solution_type=solutiontype,
        new_desktop_session=False,
        close_on_exit=False,
    )
    try:
        ultra_cold_silicon(aedt, permittivity=permittivity, loss_tangent=loss_tangent)
        delete_old_setups(aedt)
    finally:
        if hasattr(aedt, "release_desktop"):
            aedt.release_desktop(close_projects=False, close_desktop=False)


def ultra_cold_silicon(aedt, *, permittivity: float = 11.45, loss_tangent: float = 1e-7):
    materials = aedt.materials
    silicon = _resolve_material(materials, "silicon")
    if not silicon:
        raise AttributeError("Could not resolve the silicon material on the active AEDT session.")
    silicon.permittivity = permittivity
    silicon.dielectric_loss_tangent = loss_tangent


def delete_old_setups(aedt):
    setups = getattr(aedt, "setups", [])
    if setups:
        setups[0].delete()


def get_freq(epra, test_hfss, generate_plots=False):
    project_name = test_hfss.pinfo.project_name
    design_name = test_hfss.pinfo.design_name
    setMaterialProperties(project_name, design_name, solutiontype="Eigenmode", renderer=test_hfss)
    epra.sim._analyze()
    if generate_plots:
        try:
            epra.sim.plot_convergences()
            epra.sim.save_screenshot()
            epra.sim.plot_fields("main")
            epra.sim.save_screenshot()
        except Exception as exc:
            print(f"couldn't generate plots. Error: {exc}")
    f = epra.get_frequencies()
    freq = f.values[0][0] * 1e9
    print(f"freq = {round(freq / 1e9, 3)} GHz")
    return freq


def get_freq_Q_kappa(epra, test_hfss, generate_plots=False):
    project_name = test_hfss.pinfo.project_name
    design_name = test_hfss.pinfo.design_name
    setMaterialProperties(project_name, design_name, solutiontype="Eigenmode", renderer=test_hfss)
    epra.sim._analyze()
    if generate_plots:
        try:
            epra.sim.plot_convergences()
            epra.sim.save_screenshot()
            epra.sim.plot_fields("main")
            epra.sim.save_screenshot()
        except Exception:
            print("couldn't generate plots.")
    f = epra.get_frequencies()
    freq = f.values[0][0] * 1e9
    Q = f.values[0][1]
    kappa = freq / Q
    print(f"freq = {round(freq / 1e9, 3)} GHz")
    print(f"Q = {round(Q, 1)}")
    print(f"kappa = {round(kappa / 1e6, 3)} MHz")
    return freq, Q, kappa


def mesh_objects(modeler, mesh_lengths):
    for mesh_name, mesh_info in mesh_lengths.items():
        modeler.mesh_length(mesh_name, mesh_info["objects"], MaxLength=mesh_info["MaxLength"])


def add_ground_strip_and_mesh(modeler, coupler, mesh_lengths):
    bounds = coupler.qgeometry_bounds()
    bbox = {"min_x": bounds[0], "max_x": bounds[2], "min_y": bounds[1], "max_y": bounds[3]}
    center, dimensions = calculate_center_and_dimensions(bbox)
    modeler.draw_rect_center(
        [coord * 1e-3 for coord in center],
        x_size=dimensions[0] * 1e-3,
        y_size=dimensions[1] * 1e-3,
        name="ground_strip",
    )
    modeler.intersect(["ground_strip", "ground_main_plane"], True)
    modeler.subtract("ground_main_plane", ["ground_strip"], True)
    modeler.assign_perfect_E(["ground_strip"])
    mesh_lengths.update({"mesh_ground_strip": {"objects": ["ground_strip"], "MaxLength": "4um"}})

    for mesh_name, mesh_info in mesh_lengths.items():
        modeler.mesh_length(mesh_name, mesh_info["objects"], MaxLength=mesh_info["MaxLength"])
