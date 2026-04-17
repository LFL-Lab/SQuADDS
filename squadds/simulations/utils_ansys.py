from ansys.aedt.core import Hfss

from squadds.simulations.utils_geometry import calculate_center_and_dimensions


def getMeshScreenshot(projectname, designname, solutiontype="Eigenmode"):
    raise NotImplementedError()


def setMaterialProperties(projectname, designname, solutiontype="Eigenmode"):
    aedt = Hfss(
        projectname=projectname,
        designname=designname,
        solution_type=solutiontype,
        new_desktop_session=False,
        close_on_exit=False,
    )
    ultra_cold_silicon(aedt)
    delete_old_setups(aedt)
    aedt.release_desktop(close_projects=False, close_desktop=False)


def ultra_cold_silicon(aedt):
    materials = aedt.materials
    silicon = materials.checkifmaterialexists("silicon")
    silicon.permittivity = 11.45
    silicon.dielectric_loss_tangent = 1e-7


def delete_old_setups(aedt):
    if len(aedt.setups) != 0:
        aedt.setups[0].delete()


def get_freq(epra, test_hfss, generate_plots=False):
    project_name = test_hfss.pinfo.project_name
    design_name = test_hfss.pinfo.design_name
    setMaterialProperties(project_name, design_name, solutiontype="Eigenmode")
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
    setMaterialProperties(project_name, design_name, solutiontype="Eigenmode")
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
