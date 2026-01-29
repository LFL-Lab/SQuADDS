import os
import sys

# warn using `warnings` if os is mac that this is not supported
import warnings
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import qiskit_metal as metal
from qiskit_metal import Dict
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from squadds.components.coupled_systems import QubitCavity
from squadds.simulations.objects import (
    run_qubit_cavity_sweep,
    run_sweep,
    simulate_single_design,
    simulate_whole_device,
)
from squadds.simulations.utils import find_a_fq


class AnsysSimulator:
    """
    AnsysSimulator class for simulating devices using Ansys.

    Attributes:
        analyzer: The analyzer object.
        device_dict: The device dictionary containing design and setup options.
        lom_analysis_obj: The LOM analysis object.
        epr_analysis_obj: The EPR analysis object.
        design: The design planar object.
        gui: The MetalGUI object.
    """

    def __init__(self, analyzer, design_options, **kwargs):
        """
        Initialize the AnsysSimulator object.

        Args:
            analyzer: The analyzer object.
            design_options (dict): The device dictionary (renamed to device_dict internally).
        Optional arguments:
            open_gui (bool): If True, a MetalGUI instance is created and assigned to self.gui. Default is False.
        """
        self.analyzer = analyzer
        self.device_dict = deepcopy(design_options)  # Store a copy for stateful modification

        self.lom_analysis_obj = None
        self.epr_analysis_obj = None

        self.default_eigenmode_options = {
            "setup": {
                "basis_order": 1,
                "max_delta_f": 0.02,
                "max_passes": 30,
                "min_converged": 3,
                "min_converged_passes": 3,
                "min_freq_ghz": 1,
                "min_passes": 1,
                "n_modes": 1,
                "name": "default_eigenmode_setup",
                "pct_refinement": 30,
                "reuse_selected_design": True,
                "reuse_setup": True,
                "vars": {"Cj": "0fF", "Lj": "0nH"},
            }
        }
        self.default_lom_options = {
            "setup": {
                "name": "default_LOM_setup",
                "reuse_selected_design": False,
                "reuse_setup": False,
                "freq_ghz": 5.0,
                "save_fields": False,
                "enabled": True,
                "max_passes": 30,
                "min_passes": 2,
                "min_converged_passes": 1,
                "percent_error": 0.1,
                "percent_refinement": 30,
                "auto_increase_solution_order": True,
                "solution_order": "High",
                "solver_type": "Iterative",
            }
        }

        self.design = metal.designs.design_planar.DesignPlanar()
        if kwargs.get("open_gui", False):
            self.gui = metal.MetalGUI(self.design)
        self.design.overwrite_enabled = True
        self._warnings()

        self.console = Console()
        self.executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)
        self.futures = []

        print(f"selected system: {self.analyzer.selected_system}")

    def _warnings(self):
        """
        Displays a warning message if the operating system is macOS.

        Raises:
            UserWarning: If the operating system is macOS, a warning is raised indicating that `AnsysSimulator` is not supported on MacOS.
        """
        if sys.platform == "darwin":  # Checks if the operating system is macOS
            warnings.warn(
                "`AnsysSimulator` is not supported on MacOS since Ansys does not have a Mac App. Please use Windows or Linux for simulations.",
                stacklevel=2,
            )

    def update_simulation_setup(self, **kwargs):
        """
        Updates the simulation setup parameters in the stored device dictionary.
        Smartly searches for the keys in 'setup', 'setup_qubit', or 'setup_cavity_claw'.

        Args:
            **kwargs: Key-value pairs of parameters to update (e.g., max_passes=10).
        """
        targets = []
        if "setup" in self.device_dict:
            targets.append(self.device_dict["setup"])
        if "setup_qubit" in self.device_dict:
            targets.append(self.device_dict["setup_qubit"])
        if "setup_cavity_claw" in self.device_dict:
            targets.append(self.device_dict["setup_cavity_claw"])

        updated_keys = []
        for key, value in kwargs.items():
            found = False
            for target in targets:
                # We update if the key exists OR if it's a common HFSS setup key we want to force add
                # For now, let's update if key is in target, or if target looks like a setup dict
                if key in target:
                    target[key] = value
                    found = True
                # If invalid key, maybe we should just add it? Setup dicts in Metal are flexible.
                # Let's assume user knows what they are doing and add it if not found,
                # but to which one?
                # If not found in any, we add to ALL targets (risky) or just print warning.
                # Better strategy: Update existing keys, if key doesn't exist, assume it applies to all setups?

            if not found:
                # If key not found in any existing dict, add to all (e.g. adding a new param)
                for target in targets:
                    target[key] = value
                self.console.print(f"[dim]Added new setup param '{key}' to all setup dicts.[/dim]")
            else:
                updated_keys.append(key)

        if updated_keys:
            self.console.print(f"[green]Updated simulation setup params: {updated_keys}[/green]")

    def update_design_parameters(self, **kwargs):
        """
        Updates the geometry design parameters in the stored device dictionary.
        Smartly searches for the keys in 'design_options', 'design_options_qubit', etc.

        Args:
            **kwargs: Key-value pairs of design parameters to update (e.g., cross_length="300um").
        """
        targets = []
        # Single design
        if "design_options" in self.device_dict:
            targets.append(self.device_dict["design_options"])

        # Coupled design
        for key in ["design_options_qubit", "design_options_cavity_claw"]:
            if key in self.device_dict:
                targets.append(self.device_dict[key])

        # Also check nested dictionaries (like 'cplr_opts', 'cpw_opts')
        # We need a recursive search or just checks for known sub-dicts

        def update_recursive(target_dict, key, value):
            updated = False
            if key in target_dict:
                target_dict[key] = value
                return True

            for _k, v in target_dict.items():
                if isinstance(v, dict):
                    if update_recursive(v, key, value):
                        updated = True
            return updated

        updated_keys = []
        for key, value in kwargs.items():
            found = False
            for target in targets:
                if update_recursive(target, key, value):
                    found = True

            if not found:
                self.console.print(
                    f"[yellow]Warning: parameter '{key}' not found in any design options. Ignoring.[/yellow]"
                )
            else:
                updated_keys.append(key)

        if updated_keys:
            self.console.print(f"[green]Updated design parameters: {updated_keys}[/green]")

    def get_design_screenshot(self):
        """
        Saves a screenshot of the design.

        Returns:
            None
        """
        self.gui = metal.MetalGUI(self.design)
        self.gui.rebuild()
        self.gui.autoscale()
        self.gui.screenshot()

    def sweep(self, sweep_dict, emode_setup=None, lom_setup=None):
        """
        Sweeps the device based on the provided sweep dictionary.

        Returns:
            pandas.DataFrame: The sweep results.

        Raises:
            None
        """
        if emode_setup is None:
            emode_setup = self.default_eigenmode_options
        if lom_setup is None:
            lom_setup = self.default_lom_options

        # run_sweep(self.design, sweep_dict, emode_setup, lom_setup)
        # print(sweep_dict)
        if "coupler_type" in sweep_dict and sweep_dict["coupler_type"].lower() == "ncap":
            run_sweep(self.design, sweep_dict, emode_setup, lom_setup, filename="ncap_sweep")
        elif "coupler_type" in sweep_dict and sweep_dict["coupler_type"].lower() == "clt":
            run_sweep(self.design, sweep_dict, emode_setup, lom_setup, filename="clt_sweep")
        else:
            run_sweep(self.design, sweep_dict, emode_setup, lom_setup, filename="xmon_sweep")

    def simulate(self, device_dict=None, run_async=False):
        """
        Simulates the device based on the provided device dictionary.

        Args:
            device_dict (dict, optional): A dictionary containing the device design options and setup.
                                          If None, uses the internally stored device_dict.
            run_async (bool): If True, runs the simulation asynchronously.

        Returns:
            pandas.DataFrame or concurrent.futures.Future: The simulation results or a Future object.
        """
        # Use stored device_dict if none provided
        if device_dict is None:
            device_dict = self.device_dict

        # ... logic continues ...
        """
        Simulates the device based on the provided device dictionary.

        Args:
            device_dict (dict): A dictionary containing the device design options and setup.
            run_async (bool): If True, runs the simulation asynchronously using a ThreadPoolExecutor.
                              Returns the Future object. Default is False.

        Returns:
            pandas.DataFrame or concurrent.futures.Future: The simulation results or a Future object.

        Raises:
            None
        """
        if run_async:
            self.console.print(
                f"[bold cyan]Submitting async simulation for system: {self.analyzer.selected_system}[/bold cyan]"
            )
            future = self.executor.submit(self._run_simulation, device_dict)
            future.add_done_callback(self._simulation_callback)
            self.futures.append(future)
            return future
        else:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
            ) as progress:
                task = progress.add_task(
                    f"[green]Running sync simulation for {self.analyzer.selected_system}...", total=None
                )
                result = self._run_simulation(device_dict)
                progress.update(task, completed=True)
                self.console.print("[bold green]Simulation Completed![/bold green]")
                return result

    def _run_simulation(self, device_dict):
        """Helper method to run the actual simulation logic."""
        return_df = {}
        try:
            # Print simulation parameters for verification
            self.console.print("[bold blue]Simulation Parameters:[/bold blue]")
            if "setup" in device_dict:
                self.console.print(f"  [cyan]Setup:[/cyan] {device_dict['setup']}")
            if "setup_qubit" in device_dict:
                self.console.print(f"  [cyan]Qubit Setup:[/cyan] {device_dict['setup_qubit']}")
            if "setup_cavity_claw" in device_dict:
                self.console.print(f"  [cyan]Cavity Setup:[/cyan] {device_dict['setup_cavity_claw']}")

            if isinstance(self.analyzer.selected_system, list):  # have a qubit_cavity object
                self.geom_dict = Dict(
                    qubit_geoms=device_dict["design_options_qubit"],
                    cavity_geoms=device_dict["design_options_cavity_claw"],
                )
                self.setup_dict = Dict(
                    qubit_setup=device_dict["setup_qubit"], cavity_setup=device_dict["setup_cavity_claw"]
                )
                return_df, self.lom_analysis_obj, self.epr_analysis_obj = simulate_whole_device(
                    design=self.design,
                    device_dict=device_dict,
                    LOM_options=self.setup_dict.qubit_setup,
                    eigenmode_options=self.setup_dict.cavity_setup,
                )

            else:  # have a non-qubit_cavity object
                self.geom_dict = device_dict["design_options"]
                self.setup_dict = device_dict["setup"]
                return_df, self.lom_analysis_obj, self.epr_analysis_obj = simulate_single_design(
                    design=self.design, device_dict=device_dict, lom_options=self.setup_dict
                )
        except Exception as e:
            self.console.print(f"[bold red]Error during simulation: {e}[/bold red]")
            # Re-raise to be caught by future
            raise e

        return return_df

    def _simulation_callback(self, future):
        """Callback for async simulation completion."""
        try:
            future.result()
            self.console.print("[bold green]Async Simulation Completed successfully![/bold green]")
        except Exception as e:
            self.console.print(f"[bold red]Async Simulation Failed with error: {e}[/bold red]")

    def wait_for_all(self):
        """Waits for all submitted async simulations to complete."""
        from concurrent.futures import wait

        if self.futures:
            self.console.print(f"[yellow]Waiting for {len(self.futures)} simulations to complete...[/yellow]")
            wait(self.futures)
            self.console.print("[bold green]All simulations finished.[/bold green]")
            self.futures = []

    def get_renderer_screenshot(self):
        """
        Saves a screenshot of the renderer.

        If the EPR analysis object is not None, it saves a screenshot of the EPR analysis simulation.
        If the LOM analysis object is not None, it saves a screenshot of the LOM analysis simulation.
        """
        if self.epr_analysis_obj is not None:
            self.epr_analysis_obj.sim.save_screenshot()
        if self.lom_analysis_obj is not None:
            self.lom_analysis_obj.sim.save_screenshot()

    def get_xmon_info(self, xmon_dict):
        """
        Retrieves information about the Xmon qubit from the given xmon_dict.

        Parameters:
            xmon_dict (dict): A dictionary containing simulation results and design options.

        Returns:
            dict: A dictionary containing the qubit frequency in GHz and the anharmonicity in MHz.
        """
        # data = xmon_dict["sim_results"]
        cross2cpw = abs(xmon_dict["sim_results"]["cross_to_claw"]) * 1e-15
        cross2ground = abs(xmon_dict["sim_results"]["cross_to_ground"]) * 1e-15
        Lj = xmon_dict["design"]["design_options"]["aedt_q3d_inductance"] * (
            1 if xmon_dict["design"]["design_options"]["aedt_q3d_inductance"] > 1e-9 else 1e-9
        )
        a, fq = find_a_fq(cross2cpw, cross2ground, Lj)
        print(f"qubit anharmonicity = {round(a)} MHz \nqubit frequency = {round(fq, 3)} GHz")
        # return a json object
        return {"qubit_frequency_GHz": fq, "anharmonicity_MHz": a}

    def plot_device(self, device_dict):
        """
        Plot the device based on the given device dictionary.

        Parameters:
        - device_dict (dict): A dictionary containing the device information.

        Returns:
        None
        """
        self.gui = metal.MetalGUI(self.design)
        self.design.delete_all_components()
        if "g" in device_dict["sim_results"]:
            QubitCavity(self.design, "qubit_cavity", options=device_dict["design"]["design_options"])

        self.gui.rebuild()
        self.gui.autoscale()
        self.gui.screenshot()

    def sweep_qubit_cavity(self, device_dict, emode_setup=None, lom_setup=None):
        """
        Sweeps a single geometric parameter of the qubit and cavity system based on the provided sweep dictionary.

        Args:
            device_dict (dict): A dictionary containing the device design options and setup.
            emode_setup (dict): A dictionary containing the eigenmode setup options.
            lom_setup (dict): A dictionary containing the LOM setup options.

        Returns:
            results: The sweep results.
        """

        if emode_setup is None:
            emode_setup = self.default_eigenmode_options
        if lom_setup is None:
            lom_setup = self.default_lom_options

        run_qubit_cavity_sweep(self.design, device_dict, emode_setup, lom_setup, filename="qubit_cavity_sweep")
