import os
import sys

# warn using `warnings` if os is mac that this is not supported
import warnings
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import qiskit_metal as metal
from qiskit_metal import Dict
from rich.console import Console

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

        self.console = Console(force_jupyter=False)
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

    def update_simulation_setup(self, target="all", **kwargs):
        """
        Updates the simulation setup parameters in the stored device dictionary.
        Intelligently maps targets to setup keys based on system type.

        Args:
            target (str): Which setup to update. Options: "all", "qubit", "cavity", "coupler", "generic".
            **kwargs: Key-value pairs of parameters to update (e.g., max_passes=10).
        """
        # Get the appropriate setup keys based on target and system type
        target_keys = self._get_setup_targets(target)

        if not target_keys:
            self.console.print(f"[yellow]Warning: target '{target}' is not valid for this system.[/yellow]")
            return

        updated_keys = []
        unknown_params = {}

        for key in target_keys:
            if key in self.device_dict:
                # Check which parameters are unknown
                for param, value in kwargs.items():
                    if param not in self.device_dict[key]:
                        if key not in unknown_params:
                            unknown_params[key] = []
                        unknown_params[key].append((param, value))

        # If there are unknown parameters, ask for confirmation
        if unknown_params:
            self.console.print("[yellow]⚠️  Unknown parameters detected:[/yellow]")
            for setup_key, params in unknown_params.items():
                for param, value in params:
                    self.console.print(f"  • [cyan]{param}[/cyan] = {value} (not in [bold]{setup_key}[/bold])")

            response = input("\n[?] Would you like to add these new parameters? (y/n): ").strip().lower()
            if response != "y":
                self.console.print("[dim]Skipping unknown parameters. Only updating existing ones.[/dim]")
                # Filter out unknown parameters
                kwargs = {
                    k: v
                    for k, v in kwargs.items()
                    if not any(k in [p[0] for p in params] for params in unknown_params.values())
                }

        # Update the parameters
        for key in target_keys:
            if key in self.device_dict:
                self.device_dict[key].update(kwargs)
                updated_keys.append(key)

        if updated_keys:
            self.console.print(f"[green]✓ Updated {', '.join(updated_keys)}: {list(kwargs.keys())}[/green]")

    def _get_setup_targets(self, target):
        """
        Returns list of dict keys to update based on target and system type.

        Args:
            target (str): The target setup to update.

        Returns:
            list: List of setup dictionary keys to update.
        """
        if isinstance(self.analyzer.selected_system, list):
            # Coupled system (qubit + cavity)
            mapping = {
                "qubit": ["setup_qubit"],
                "cavity_claw": ["setup_cavity_claw"],
                "coupler": ["setup_coupler"],
                "all": ["setup_qubit", "setup_cavity_claw", "setup_coupler"],
            }
        else:
            # Single system
            mapping = {"generic": ["setup"], "all": ["setup"]}

        # Return the keys, filtering out any that don't exist in device_dict
        keys = mapping.get(target, [])
        return [k for k in keys if k in self.device_dict]

    def get_simulation_setup(self, target="all"):
        """
        Displays the current simulation setup parameters.

        Args:
            target (str): Which setup to display. Options: "all", "qubit", "cavity", "coupler", "generic".

        Returns:
            dict: Dictionary containing the requested setup(s).
        """
        from rich.table import Table

        target_keys = self._get_setup_targets(target)

        if not target_keys:
            self.console.print(f"[yellow]No setup found for target '{target}'[/yellow]")
            return {}

        result = {}
        for key in target_keys:
            if key in self.device_dict:
                result[key] = self.device_dict[key]

                # Create a nice table for display
                table = Table(title=f"[bold cyan]{key}[/bold cyan]", show_header=True)
                table.add_column("Parameter", style="cyan")
                table.add_column("Value", style="green")

                for param, value in self.device_dict[key].items():
                    table.add_row(str(param), str(value))

                self.console.print(table)

        return result

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
            raise NotImplementedError(
                "Ansys-native parallel simulation (run_async=True) is not yet implemented. "
                "This feature will render all designs to the same Ansys project and leverage "
                "Ansys's built-in scheduling for parallel execution. "
                "Please use run_async=False (default) for sequential simulation. "
                "Track progress at: https://github.com/shanto268/SQuADDS/issues"
            )

        if device_dict is None:
            device_dict = self.device_dict

        # Use a simple print instead of Progress to avoid recursion issues in some Jupyter environments
        # when nested prints occur (e.g. from pyEPR)
        self.console.rule("[bold cyan]Starting Simulation[/bold cyan]")
        try:
            result = self._run_simulation(device_dict)
            self.console.print("[bold green]Simulation Completed Successfully![/bold green]")
            return result
        except Exception as e:
            # Avoid using self.console.print here if it might cause further recursion
            print(f"Error during simulation: {e}")
            raise e

    def _run_simulation(self, device_dict):
        """Helper method to run the actual simulation logic."""
        return_df = {}
        try:
            # Print simulation plan
            self.console.print("\n[bold blue]═══ Simulation Plan ═══[/bold blue]")

            if isinstance(self.analyzer.selected_system, list):
                # Coupled system
                sim_count = 2  # Eigenmode + Qubit LOM
                self.console.print("[cyan]System Type:[/cyan] Coupled (Qubit + Cavity)")
                self.console.print("[cyan]Simulation Types:[/cyan]")
                self.console.print("  1. [green]Eigenmode[/green] on Cavity")
                self.console.print("  2. [green]LOM (Capacitance)[/green] on Qubit")

                # Check for half-wave coupler
                if "setup_coupler" in device_dict:
                    sim_count = 3
                    self.console.print("  3. [green]LOM (Capacitance)[/green] on Coupler (NCap)")
                    self.console.print(f"[cyan]Total Simulations:[/cyan] {sim_count} (Half-Wave Resonator)")
                else:
                    self.console.print(f"[cyan]Total Simulations:[/cyan] {sim_count}")
            else:
                # Single component
                self.console.print("[cyan]System Type:[/cyan] Single Component")
                self.console.print("[cyan]Total Simulations:[/cyan] 1")
                self.console.print("[cyan]Simulation Types:[/cyan]")
                self.console.print("  1. [green]LOM (Capacitance)[/green]")

            self.console.print("[bold blue]═══════════════════════[/bold blue]\n")

            # Print simulation parameters for verification
            self.console.print("[bold blue]Simulation Parameters:[/bold blue]")
            if "setup" in device_dict:
                self.console.print(f"  [cyan]Setup:[/cyan] {device_dict['setup']}")
            if "setup_qubit" in device_dict:
                self.console.print(f"  [cyan]Qubit Setup:[/cyan] {device_dict['setup_qubit']}")
            if "setup_cavity_claw" in device_dict:
                self.console.print(f"  [cyan]Cavity Setup:[/cyan] {device_dict['setup_cavity_claw']}")
            if "setup_coupler" in device_dict:
                self.console.print(f"  [cyan]Coupler Setup:[/cyan] {device_dict['setup_coupler']}")

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
                    generate_plots=False,
                )

            else:  # have a non-qubit_cavity object
                self.geom_dict = device_dict["design_options"]
                self.setup_dict = device_dict["setup"]
                return_df, self.lom_analysis_obj, self.epr_analysis_obj = simulate_single_design(
                    design=self.design, device_dict=device_dict, lom_options=self.setup_dict, generate_plots=False
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
