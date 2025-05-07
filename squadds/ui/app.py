import json
import os

import numpy as np
import plotly.express as px
import streamlit as st

from squadds import Analyzer, SQuADDS_DB
from squadds.interpolations.physics import ScalingInterpolator
from squadds.ui.utils_query import (extract_cavity_only_params,
                                    find_closest_cached)

# SQuADDS logo and links
LOGO_PATH = "https://raw.githubusercontent.com/LFL-Lab/SQuADDS/master/docs/_static/images/squadds_logo_transparent.png"
HF_LINK = "https://huggingface.co/datasets/SQuADDS/SQuADDS_DB"
GITHUB_LINK = "https://github.com/LFL-Lab/SQuADDS"
DOCSITE_LINK = "https://lfl-lab.github.io/SQuADDS/"
PAPER_LINK = "https://quantum-journal.org/papers/q-2024-09-09-1465/"
DEEPWIKI_LINK = "https://deepwiki.com/LFL-Lab/SQuADDS/1-overview"
PORTAL_LINK = "https://squadds-portal.vercel.app"


def convert_numpy_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_list(v) for v in obj]
    else:
        return obj

def initialize_session_state():
    """Initialize session state variables"""
    if 'db' not in st.session_state:
        st.session_state.db = SQuADDS_DB()
    if 'supported_qubits' not in st.session_state:
        qubits = st.session_state.db.get_component_names("qubit")
        st.session_state.supported_qubits = qubits
    if 'supported_cavities' not in st.session_state:
        cavities = st.session_state.db.get_component_names("cavity_claw")
        st.session_state.supported_cavities = cavities

def main():
    st.set_page_config(
        page_title="SQuADDS WebUI",
        page_icon="🔍",
        layout="wide"
    )
    
    initialize_session_state()
    db = st.session_state.db
    
    # SQuADDS logo and title
    st.markdown(f"""
    <div style='display: flex; align-items: center; gap: 2rem;'>
        <img src='{LOGO_PATH}' alt='SQuADDS Logo' style='height: 80px;'>
        <h1 style='margin-bottom: 0;'>SQuADDS WebUI</h1>
    </div>
    """, unsafe_allow_html=True)
    # Links row
    st.markdown(f"""
    <div style='margin-bottom: 1.5rem;'>
        <a href='{HF_LINK}' target='_blank' style='margin-right: 1.5rem;'>🤗 HuggingFace</a>
        <a href='{GITHUB_LINK}' target='_blank' style='margin-right: 1.5rem;'>🐙 GitHub</a>
        <a href='{DOCSITE_LINK}' target='_blank' style='margin-right: 1.5rem;'>📖 Docsite</a>
        <a href='{PAPER_LINK}' target='_blank' style='margin-right: 1.5rem;'>📄 Paper</a>
        <a href='{DEEPWIKI_LINK}' target='_blank' style='margin-right: 1.5rem;'>🧠 DeepWiki</a>
        <a href='{PORTAL_LINK}' target='_blank'>🌐 Portal</a>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    Find the closest superconducting device designs based on your target parameters.
    """)
    
    # Sidebar for system selection
    with st.sidebar:
        st.header("System Configuration")
        # Track previous selections
        prev_system_type = st.session_state.get('prev_system_type', None)
        prev_qubit_type = st.session_state.get('prev_qubit_type', None)
        prev_cavity_type = st.session_state.get('prev_cavity_type', None)
        prev_coupling_type = st.session_state.get('prev_coupling_type', None)
        prev_resonator_type = st.session_state.get('prev_resonator_type', None)

        system_type = st.selectbox(
            "Select System Type",
            ["Qubit-Cavity", "Qubit Only", "Cavity Only"],
            index=0
        )
        # Qubit selection
        if system_type in ["Qubit-Cavity", "Qubit Only"]:
            qubit_type = st.selectbox(
                "Select Qubit Type",
                st.session_state.supported_qubits,
                index=st.session_state.supported_qubits.index("TransmonCross") if "TransmonCross" in st.session_state.supported_qubits else 0
            )
            if qubit_type == "CLT":
                st.warning("⚠️ 'CLT' is not a valid qubit type for design search. Please select another qubit type.")
        else:
            qubit_type = None
        # Cavity selection
        if system_type in ["Qubit-Cavity", "Cavity Only"]:
            cavity_type = st.selectbox(
                "Select Cavity Type",
                st.session_state.supported_cavities,
                index=st.session_state.supported_cavities.index("RouteMeander") if "RouteMeander" in st.session_state.supported_cavities else 0
            )
            if cavity_type == "CLT":
                st.warning("⚠️ 'CLT' is not a valid cavity type for design search. Please select another cavity type.")
        else:
            cavity_type = None
        # Coupling type for Qubit-Cavity
        show_num_cpu = False
        num_cpu = "auto"
        if system_type == "Qubit-Cavity":
            coupling_type = st.selectbox(
                "Select Coupling Type",
                ["Capacitive"],
                index=0
            )
            resonator_type = st.selectbox(
                "Select Resonator Type",
                ["quarter", "half"],
                index=0
            )
            if resonator_type == "half":
                show_num_cpu = True
        elif system_type == "Cavity Only":
            coupling_type = None
            resonator_type = st.selectbox(
                "Select Resonator Type",
                ["quarter", "half"],
                index=0
            )
        else:
            coupling_type = None
            resonator_type = None
        # Dynamic num_cpu for Qubit-Cavity + half
        if show_num_cpu:
            cpu_options = ["auto"] + [str(i) for i in range(1, os.cpu_count() + 1)]
            num_cpu = st.selectbox("Number of CPUs", cpu_options, index=0)
        # Reset results if any major config changes
        if (
            prev_system_type != system_type or
            prev_qubit_type != qubit_type or
            prev_cavity_type != cavity_type or
            prev_coupling_type != coupling_type or
            prev_resonator_type != resonator_type
        ):
            for k in ["results", "params", "analyzer", "data_qubit", "data_cpw", "data_coupler", "LJs"]:
                if k in st.session_state:
                    del st.session_state[k]
        # Update previous selections
        st.session_state.prev_system_type = system_type
        st.session_state.prev_qubit_type = qubit_type
        st.session_state.prev_cavity_type = cavity_type
        st.session_state.prev_coupling_type = coupling_type
        st.session_state.prev_resonator_type = resonator_type
        
        st.divider()
        st.markdown("### Search Settings")
        num_results = st.slider("Number of results", 1, 10, 1)  # Default to 1
        
    # Main content area
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.subheader("Target Hamiltonian Parameters")
        
        # Parameter input fields based on system type
        params = {}
        
        if system_type in ["Qubit-Cavity", "Qubit Only"]:
            params["qubit_frequency_GHz"] = st.number_input("Qubit Frequency (GHz)", 3.0, 8.0, 4.0)
            params["anharmonicity_MHz"] = st.number_input("Anharmonicity (MHz)", -500.0, -50.0, -200.0)
            
        if system_type in ["Qubit-Cavity", "Cavity Only"]:
            params["cavity_frequency_GHz"] = st.number_input("Cavity Frequency (GHz)", 5.0, 12.0, 9.2)
            params["kappa_kHz"] = st.number_input("Kappa (kHz)", 10.0, 1000.0, 80.0)
            
        if system_type == "Qubit-Cavity":
            params["g_MHz"] = st.number_input("Coupling Strength g (MHz)", 10.0, 200.0, 70.0)
            params["resonator_type"] = resonator_type
        elif system_type == "Cavity Only":
            params["resonator_type"] = resonator_type
        # No resonator_type for Qubit Only
        
        if st.button("Find Designs", type="primary"):
            # Show message for half-wave Qubit-Cavity search
            if system_type == "Qubit-Cavity" and resonator_type == "half":
                st.info("Half-wave Qubit-Cavity searches may take a while due to the size of the dataset and parallel processing. If you'd like to help speed this up, please consider contributing code! 🙏🏽 [Contribute on GitHub](https://github.com/LFL-Lab/SQuADDS)")
            # Block Cavity Only + half-wave and show info message
            if system_type == "Cavity Only" and resonator_type == "half":
                st.info("Half-wave Cavity Only search is not yet supported. We're working on this feature!")
                return
            # Track last-used system config for smart skip logic
            last_config = st.session_state.get('last_system_config', None)
            current_config = {
                'system_type': system_type,
                'qubit_type': qubit_type,
                'cavity_type': cavity_type,
                'resonator_type': resonator_type,
                'num_cpu': num_cpu,
                'num_results': num_results
            }
            skip_df_gen = False
            if last_config is not None:
                # If only params changed, skip df gen
                config_keys = ['system_type', 'qubit_type', 'cavity_type', 'resonator_type', 'num_cpu', 'num_results']
                if all(current_config[k] == last_config[k] for k in config_keys):
                    skip_df_gen = True
            with st.spinner("Searching for closest designs..."):
                try:
                    results, analyzer = find_closest_cached(system_type, qubit_type, cavity_type, resonator_type, params, num_results, num_cpu, skip_df_gen)
                    st.session_state.results = results
                    st.session_state.params = params
                    st.session_state.analyzer = analyzer
                    st.session_state.last_system_config = current_config
                    if system_type == "Qubit-Cavity":
                        st.session_state.data_qubit = convert_numpy_to_list(analyzer.get_qubit_options(results))
                        st.session_state.data_cpw = convert_numpy_to_list(analyzer.get_cpw_options(results))
                        st.session_state.data_coupler = convert_numpy_to_list(analyzer.get_coupler_options(results))
                        st.session_state.LJs = convert_numpy_to_list(analyzer.get_Ljs(results))
                        # --- Interpolated Results for Qubit-Cavity ---
                        try:
                            interpolator = ScalingInterpolator(analyzer, params)
                            interpolated_df = interpolator.get_design()
                            st.session_state.interpolated_df = interpolated_df
                        except Exception as e:
                            st.session_state.interpolated_df = None
                            st.session_state.interpolated_error = str(e)
                    elif system_type == "Qubit Only":
                        st.session_state.data_qubit = convert_numpy_to_list(analyzer.get_qubit_options(results))
                        st.session_state.data_cpw = None
                        st.session_state.data_coupler = None
                        st.session_state.LJs = None
                        st.session_state.interpolated_df = None
                    elif system_type == "Cavity Only":
                        # Use utility function for extraction
                        cpw_vals, coupler_vals = extract_cavity_only_params(results)
                        st.session_state.data_qubit = None
                        st.session_state.data_cpw = cpw_vals
                        st.session_state.data_coupler = coupler_vals
                        st.session_state.LJs = None
                        st.session_state.interpolated_df = None
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.error("Please check your system configuration and parameters.")
    
    with col2:
        if 'results' in st.session_state:
            st.subheader("Design Space Results")
            
            # Create tabs for different visualizations
            tab_labels = ["Design Parameters", "ℋ Parameter Space Plots", "`design` Options"]
            if system_type == "Qubit-Cavity":
                tab_labels.append("Interpolated Results")
            tabs = st.tabs(tab_labels)
            tab1, tab2, tab3 = tabs[:3]
            if system_type == "Qubit-Cavity":
                tab4 = tabs[3]
            
            with tab1:
                # Design Parameters Tab (was previously Extracted Data)
                analyzer = st.session_state.analyzer
                results = st.session_state.results
                if system_type == "Qubit Only":
                    keys_cross = ["cross_gap", "cross_length", "cross_width"]
                    keys_claw = ["claw_gap", "claw_length", "claw_width", "ground_spacing"]
                    cross_vals = {k: [] for k in keys_cross}
                    claw_vals = {k: [] for k in keys_claw}
                    for idx, row in results.iterrows():
                        opts = convert_numpy_to_list(row['design_options'])
                        # Cross parameters
                        for k in keys_cross:
                            v = opts.get(k)
                            if v is not None:
                                cross_vals[k].append(v)
                        # Claw parameters (nested in connection_pads.readout)
                        claw = opts.get('connection_pads', {}).get('readout', {})
                        for k in keys_claw:
                            v = claw.get(k)
                            if v is not None:
                                claw_vals[k].append(v)
                    LJs = convert_numpy_to_list(analyzer.get_Ljs(results))
                    st.markdown("**TransmonCross (Qubit) Values**")
                    st.code(json.dumps(cross_vals, indent=2))
                    st.markdown("**Claw Values**")
                    st.code(json.dumps(claw_vals, indent=2))
                    st.markdown("**Josephson Inductances (LJs, nH)**")
                    st.code(json.dumps(LJs, indent=2))
                elif system_type == "Cavity Only":
                    # Use extracted values from session state
                    cpw_vals = st.session_state.data_cpw
                    coupler_vals = st.session_state.data_coupler
                    st.markdown("**RouteMeander (CPW) Values**")
                    st.code(json.dumps(cpw_vals, indent=2))
                    st.markdown("**CoupledLineTee (Coupler) Values**")
                    st.code(json.dumps(coupler_vals, indent=2))
                else:
                    # Qubit-Cavity: show with QComponent names and 'Values'
                    if st.session_state.data_qubit is not None:
                        st.markdown("**TransmonCross (Qubit) Values**")
                        st.code(json.dumps(st.session_state.data_qubit, indent=2))
                    if st.session_state.data_cpw is not None:
                        st.markdown("**RouteMeander (CPW) Values**")
                        st.code(json.dumps(st.session_state.data_cpw, indent=2))
                    if st.session_state.data_coupler is not None:
                        # Try to get coupler type from results
                        coupler_type = results.iloc[0]['coupler_type'] if 'coupler_type' in results.iloc[0] else 'CoupledLineTee'
                        coupler_qcomp = "CapNInterdigital" if coupler_type in ["CapNInterdigital", "CapNInterdigitalTee"] else "CoupledLineTee"
                        st.markdown(f"**{coupler_qcomp} (Coupler) Values**")
                        st.code(json.dumps(st.session_state.data_coupler, indent=2))
                    if st.session_state.LJs is not None:
                        st.markdown("**Josephson Inductances (LJs, nH)**")
                        st.code(json.dumps(st.session_state.LJs, indent=2))
            with tab2:
                # Parameter Space Plots (was previously tab1, now tab2)
                try:
                    results = st.session_state.results
                    target = st.session_state.params
                    if system_type in ["Qubit-Cavity", "Qubit Only"]:
                        fig1 = px.scatter(
                            results,
                            x="qubit_frequency_GHz",
                            y="anharmonicity_MHz",
                            title="Qubit Frequency vs Anharmonicity",
                            labels={
                                "qubit_frequency_GHz": "Qubit Frequency (GHz)",
                                "anharmonicity_MHz": "Anharmonicity (MHz)"
                            }
                        )
                        fig1.add_scatter(
                            x=[target["qubit_frequency_GHz"]],
                            y=[target["anharmonicity_MHz"]],
                            mode="markers",
                            marker=dict(size=15, symbol="x", color="red"),
                            name="Target"
                        )
                        st.plotly_chart(fig1, use_container_width=True)
                    if system_type in ["Qubit-Cavity", "Cavity Only"]:
                        fig2 = px.scatter(
                            results,
                            x="cavity_frequency_GHz",
                            y="kappa_kHz",
                            title="Cavity Frequency vs Kappa",
                            labels={
                                "cavity_frequency_GHz": "Cavity Frequency (GHz)",
                                "kappa_kHz": "Kappa (kHz)"
                            }
                        )
                        fig2.add_scatter(
                            x=[target["cavity_frequency_GHz"]],
                            y=[target["kappa_kHz"]],
                            mode="markers",
                            marker=dict(size=15, symbol="x", color="red"),
                            name="Target"
                        )
                        st.plotly_chart(fig2, use_container_width=True)
                except Exception as e:
                    st.error(f"Error plotting results: {str(e)}")
            with tab3:
                # Design Options (was previously tab2, now tab3)
                for i, design in st.session_state.results.iterrows():
                    with st.expander(f"Design {i+1}"):
                        c1, c2 = st.columns(2)
                        if system_type == "Qubit Only":
                            with c1:
                                st.markdown("#### Qubit Parameters")
                                st.write(f"Frequency: {design['qubit_frequency_GHz']:.2f} GHz")
                                st.write(f"Anharmonicity: {design['anharmonicity_MHz']:.2f} MHz")
                                if 'EJ' in design and 'EC' in design:
                                    st.write(f"EJ/EC: {design['EJ']/design['EC']:.2f}")
                                qubit_options = convert_numpy_to_list(design.get('design_options', {}))
                                st.code(json.dumps(qubit_options, indent=2))
                                if st.button(f"Copy Qubit Design {i+1}", key=f"copy_qubit_{i}"):
                                    st.write("Qubit design copied to clipboard! ✅")
                                    st.session_state[f"clipboard_qubit_{i}"] = json.dumps(qubit_options)
                        elif system_type == "Cavity Only":
                            with c2:
                                st.markdown("#### Cavity Parameters")
                                st.write(f"Frequency: {design['cavity_frequency_GHz']:.2f} GHz")
                                st.write(f"Kappa: {design['kappa_kHz']:.2f} kHz")
                                if 'g_MHz' in design:
                                    st.write(f"Coupling g: {design['g_MHz']:.2f} MHz")
                                cavity_options = convert_numpy_to_list(design.get('design_options', {}))
                                st.code(json.dumps(cavity_options, indent=2))
                                if st.button(f"Copy Cavity Design {i+1}", key=f"copy_cavity_{i}"):
                                    st.write("Cavity design copied to clipboard! ✅")
                                    st.session_state[f"clipboard_cavity_{i}"] = json.dumps(cavity_options)
                        elif system_type == "Qubit-Cavity":
                            with c1:
                                st.markdown("#### Qubit Parameters")
                                st.write(f"Frequency: {design['qubit_frequency_GHz']:.2f} GHz")
                                st.write(f"Anharmonicity: {design['anharmonicity_MHz']:.2f} MHz")
                                if 'EJ' in design and 'EC' in design:
                                    st.write(f"EJ/EC: {design['EJ']/design['EC']:.2f}")
                                qubit_options = convert_numpy_to_list(design['design_options'].get('qubit_options', {}))
                                st.code(json.dumps(qubit_options, indent=2))
                                if st.button(f"Copy Qubit Design {i+1}", key=f"copy_qubit_{i}"):
                                    st.write("Qubit design copied to clipboard! ✅")
                                    st.session_state[f"clipboard_qubit_{i}"] = json.dumps(qubit_options)
                            with c2:
                                st.markdown("#### Cavity Parameters")
                                st.write(f"Frequency: {design['cavity_frequency_GHz']:.2f} GHz")
                                st.write(f"Kappa: {design['kappa_kHz']:.2f} kHz")
                                if 'g_MHz' in design:
                                    st.write(f"Coupling g: {design['g_MHz']:.2f} MHz")
                                cavity_options = convert_numpy_to_list(design['design_options'].get('cavity_claw_options', {}))
                                st.code(json.dumps(cavity_options, indent=2))
                                if st.button(f"Copy Cavity Design {i+1}", key=f"copy_cavity_{i}"):
                                    st.write("Cavity design copied to clipboard! ✅")
                                    st.session_state[f"clipboard_cavity_{i}"] = json.dumps(cavity_options)
                            # Show full design options with copy button
                            if st.checkbox(f"Show Full Design Options {i+1}", key=f"design_{i}"):
                                st.code(json.dumps(convert_numpy_to_list(design['design_options']), indent=2))
                                if st.button(f"Copy Full Design {i+1}", key=f"copy_full_{i}"):
                                    st.write("Full design copied to clipboard! ✅")
                                    st.session_state[f"clipboard_full_{i}"] = json.dumps(convert_numpy_to_list(design['design_options']))
            if system_type == "Qubit-Cavity":
                with tab4:
                    # Interpolated Results Tab
                    interpolated_df = st.session_state.get("interpolated_df", None)
                    if interpolated_df is None:
                        st.warning(st.session_state.get("interpolated_error", "No interpolated results available."))
                    else:
                        for i, row in interpolated_df.iterrows():
                            # Extract values for Design Parameters (mirroring main tab, but from interpolated data)
                            # Qubit
                            keys_cross = ["cross_gap", "cross_length", "cross_width"]
                            keys_claw = ["claw_gap", "claw_length", "claw_width", "ground_spacing"]
                            cross_vals = {k: [] for k in keys_cross}
                            claw_vals = {k: [] for k in keys_claw}
                            qubit_opts = convert_numpy_to_list(row['design_options'].get('qubit_options', {}))
                            for k in keys_cross:
                                v = qubit_opts.get(k)
                                if v is not None:
                                    cross_vals[k].append(v)
                            claw = qubit_opts.get('connection_pads', {}).get('readout', {})
                            for k in keys_claw:
                                v = claw.get(k)
                                if v is not None:
                                    claw_vals[k].append(v)
                            # CPW
                            cpw_keys = ["total_length", "trace_gap", "trace_width"]
                            cpw_vals = {k: [] for k in cpw_keys}
                            cpw_opts = convert_numpy_to_list(row['design_options'].get('cavity_claw_options', {}).get('cpw_opts', {}).get('left_options', {}))
                            for k in cpw_keys:
                                v = cpw_opts.get(k)
                                if v is not None:
                                    cpw_vals[k].append(v)
                            # Coupler
                            coupler_keys = ["coupling_length", "coupling_space", "down_length", "orientation", "prime_gap", "prime_width", "second_gap", "second_width", "cap_distance", "cap_gap", "cap_gap_ground", "cap_width", "finger_count", "finger_length"]
                            coupler_vals = {k: [] for k in coupler_keys}
                            coupler_opts = convert_numpy_to_list(row['design_options'].get('cavity_claw_options', {}).get('coupler_options', {}))
                            for k in coupler_keys:
                                v = coupler_opts.get(k)
                                if v is not None:
                                    coupler_vals[k].append(v)
                            with st.expander(f"Interpolated Design {i+1} - Design Parameters"):
                                st.markdown("**TransmonCross (Qubit) Values**")
                                st.code(json.dumps(cross_vals, indent=2))
                                st.markdown("**Claw Values**")
                                st.code(json.dumps(claw_vals, indent=2))
                                st.markdown("**RouteMeander (CPW) Values**")
                                st.code(json.dumps(cpw_vals, indent=2))
                                coupler_type = row['design_options'].get('cavity_claw_options', {}).get('coupler_type', 'CoupledLineTee')
                                coupler_qcomp = "CapNInterdigital" if coupler_type in ["CapNInterdigital", "CapNInterdigitalTee"] else "CoupledLineTee"
                                st.markdown(f"**{coupler_qcomp} (Coupler) Values**")
                                st.code(json.dumps(coupler_vals, indent=2))
                                if 'LJ' in row:
                                    st.markdown("**Josephson Inductance (LJ, nH)**")
                                    st.code(json.dumps(row['LJ'], indent=2))
                            with st.expander(f"Interpolated Design {i+1} - `design` Options"):
                                st.markdown("#### Qubit Parameters")
                                st.code(json.dumps(qubit_opts, indent=2))
                                st.markdown("#### Cavity Parameters")
                                cavity_opts = convert_numpy_to_list(row['design_options'].get('cavity_claw_options', {}))
                                st.code(json.dumps(cavity_opts, indent=2))
                                st.markdown("#### Coupler Parameters")
                                st.code(json.dumps(coupler_opts, indent=2))
                                st.markdown("#### CPW Parameters")
                                st.code(json.dumps(cpw_opts, indent=2))

    # Simple feedback link in the bottom right corner
    st.markdown(
        """
        <div style="position: fixed; bottom: 16px; right: 24px; z-index: 9999;">
            <a href="mailto:shanto@usc.edu?subject=SQuADDS%20WebUI%20Feedback%20or%20Feature%20Request&body=Please%20describe%20your%20bug%2C%20feature%20request%2C%20or%20feedback%20below%3A%0A%0A"
               style="color: #3578e5; font-size: 0.95em; text-decoration: underline; background: rgba(255,255,255,0.85); padding: 2px 10px; border-radius: 1em; box-shadow: 0 2px 8px rgba(0,0,0,0.04);">
                💬 Feedback
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main() 