from squadds.simulations.utils_geometry import get_cavity_claw_options_keys


def chunk_sweep_options(sweep_opts, N):
    if N <= 0:
        raise ValueError("N must be greater than 0.")

    cpw_opts_key, cplr_opts_key = get_cavity_claw_options_keys(sweep_opts)
    if cpw_opts_key is None or cplr_opts_key is None:
        raise ValueError("Sweep options must include both cavity and coupler option keys.")

    claw_lengths = sweep_opts["claw_opts"]["connection_pads"]["readout"]["claw_length"]
    total_lengths = sweep_opts[cpw_opts_key]["total_length"]
    base_chunk_size = len(claw_lengths) // N
    remainder = len(claw_lengths) % N
    chunks = []
    start_idx = 0

    for index in range(N):
        chunk_size = base_chunk_size + (1 if index < remainder else 0)
        claw_length_chunk = claw_lengths[start_idx : start_idx + chunk_size]
        new_sweep_opts = {
            "claw_opts": {"connection_pads": {"readout": sweep_opts["claw_opts"]["connection_pads"]["readout"].copy()}},
            cpw_opts_key: sweep_opts[cpw_opts_key].copy(),
            cplr_opts_key: sweep_opts[cplr_opts_key].copy(),
        }
        new_sweep_opts["claw_opts"]["connection_pads"]["readout"]["claw_length"] = claw_length_chunk
        new_sweep_opts[cpw_opts_key]["total_length"] = total_lengths
        chunks.append(new_sweep_opts)
        start_idx += chunk_size

    return chunks
