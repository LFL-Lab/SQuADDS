"""Universal GDS Embedding framework: GDS -> scene -> scene graph (+ physics-proxy plugins) -> universal vector / GNN.

The recipe is dataset-agnostic; what a user adapts is (1) the layer map / net naming of their GDS, (2) the physics proxy
plugin that fits their observable (capacitance, resonance frequency, kappa, ...), and (3) the target adapter.
"""
