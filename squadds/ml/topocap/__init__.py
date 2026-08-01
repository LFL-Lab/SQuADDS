"""TopoCap: topology-general physical capacitance graph learning."""

from .adaptation import EBRAAdapter, EBRAConfig, adapt_foundation, foundation_fingerprint
from .evidence_gate import EvidenceGateConfig, EvidenceGatedTopoCap, GateEvidence
from .model import (
    CapacitancePrediction,
    EquivariantFeatureBuilder,
    GraphFeatureSignature,
    LatentCapacitancePrediction,
    TopoCapConfig,
    TopoCapFeatureMap,
    TopoCapFoundationModel,
)
from .retrieval import (
    RETRIEVAL_CONTROL_NAMES,
    RETRIEVAL_DISTANCE_CONTROL_NAMES,
    RETRIEVAL_PROTOCOL_VERSION,
    RETRIEVAL_SHUFFLE_SEED_XOR,
    RETRIEVAL_SOURCE_BUDGET,
    SupportConditionedSourceRetriever,
    SupportRetrievalSelection,
    retrieval_source_ids_sha256,
    retrieve_support_conditioned_source,
)
from .schema import CapacitanceGraph, canonical_edge_index, inverse_permutation
from .targets import (
    MaxwellComponents,
    MaxwellDiagnostics,
    components_to_maxwell,
    logs_to_maxwell,
    maxwell_diagnostics,
    maxwell_to_components,
)
from .views import (
    CAPN_INTERDIGITAL_TEE_CONTROLS,
    GENERALIZED_NCAP_CONTROLS,
    CanonicalControl,
    ControlSchema,
    build_active_geometry_view,
    build_topology_control_view,
    ncap_control_schema,
)

__all__ = [
    "CapacitanceGraph",
    "CapacitancePrediction",
    "EBRAAdapter",
    "EBRAConfig",
    "EvidenceGateConfig",
    "EvidenceGatedTopoCap",
    "EquivariantFeatureBuilder",
    "GraphFeatureSignature",
    "GateEvidence",
    "LatentCapacitancePrediction",
    "MaxwellComponents",
    "MaxwellDiagnostics",
    "RETRIEVAL_CONTROL_NAMES",
    "RETRIEVAL_DISTANCE_CONTROL_NAMES",
    "RETRIEVAL_PROTOCOL_VERSION",
    "RETRIEVAL_SHUFFLE_SEED_XOR",
    "RETRIEVAL_SOURCE_BUDGET",
    "SupportConditionedSourceRetriever",
    "SupportRetrievalSelection",
    "TopoCapConfig",
    "TopoCapFeatureMap",
    "TopoCapFoundationModel",
    "CAPN_INTERDIGITAL_TEE_CONTROLS",
    "CanonicalControl",
    "ControlSchema",
    "GENERALIZED_NCAP_CONTROLS",
    "adapt_foundation",
    "build_active_geometry_view",
    "build_topology_control_view",
    "canonical_edge_index",
    "components_to_maxwell",
    "foundation_fingerprint",
    "inverse_permutation",
    "logs_to_maxwell",
    "maxwell_diagnostics",
    "maxwell_to_components",
    "ncap_control_schema",
    "retrieval_source_ids_sha256",
    "retrieve_support_conditioned_source",
]
