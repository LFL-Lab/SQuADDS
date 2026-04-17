"""
Pydantic schemas for MCP tool inputs and outputs.
=================================================

These models define the structured data contracts for all MCP tools.
They enable:
  - Automatic JSON Schema generation for MCP clients
  - Input validation before tool execution
  - Type-safe, documented responses

Adding a New Schema
-------------------
1. Define a Pydantic BaseModel class with typed fields.
2. Use ``Field(...)`` to add descriptions — these become tool parameter docs.
3. Import and use the schema in your tool function's type annotations.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Input schemas
# ---------------------------------------------------------------------------


class SystemConfig(BaseModel):
    """Configuration for selecting a quantum system in SQuADDS."""

    system_type: str = Field(
        description=(
            "Type of system to search. "
            "One of: 'qubit_cavity' (coupled qubit+cavity), "
            "'qubit' (qubit only), 'cavity_claw' (cavity only)."
        ),
    )
    qubit: str = Field(
        default="TransmonCross",
        description="Qubit component name (e.g. 'TransmonCross').",
    )
    cavity: str = Field(
        default="RouteMeander",
        description="Cavity component name (e.g. 'RouteMeander').",
    )
    resonator_type: str = Field(
        default="quarter",
        description="Resonator type: 'quarter' or 'half'.",
    )


class TargetParams(BaseModel):
    """Target Hamiltonian parameters for a design search.

    Not all fields are required — which ones to set depends on ``system_type``:
    - **qubit**: ``qubit_frequency_GHz``, ``anharmonicity_MHz``
    - **cavity_claw**: ``cavity_frequency_GHz``, ``kappa_kHz``, ``resonator_type``
    - **qubit_cavity**: all of the above plus ``g_MHz``
    """

    qubit_frequency_GHz: Optional[float] = Field(
        default=None,
        description="Target qubit frequency in GHz (typical range: 3–8 GHz).",
    )
    anharmonicity_MHz: Optional[float] = Field(
        default=None,
        description="Target anharmonicity in MHz (typical range: −500 to −50 MHz, negative).",
    )
    cavity_frequency_GHz: Optional[float] = Field(
        default=None,
        description="Target cavity/resonator frequency in GHz (typical range: 5–12 GHz).",
    )
    kappa_kHz: Optional[float] = Field(
        default=None,
        description="Target cavity linewidth (kappa) in kHz (typical range: 10–1000 kHz).",
    )
    g_MHz: Optional[float] = Field(
        default=None,
        description="Target qubit-cavity coupling strength in MHz (typical range: 10–200 MHz).",
    )
    resonator_type: Optional[str] = Field(
        default=None,
        description="Resonator type: 'quarter' or 'half'. Required for cavity searches.",
    )

    def to_search_dict(self) -> dict[str, Any]:
        """Convert to a plain dict, dropping None values."""
        return {k: v for k, v in self.model_dump().items() if v is not None}


# ---------------------------------------------------------------------------
# Output schemas
# ---------------------------------------------------------------------------


class ComponentListResult(BaseModel):
    """List of supported components or component names."""

    items: list[str] = Field(description="List of component identifiers.")
    count: int = Field(description="Number of items.")


class ConfigListResult(BaseModel):
    """List of dataset configuration strings."""

    configs: list[str] = Field(description="List of config name strings (e.g. 'qubit-TransmonCross-cap_matrix').")
    count: int = Field(description="Number of configs.")


class DatasetEntry(BaseModel):
    """A single row from a SQuADDS dataset, serialized as a dict."""

    data: dict[str, Any] = Field(description="Column name → value mapping for this row.")


class DatasetResult(BaseModel):
    """Paginated dataset result."""

    rows: list[dict[str, Any]] = Field(description="List of row dicts.")
    total_rows: int = Field(description="Total number of rows in the full dataset.")
    offset: int = Field(description="Starting offset of this page.")
    limit: int = Field(description="Maximum rows per page.")
    component: str = Field(description="Component type.")
    component_name: str = Field(description="Component name.")
    data_type: str = Field(description="Data type.")


class DatasetInfoResult(BaseModel):
    """Metadata about a SQuADDS dataset."""

    config: str = Field(description="Config string (component-name-data_type).")
    num_rows: int = Field(description="Number of rows.")
    features: dict[str, str] = Field(description="Feature name → type mapping.")
    description: str = Field(default="", description="Dataset description.")
    size_bytes: Optional[int] = Field(default=None, description="Dataset size in bytes.")


class DatasetSummaryRow(BaseModel):
    """Summary of a single available dataset."""

    component: str
    component_name: str
    data_type: str


class DatasetSummaryResult(BaseModel):
    """Summary table of all available datasets."""

    datasets: list[DatasetSummaryRow]
    count: int


class MeasuredDeviceResult(BaseModel):
    """Information about a measured (experimental) device."""

    name: str = Field(description="Device name.")
    design_code: Optional[str] = Field(default=None, description="Design code identifier.")
    paper_link: Optional[str] = Field(default=None, description="Link to associated paper.")
    foundry: Optional[str] = Field(default=None, description="Fabrication foundry.")
    fabrication_recipe: Optional[Any] = Field(default=None, description="Fabrication recipe details.")


class DesignResult(BaseModel):
    """Result from a closest-design search."""

    rank: int = Field(description="Rank (1 = closest).")
    design_options: dict[str, Any] = Field(description="Full design options dict.")
    hamiltonian_params: dict[str, Any] = Field(description="Computed Hamiltonian parameters.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata (coupler_type, etc.).")


class ClosestDesignsResult(BaseModel):
    """Result set from find_closest_designs."""

    designs: list[DesignResult] = Field(description="Ranked list of closest designs.")
    num_results: int = Field(description="Number of results returned.")
    target_params: dict[str, Any] = Field(description="The target parameters used for the search.")
    system_config: dict[str, Any] = Field(description="The system configuration used.")


class InterpolatedDesignResult(BaseModel):
    """Result from scaling interpolation."""

    design_options: dict[str, Any] = Field(description="Interpolated design options (qubit + cavity + coupler).")
    qubit_options: dict[str, Any] = Field(description="Qubit-specific design options.")
    cavity_options: dict[str, Any] = Field(description="Cavity-specific design options.")
    coupler_type: Optional[str] = Field(default=None, description="Coupler type used.")
    scaling_info: dict[str, Any] = Field(default_factory=dict, description="Scaling factors applied.")


class ContributorInfo(BaseModel):
    """Information about a data contributor."""

    uploader: Optional[str] = None
    PI: Optional[str] = None
    group: Optional[str] = None
    institution: Optional[str] = None


class ContributorsResult(BaseModel):
    """List of contributors."""

    contributors: list[ContributorInfo]
    count: int


class HamiltonianKeysResult(BaseModel):
    """Valid Hamiltonian parameter keys for a given system type."""

    keys: list[str] = Field(description="List of valid target parameter keys.")
    system_type: str = Field(description="The system type these keys apply to.")


class VersionResult(BaseModel):
    """SQuADDS version information."""

    squadds_version: str
    mcp_server_version: str
    repo_name: str
