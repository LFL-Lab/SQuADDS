"""
Tests for SQuADDS MCP Pydantic schemas.
========================================

Validates that all schemas serialize/deserialize correctly and
enforce their constraints.
"""

from squadds_mcp.schemas import (
    ClosestDesignsResult,
    ComponentListResult,
    DatasetResult,
    DesignResult,
    HamiltonianKeysResult,
    InterpolatedDesignResult,
    TargetParams,
    VersionResult,
)


class TestTargetParams:
    """Tests for TargetParams schema."""

    def test_full_params(self):
        params = TargetParams(
            qubit_frequency_GHz=4.5,
            anharmonicity_MHz=-200.0,
            cavity_frequency_GHz=9.2,
            kappa_kHz=80.0,
            g_MHz=70.0,
            resonator_type="quarter",
        )
        assert params.qubit_frequency_GHz == 4.5
        assert params.resonator_type == "quarter"

    def test_partial_params(self):
        params = TargetParams(qubit_frequency_GHz=5.0, anharmonicity_MHz=-180.0)
        assert params.cavity_frequency_GHz is None
        assert params.g_MHz is None

    def test_to_search_dict_drops_none(self):
        params = TargetParams(qubit_frequency_GHz=5.0, anharmonicity_MHz=-180.0)
        d = params.to_search_dict()
        assert "qubit_frequency_GHz" in d
        assert "cavity_frequency_GHz" not in d
        assert len(d) == 2

    def test_all_none(self):
        params = TargetParams()
        d = params.to_search_dict()
        assert d == {}


class TestComponentListResult:
    """Tests for ComponentListResult."""

    def test_basic(self):
        result = ComponentListResult(items=["qubit", "cavity_claw", "coupler"], count=3)
        assert result.count == 3
        assert "qubit" in result.items

    def test_empty(self):
        result = ComponentListResult(items=[], count=0)
        assert result.count == 0


class TestDesignResult:
    """Tests for DesignResult."""

    def test_basic(self):
        result = DesignResult(
            rank=1,
            design_options={"cross_length": "300um"},
            hamiltonian_params={"qubit_frequency_GHz": 4.5},
        )
        assert result.rank == 1
        assert result.design_options["cross_length"] == "300um"

    def test_with_metadata(self):
        result = DesignResult(
            rank=2,
            design_options={},
            hamiltonian_params={},
            metadata={"coupler_type": "CLT"},
        )
        assert result.metadata["coupler_type"] == "CLT"


class TestClosestDesignsResult:
    """Tests for ClosestDesignsResult."""

    def test_basic(self):
        designs = [
            DesignResult(rank=1, design_options={}, hamiltonian_params={"fq": 4.5}),
            DesignResult(rank=2, design_options={}, hamiltonian_params={"fq": 4.6}),
        ]
        result = ClosestDesignsResult(
            designs=designs,
            num_results=2,
            target_params={"fq": 4.5},
            system_config={"system_type": "qubit"},
        )
        assert result.num_results == 2
        assert result.designs[0].rank == 1


class TestInterpolatedDesignResult:
    """Tests for InterpolatedDesignResult."""

    def test_basic(self):
        result = InterpolatedDesignResult(
            design_options={"qubit_options": {}, "cavity_claw_options": {}},
            qubit_options={"cross_length": "310um"},
            cavity_options={"total_length": "4200um"},
            coupler_type="CLT",
        )
        assert result.coupler_type == "CLT"
        assert result.qubit_options["cross_length"] == "310um"


class TestDatasetResult:
    """Tests for DatasetResult."""

    def test_pagination_fields(self):
        result = DatasetResult(
            rows=[{"a": 1}, {"a": 2}],
            total_rows=100,
            offset=0,
            limit=50,
            component="qubit",
            component_name="TransmonCross",
            data_type="cap_matrix",
        )
        assert result.total_rows == 100
        assert len(result.rows) == 2
        assert result.offset == 0


class TestHamiltonianKeysResult:
    """Tests for HamiltonianKeysResult."""

    def test_basic(self):
        result = HamiltonianKeysResult(
            keys=["qubit_frequency_GHz", "anharmonicity_MHz"],
            system_type="qubit",
        )
        assert len(result.keys) == 2
        assert result.system_type == "qubit"


class TestVersionResult:
    """Tests for VersionResult."""

    def test_basic(self):
        result = VersionResult(
            squadds_version="0.4.5",
            mcp_server_version="0.1.0",
            repo_name="SQuADDS/SQuADDS_DB",
        )
        assert result.squadds_version == "0.4.5"


class TestSerializationRoundtrip:
    """Test that schemas serialize to JSON and back."""

    def test_target_params_roundtrip(self):
        params = TargetParams(qubit_frequency_GHz=5.0, anharmonicity_MHz=-200.0)
        json_str = params.model_dump_json()
        reconstructed = TargetParams.model_validate_json(json_str)
        assert reconstructed.qubit_frequency_GHz == 5.0

    def test_design_result_roundtrip(self):
        result = DesignResult(
            rank=1,
            design_options={"nested": {"key": [1, 2, 3]}},
            hamiltonian_params={"fq": 4.5},
        )
        json_str = result.model_dump_json()
        reconstructed = DesignResult.model_validate_json(json_str)
        assert reconstructed.design_options["nested"]["key"] == [1, 2, 3]
