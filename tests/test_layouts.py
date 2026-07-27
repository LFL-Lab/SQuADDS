import json

import numpy as np
import pandas as pd
import pytest

from squadds.core.db import SQuADDS_DB
from squadds.layouts import (
    UNIVERSAL_EMBEDDING_DIMENSIONS,
    LayoutClient,
    LayoutEmbeddingClient,
    StaticEmbeddingClient,
    V0KernelFeatureProjector,
    build_geometry_features,
    build_static_embeddings,
    canonical_design_id,
    compress_v0_embeddings,
    infer_layout_component_name,
    mean_pairwise_cosine_similarity,
    parameter_channels,
    parameter_sum,
    parse_gds_polygons,
    parse_gds_summary,
    write_universal_embedding_dataset,
)


def test_canonical_design_id_is_order_independent():
    first = canonical_design_id("GeneralizedCapNInterdigital", {"finger_count": "5", "width": "10um"})
    second = canonical_design_id("GeneralizedCapNInterdigital", {"width": "10um", "finger_count": "5"})

    assert first == second
    assert first.startswith("design:sha256:")


def test_layout_client_finds_local_manifest_record(tmp_path):
    manifest = tmp_path / "manifest.parquet"
    artifact = tmp_path / "raw" / "cap.gds"
    artifact.parent.mkdir()
    artifact.write_bytes(b"gds")
    pd.DataFrame(
        [
            {
                "layout_id": "layout:sha256:layout",
                "artifact_id": "sha256:8261e1a4f86abe76f7a1e8a66ae713e37dc9e651408b95b7db70ea062d5fc071",
                "gds_path": "raw/cap.gds",
                "component_name": "GeneralizedCapNInterdigital",
                "source_id": "exp6/cap_0000",
                "design_id": "design:sha256:design",
            }
        ]
    ).to_parquet(manifest, index=False)

    client = LayoutClient(manifest_path=manifest, artifact_root=tmp_path)
    reference = client.find(source_id="exp6/cap_0000")

    assert reference.gds_path == "raw/cap.gds"
    assert reference.component_name == "GeneralizedCapNInterdigital"
    assert client.download(reference) == artifact


def test_layout_client_resolves_repeated_design_rows_with_one_geometry(tmp_path):
    manifest = tmp_path / "manifest.parquet"
    pd.DataFrame(
        [
            {
                "layout_id": "layout:shared",
                "artifact_id": "sha256:first",
                "gds_path": "raw/first.gds",
                "component_name": "CavityClawRouteMeander",
                "design_id": "design:shared",
            },
            {
                "layout_id": "layout:shared",
                "artifact_id": "sha256:second",
                "gds_path": "raw/second.gds",
                "component_name": "CavityClawRouteMeander",
                "design_id": "design:shared",
            },
        ]
    ).to_parquet(manifest, index=False)

    reference = LayoutClient(manifest_path=manifest).find(design_id="design:shared")

    assert reference.layout_id == "layout:shared"
    assert reference.gds_path == "raw/first.gds"


def test_layout_client_reads_local_geometry_features(tmp_path):
    manifest = tmp_path / "manifest.parquet"
    features = tmp_path / "geometry-features.parquet"
    pd.DataFrame(
        [
            {
                "layout_id": "layout:sha256:layout",
                "artifact_id": "sha256:artifact",
                "gds_path": "raw/cap.gds",
                "component_name": "GeneralizedCapNInterdigital",
            }
        ]
    ).to_parquet(manifest, index=False)
    pd.DataFrame(
        [
            {
                "layout_id": "layout:sha256:layout",
                "geometry_feature_schema_version": "1.0.0",
                "polygon_count": 5,
            }
        ]
    ).to_parquet(features, index=False)

    client = LayoutClient(manifest_path=manifest, geometry_features_path=features)
    reference = client.find(layout_id="layout:sha256:layout")

    assert client.geometry_features(reference)["polygon_count"] == 5


def test_database_layout_bridge_uses_source_id():
    class FakeLayoutClient:
        def find(self, **kwargs):
            return kwargs

    row = {"notes": {"source_id": "exp7/cap_0001"}}

    assert SQuADDS_DB.get_layout_ref(row, layout_client=FakeLayoutClient()) == {"source_id": "exp7/cap_0001"}


def test_database_layout_bridge_derives_design_id_without_source_id():
    class FakeLayoutClient:
        def find(self, **kwargs):
            return kwargs

    options = {"finger_count": "5", "finger_width": "10um"}
    row = {"design": {"coupler_type": "CapNInterdigitalTee", "design_options": options}}

    assert SQuADDS_DB.get_layout_ref(row, layout_client=FakeLayoutClient()) == {
        "design_id": canonical_design_id("CapNInterdigitalTee", options)
    }


def test_database_layout_bridge_infers_cavity_and_transmon_layout_families():
    class FakeLayoutClient:
        def find(self, **kwargs):
            return kwargs

    cavity_options = {"claw_opts": {}, "cpw_opts": {}, "cplr_opts": {}}
    cavity_row = {"design": {"design_options": cavity_options, "coupler_type": "CLT"}}
    transmon_options = {"cross_width": "30um", "cross_length": "300um", "cross_gap": "20um"}
    transmon_row = {"design": {"design_options": transmon_options}}

    assert infer_layout_component_name(cavity_row["design"]) == "CavityClawRouteMeander"
    assert infer_layout_component_name(transmon_row["design"]) == "TransmonCross"
    assert SQuADDS_DB.get_layout_ref(cavity_row, layout_client=FakeLayoutClient()) == {
        "design_id": canonical_design_id("CavityClawRouteMeander", cavity_options)
    }
    assert SQuADDS_DB.get_layout_ref(transmon_row, layout_client=FakeLayoutClient()) == {
        "design_id": canonical_design_id("TransmonCross", transmon_options)
    }


def test_database_layout_bridge_infers_flattened_simulation_rows():
    class FakeLayoutClient:
        def find(self, **kwargs):
            return kwargs

    options = {"cross_width": "30um", "cross_length": "300um", "cross_gap": "20um"}
    row = {"design_options": options}

    assert SQuADDS_DB.get_layout_ref(row, layout_client=FakeLayoutClient()) == {
        "design_id": canonical_design_id("TransmonCross", options)
    }


def test_database_embedding_bridge_uses_resolved_layout():
    class FakeLayoutClient:
        def find(self, **kwargs):
            assert kwargs == {"source_id": "exp7/cap_0001"}
            return type("Reference", (), {"layout_id": "layout:one"})()

    class FakeEmbeddingClient:
        def get(self, layout_id):
            return {"layout_id": layout_id, "embedding": [1.0]}

    row = {"notes": {"source_id": "exp7/cap_0001"}}

    assert SQuADDS_DB.get_layout_embedding(
        row,
        layout_client=FakeLayoutClient(),
        embedding_client=FakeEmbeddingClient(),
    ) == {"layout_id": "layout:one", "embedding": [1.0]}


def test_geometry_features_are_layer_aware_and_model_ready():
    manifest = pd.DataFrame(
        [
            {
                "layout_id": "layout:one",
                "artifact_id": "sha256:one",
                "component_name": "CapNInterdigitalTee",
                "source_id": "generated/capn_0000",
                "bbox_um": {"left": 0, "bottom": 0, "right": 10, "top": 5},
                "polygon_count": 3,
                "cell_count": 2,
                "layers": [
                    {"layer": 2, "datatype": 0, "polygon_count": 1, "area_um2": 4.0},
                    {"layer": 1, "datatype": 10, "polygon_count": 2, "area_um2": 6.0},
                ],
            }
        ]
    )

    feature = build_geometry_features(manifest).iloc[0]

    assert feature["bbox_area_um2"] == 50.0
    assert feature["bbox_aspect_ratio"] == 2.0
    assert feature["total_area_um2"] == 10.0
    assert feature["layer_features"] == [
        {"layer": 1, "datatype": 10, "polygon_count": 2, "area_um2": 6.0},
        {"layer": 2, "datatype": 0, "polygon_count": 1, "area_um2": 4.0},
    ]


def test_static_v0_embeddings_include_96x96_shape_and_are_searchable(tmp_path):
    kdb = pytest.importorskip("klayout.db")
    manifest_records = []
    design_options = {}
    paths = {}
    for index in range(3):
        layout = kdb.Layout()
        layout.dbu = 0.001
        top = layout.create_cell("TOP")
        top.shapes(layout.layer(1, 10)).insert(kdb.Box(0, 0, 1000 + 200 * index, 2000))
        path = tmp_path / f"layout_{index}.gds"
        layout.write(str(path))
        design_id = f"design:{index}"
        manifest_records.append(
            {
                "layout_id": f"layout:{index}",
                "artifact_id": f"sha256:{index}",
                "design_id": design_id,
                "component_name": "CapNInterdigitalTee",
                "source_id": f"capn/{index}",
                "gds_path": path.name,
            }
        )
        design_options[design_id] = {"finger_width": f"{index + 1}um", "finger_count": index + 1}
        paths[path.name] = path

    vectors, schema = build_static_embeddings(
        pd.DataFrame(manifest_records),
        design_options,
        lambda record: paths[record["gds_path"]],
    )
    duplicate = vectors.iloc[[0]].assign(source_id="capn/duplicate")
    vectors = pd.concat([vectors, duplicate], ignore_index=True)
    vector_path = tmp_path / "vectors.parquet"
    vectors.to_parquet(vector_path, index=False)

    client = StaticEmbeddingClient(embedding_path=vector_path)

    assert schema["model"] == "static-shape-v0"
    assert schema["blocks"]["shape_bitmap"]["shape"] == [96, 96]
    assert schema["dimensions"] == 9227
    assert len(client.get("layout:0")["embedding"]) == 9227
    assert client.get("layout:0")["source_id"] == "capn/0"
    assert client.shape_bitmap("layout:0").shape == (96, 96)
    assert np.linalg.norm(np.asarray(client.get("layout:0")["embedding"])) == pytest.approx(1.0)
    assert len(client.nearest("layout:0", limit=2)) == 2


def test_parameter_sum_is_permutation_and_dimension_invariant():
    first = {"width": "10um", "nested": {"count": 3, "enabled": True}}
    second = {"enabled": True, "count": 3, "width": "0.01mm"}

    assert parameter_sum(first) == parameter_sum(second)


def test_universal_v1_embeddings_are_compact_versioned_and_controllable(tmp_path):
    kdb = pytest.importorskip("klayout.db")
    manifest_records = []
    design_options = {}
    paths = {}
    for index in range(3):
        layout = kdb.Layout()
        layout.dbu = 0.001
        top = layout.create_cell("TOP")
        top.shapes(layout.layer(1, 10)).insert(kdb.Box(0, 0, 1000 + 200 * index, 2000))
        top.shapes(layout.layer(2, 0)).insert(kdb.Box(400, 2000, 600, 2200))
        path = tmp_path / f"universal_{index}.gds"
        layout.write(str(path))
        design_id = f"universal-design:{index}"
        manifest_records.append(
            {
                "layout_id": f"universal-layout:{index}",
                "artifact_id": f"sha256:universal-{index}",
                "design_id": design_id,
                "component_name": "GeneralizedCapNInterdigital",
                "source_id": f"generalized/{index}",
                "gds_path": path.name,
            }
        )
        design_options[design_id] = {
            "finger_width": f"{index + 1}um",
            "finger_count": index + 2,
        }
        paths[path.name] = path

    output = tmp_path / "release"
    count, dimensions = write_universal_embedding_dataset(
        pd.DataFrame(manifest_records),
        design_options,
        lambda record: paths[record["gds_path"]],
        output,
        component_name="GeneralizedCapNInterdigital",
    )
    vector_path = output / "metadata" / "universal-geometry-v1.parquet"
    client = LayoutEmbeddingClient(
        version="v1",
        embedding_path=vector_path,
        schema_path=output / "models" / "universal-geometry-v1" / "schema.json",
        control_map_path=output / "models" / "universal-geometry-v1" / "control-map.parquet",
    )
    record = client.get("universal-layout:0")

    assert count == 3
    assert dimensions == UNIVERSAL_EMBEDDING_DIMENSIONS == 1024
    assert record["embedding_version"] == "v1"
    assert record["embedding_model"] == "universal-geometry-v1"
    assert len(record["geometry_metrics"]) == 32
    assert len(record["embedding"]) == 1024
    assert record["shape_descriptor_norm"] > 0
    assert np.linalg.norm(np.asarray(record["embedding"])) == pytest.approx(1.0)
    assert list(record["parameter_names"]) == ["finger_count", "finger_width"]
    assert len(client.nearest("universal-layout:0", limit=2)) == 2
    rasters = client.shape_rasters("universal-layout:0")
    assert set(rasters) == {
        "signed_functional_material",
        "signed_distance_to_functional_boundary",
    }
    assert rasters["signed_functional_material"].shape == (96, 96)
    assert client.schema()["dimensions"] == 1024
    assert client.schema()["embedding_schema_version"] == "1.1.0"
    assert set(client.control_map()["parameter_name"]) == {"finger_count", "finger_width"}
    assert client.available_versions() == ("v0", "v1")
    assert (output / "models" / "universal-geometry-v1" / "schema.json").is_file()
    assert (output / "models" / "universal-geometry-v1" / "control-map.parquet").is_file()
    release_manifest = json.loads((output / "models" / "universal-geometry-v1" / "release-manifest.json").read_text())
    assert release_manifest["rows"] == 3
    assert release_manifest["dimensions"] == 1024
    with pytest.raises(ValueError, match="only available for v0"):
        client.shape_bitmap("universal-layout:0")


def test_v1_parameter_channels_keep_name_identity_and_order_invariance():
    first = {"finger_width": "4um", "finger_count": 8}
    second = {"finger_count": 8, "finger_width": "0.004mm"}

    first_vector, first_names, first_values, first_indices, _ = parameter_channels(first)
    second_vector, second_names, second_values, second_indices, _ = parameter_channels(second)

    assert np.array_equal(first_vector, second_vector)
    assert first_names == second_names == ["finger_count", "finger_width"]
    assert first_values == second_values == [8.0, 4.0]
    assert first_indices == second_indices


def test_embedding_client_rejects_unknown_versions():
    with pytest.raises(ValueError, match="Unknown embedding version"):
        LayoutEmbeddingClient(version="future")


def test_v0_kernel_projector_is_deterministic_and_nonlinear():
    rng = np.random.default_rng(16)
    embeddings = rng.normal(size=(8, 9227))
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    first = V0KernelFeatureProjector(
        pooled_shape_size=12,
        kernel_dimensions=32,
        random_seed=4,
    ).fit_transform(embeddings)
    second = V0KernelFeatureProjector(
        pooled_shape_size=12,
        kernel_dimensions=32,
        random_seed=4,
    ).fit_transform(embeddings)

    assert first.shape == (8, 187)
    assert np.allclose(first, second)
    assert not np.allclose(first[:, -32:], 0.0)

    compact = compress_v0_embeddings(embeddings, pooled_shape_size=12)
    streamed = V0KernelFeatureProjector(
        pooled_shape_size=12,
        kernel_dimensions=32,
        random_seed=4,
    ).fit_transform_compact(compact)

    assert streamed.shape == first.shape
    assert np.allclose(streamed, first)


def test_mean_pairwise_cosine_similarity_matches_explicit_matrix():
    first = np.asarray([[1.0, 0.0], [1.0, 1.0]])
    second = np.asarray([[0.0, 1.0], [1.0, -1.0]])
    normalized_first = first / np.linalg.norm(first, axis=1, keepdims=True)
    normalized_second = second / np.linalg.norm(second, axis=1, keepdims=True)

    expected = float(np.mean(normalized_first @ normalized_second.T))

    assert mean_pairwise_cosine_similarity(first, second) == pytest.approx(expected)


def test_parse_gds_summary_extracts_layers_and_units(tmp_path):
    kdb = pytest.importorskip("klayout.db")
    layout = kdb.Layout()
    layout.dbu = 0.001
    top = layout.create_cell("TOP")
    top.shapes(layout.layer(1, 10)).insert(kdb.Box(0, 0, 1000, 2000))
    gds_path = tmp_path / "layout.gds"
    layout.write(str(gds_path))

    summary = parse_gds_summary(gds_path)

    assert summary["top_cell"] == "TOP"
    assert summary["dbu_um"] == 0.001
    assert summary["polygon_count"] == 1
    assert summary["layers"] == [
        {
            "layer": 1,
            "datatype": 10,
            "polygon_count": 1,
            "area_um2": 2.0,
            "bbox_um": {"left": 0.0, "bottom": 0.0, "right": 1.0, "top": 2.0},
        }
    ]
    assert parse_gds_polygons(gds_path, layer=1, datatype=10) == [
        {
            "layer": 1,
            "datatype": 10,
            "polygon_index": 0,
            "points_um": [
                {"x": 0.0, "y": 0.0},
                {"x": 0.0, "y": 2.0},
                {"x": 1.0, "y": 2.0},
                {"x": 1.0, "y": 0.0},
            ],
            "area_um2": 2.0,
            "bbox_um": {"left": 0.0, "bottom": 0.0, "right": 1.0, "top": 2.0},
        }
    ]
