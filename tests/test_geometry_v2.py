"""Contract tests for the catalogue-free universal-geometry-v2 encoder.

These check the properties that make v2 usable as a foundation-model substrate:
a fixed width for any parameter schema, exact translation invariance, deliberate
scale sensitivity, and monotone response to the geometric quantities that set
capacitance.
"""

from __future__ import annotations

import numpy as np
import pytest

from squadds.layouts.geometry_v2 import (
    COUPLING_BINS,
    METRIC_BLOCK_SIZE,
    METRIC_NAMES,
    V2_DIMENSIONS,
    classify_parameter,
    encode,
    parameter_block,
    universal_v2_schema,
)

OPTIONS = {
    "finger_count": 6,
    "finger_length": "40um",
    "finger_width": "4um",
    "finger_gap": "3um",
}


def build_interdigital(
    path,
    *,
    finger_count=6,
    finger_length=40.0,
    finger_width=4.0,
    finger_gap=3.0,
    ground_gap=10.0,
    scale=1.0,
    shift=(0.0, 0.0),
    mirror=False,
):
    """Write a two-terminal interdigital capacitor with a cut-out ground plane."""
    kdb = pytest.importorskip("klayout.db")
    layout = kdb.Layout()
    layout.dbu = 0.001
    top = layout.create_cell("TOP")
    conductor = layout.layer(1, 10)
    domain = layout.layer(1, 0)
    ports = (layout.layer(2, 0), layout.layer(3, 0))

    def box(x0, y0, x1, y1):
        if mirror:
            x0, x1 = -x1, -x0
        shift_x, shift_y = shift
        return kdb.Box(
            int(round((x0 * scale + shift_x) * 1000)),
            int(round((y0 * scale + shift_y) * 1000)),
            int(round((x1 * scale + shift_x) * 1000)),
            int(round((y1 * scale + shift_y) * 1000)),
        )

    pitch = 2 * (finger_width + finger_gap)
    span = finger_count * pitch
    spine = 6.0
    half = finger_length / 2
    outer = half + spine

    ground = kdb.Region(box(-span, -3 * outer, 2 * span, 3 * outer))
    conductors = kdb.Region()
    conductors.insert(box(0.0, half, span, outer))
    conductors.insert(box(0.0, -outer, span, -half))
    for index in range(finger_count):
        left = index * pitch + finger_gap
        conductors.insert(box(left, -half + finger_gap, left + finger_width, half))
        opposing = left + finger_width + finger_gap
        conductors.insert(box(opposing, -half, opposing + finger_width, half - finger_gap))

    moat = conductors.sized(int(round(ground_gap * scale * 1000)))
    top.shapes(domain).insert(ground - moat)
    top.shapes(conductor).insert(conductors)
    top.shapes(ports[0]).insert(box(span / 2 - 1, outer, span / 2 + 1, outer + 2))
    top.shapes(ports[1]).insert(box(span / 2 - 1, -outer - 2, span / 2 + 1, -outer))
    layout.write(str(path))
    return path


@pytest.fixture
def baseline(tmp_path):
    return encode(build_interdigital(tmp_path / "baseline.gds"), OPTIONS)


def test_schema_is_frozen_and_declares_no_catalogue_fit():
    schema = universal_v2_schema()

    assert schema["dimensions"] == V2_DIMENSIONS == 512
    assert schema["fitted_on_catalogue"] is False
    assert schema["input_contract"]["simulation_results_used"] is False
    assert len(METRIC_NAMES) == METRIC_BLOCK_SIZE
    offsets = [block["offset"] for block in schema["blocks"].values()]
    widths = [block["dimensions"] for block in schema["blocks"].values()]
    assert offsets == sorted(offsets)
    assert sum(widths) == V2_DIMENSIONS
    assert offsets[0] == 0 and offsets[-1] + widths[-1] == V2_DIMENSIONS


def test_width_is_fixed_for_any_parameter_schema(tmp_path):
    path = build_interdigital(tmp_path / "schema.gds")
    empty = encode(path, {})
    local = encode(path, OPTIONS)
    foreign = encode(path, {f"whacky_knob_{index}": f"{1.5 * index}um" for index in range(28)})
    enormous = encode(path, {f"p{index}": index * 0.25 for index in range(500)})

    assert empty.shape == local.shape == foreign.shape == enormous.shape == (V2_DIMENSIONS,)
    assert all(np.isfinite(vector).all() for vector in (empty, local, foreign, enormous))
    # Geometry blocks are identical; only the parameter block reacts.
    geometry_width = V2_DIMENSIONS - 96 - 48
    assert np.array_equal(empty[:geometry_width], foreign[:geometry_width])
    assert not np.array_equal(empty, foreign)


def test_translation_is_exactly_invariant(tmp_path, baseline):
    shifted = encode(build_interdigital(tmp_path / "shifted.gds", shift=(4321.0, -8765.0)), OPTIONS)

    assert np.array_equal(baseline, shifted)


def test_encoding_is_deterministic(tmp_path, baseline):
    repeat = encode(build_interdigital(tmp_path / "repeat.gds"), OPTIONS)

    assert np.array_equal(baseline, repeat)


def test_scale_changes_the_vector_unlike_v0(tmp_path, baseline):
    """v0 and v1 crop to functional bounds, so a scaled copy is identical there.

    Capacitance is set by absolute separation, so v2 must resolve the difference.
    """
    from squadds.layouts.embeddings import rasterize_functional_shape

    small = build_interdigital(tmp_path / "small.gds")
    large = build_interdigital(tmp_path / "large.gds", scale=4.0)

    v0_small, _, _ = rasterize_functional_shape(small, "CapNInterdigitalTee")
    v0_large, _, _ = rasterize_functional_shape(large, "CapNInterdigitalTee")
    assert np.array_equal(v0_small, v0_large), "v0 bitmaps are expected to be scale blind"

    scaled = encode(large, OPTIONS)
    assert not np.array_equal(baseline, scaled)
    assert float(np.max(np.abs(baseline - scaled))) > 1.0


def test_coupling_spectrum_tracks_the_conductor_gap(tmp_path):
    """A wider finger gap must move facing boundary length into farther bins."""
    narrow = encode(build_interdigital(tmp_path / "narrow.gds", finger_gap=2.0), OPTIONS)
    wide = encode(build_interdigital(tmp_path / "wide.gds", finger_gap=12.0), OPTIONS)

    start = METRIC_BLOCK_SIZE
    narrow_spectrum = narrow[start : start + COUPLING_BINS]
    wide_spectrum = wide[start : start + COUPLING_BINS]
    positions = np.arange(COUPLING_BINS)

    def centroid(spectrum):
        weight = np.maximum(spectrum, 0.0)
        return float((positions * weight).sum() / max(weight.sum(), 1e-12))

    assert narrow_spectrum.sum() > 0 and wide_spectrum.sum() > 0
    assert centroid(wide_spectrum) > centroid(narrow_spectrum)

    gap_index = METRIC_NAMES.index("log1p_minimum_pair_gap_um")
    assert wide[gap_index] > narrow[gap_index]


def test_ground_spectrum_responds_to_the_ground_moat(tmp_path):
    tight = encode(build_interdigital(tmp_path / "tight.gds", ground_gap=4.0), OPTIONS)
    loose = encode(build_interdigital(tmp_path / "loose.gds", ground_gap=30.0), OPTIONS)

    assert not np.array_equal(tight, loose)
    ground_start = METRIC_BLOCK_SIZE + 6 * COUPLING_BINS
    assert tight[ground_start : ground_start + 12].sum() > 0
    assert loose[ground_start : ground_start + 12].sum() > 0


def test_finger_count_is_resolved(tmp_path):
    few = encode(build_interdigital(tmp_path / "few.gds", finger_count=3), OPTIONS)
    many = encode(build_interdigital(tmp_path / "many.gds", finger_count=11), OPTIONS)

    assert float(np.max(np.abs(few - many))) > 0.5


def test_mirror_leaves_orientation_free_blocks_stable(tmp_path, baseline):
    """Mirror invariance is approximate, not exact.

    The coupling spectrum bins distances sampled along a boundary, so mirroring
    reshuffles which samples sit near a bin edge.  The schema advertises this as
    approximate; the tolerance below is what "approximate" is allowed to mean.
    """
    mirrored = encode(build_interdigital(tmp_path / "mirrored.gds", mirror=True), OPTIONS)

    start = METRIC_BLOCK_SIZE
    stop = start + 6 * COUPLING_BINS
    original = baseline[start:stop]
    reflected = mirrored[start:stop]
    assert np.allclose(original, reflected, rtol=0.01, atol=0.01)
    cosine = float(original @ reflected / (np.linalg.norm(original) * np.linalg.norm(reflected)))
    assert cosine > 0.9999


def test_parameter_classification_is_dimension_aware():
    assert classify_parameter("finger_width", "4um") == ("length", 4.0)
    assert classify_parameter("finger_width", "4000nm") == ("length", 4.0)
    assert classify_parameter("finger_count", 6) == ("count", 6.0)
    assert classify_parameter("orientation", "90deg") == ("angle", 90.0)
    assert classify_parameter("enabled", True) == ("boolean", 1.0)
    assert classify_parameter("label", "not-a-number") is None


def test_parameter_block_is_unit_normalized_and_order_independent():
    metric, _ = parameter_block({"a_width": "10um", "b_count": 3})
    reordered, _ = parameter_block({"b_count": 3, "a_width": "0.01mm"})

    assert np.allclose(metric, reordered)


def test_parameter_block_separates_dimension_classes():
    lengths, metadata = parameter_block({"gap": "2um", "length": "80um", "count": 4})

    assert metadata["parameter_count"] == 3
    assert metadata["parameter_classes"].count("length") == 2
    assert metadata["parameter_classes"].count("count") == 1
    assert np.isfinite(lengths).all()


def test_nearest_returns_a_true_cosine_for_unnormalized_vectors(tmp_path):
    """v2 vectors are absolute measurements, not unit vectors.

    ``nearest`` previously returned a bare dot product, which coincided with the
    cosine only because v0 and v1 are unit normalized.  On v2 that reported
    similarities in the thousands.
    """
    import pandas as pd

    from squadds.layouts import LayoutEmbeddingClient

    rng = np.random.default_rng(0)
    rows = []
    for index in range(6):
        vector = (rng.normal(size=V2_DIMENSIONS) * 50.0).astype(np.float32)
        rows.append(
            {
                "layout_id": f"layout:{index}",
                "artifact_id": f"sha256:{index}",
                "design_id": f"design:{index}",
                "component_name": "GeneralizedCapNInterdigital",
                "source_id": f"campaign/{index}",
                "embedding": vector.tolist(),
            }
        )
    path = tmp_path / "v2.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)

    client = LayoutEmbeddingClient(version="v2", embedding_path=path)
    neighbours = client.nearest("layout:0", limit=5)

    assert len(neighbours) == 5
    similarities = [item["cosine_similarity"] for item in neighbours]
    assert all(-1.0001 <= value <= 1.0001 for value in similarities), similarities
    assert similarities == sorted(similarities, reverse=True)


def test_metric_is_optional_and_raw_cosine_still_available(tmp_path):
    """v0 and v1 have no published metric; nearest must still work on them."""
    import pandas as pd

    from squadds.layouts import LayoutEmbeddingClient

    rng = np.random.default_rng(4)
    rows = []
    for index in range(6):
        vector = (rng.normal(size=V2_DIMENSIONS) * 30.0).astype(np.float32)
        rows.append(
            {
                "layout_id": f"layout:{index}",
                "design_id": f"design:{index}",
                "component_name": "GeneralizedCapNInterdigital",
                "source_id": f"campaign/{index}",
                "embedding": vector.tolist(),
            }
        )
    path = tmp_path / "v2.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)

    client = LayoutEmbeddingClient(version="v2", embedding_path=path)
    client._metric = {}  # emulate a revision that predates the published metric
    neighbours = client.nearest("layout:0", limit=3, metric="raw")

    assert len(neighbours) == 3
    assert all(-1.0001 <= item["cosine_similarity"] <= 1.0001 for item in neighbours)
    with pytest.raises(LookupError):
        client.nearest("layout:0", limit=3, metric="whitened")
