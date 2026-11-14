import json
import math

import numpy as np

from src import parameters as P


def test_fifty_parameters_with_sane_defaults():
    assert len(P.PARAMETER_NAMES) == 50
    assert len(set(P.PARAMETER_NAMES)) == 50
    for spec in P.PARAMETER_SPECS:
        assert spec.low < spec.high
        assert spec.low <= spec.default <= spec.high
        assert spec.name not in P.FIXED_DISPLAY_PARAMETERS


def test_codec_round_trip():
    codec = P.ParameterCodec()
    rng = np.random.default_rng(0)
    for _ in range(20):
        values = {
            s.name: s.clamp(rng.uniform(s.low, s.high)) for s in P.PARAMETER_SPECS
        }
        decoded = codec.decode(codec.encode(values))
        for spec in P.PARAMETER_SPECS:
            if spec.circular:
                difference = abs((decoded[spec.name] - values[spec.name] + 180) % 360 - 180)
                assert difference < 1e-2
            else:
                assert math.isclose(decoded[spec.name], values[spec.name], rel_tol=1e-4, abs_tol=1e-4)


def test_decode_never_leaves_the_safe_range():
    codec = P.ParameterCodec()
    for vector in (
        np.full(codec.dim, 1e9),
        np.full(codec.dim, -1e9),
        np.full(codec.dim, np.nan),
        np.zeros(codec.dim),
        np.random.default_rng(1).normal(0, 50, codec.dim),
    ):
        decoded = codec.decode(vector)
        assert set(decoded) == set(P.PARAMETER_NAMES)
        for spec in P.PARAMETER_SPECS:
            value = decoded[spec.name]
            assert math.isfinite(value)
            assert spec.low <= value <= spec.high
            if spec.integer:
                assert value == int(value)
        assert decoded["cellSize"] >= 12 and decoded["overlayStep"] >= 6


def test_hues_average_around_the_circle():
    codec = P.ParameterCodec()
    a = codec.encode({**P.default_parameters(), "backgroundHue": 350.0})
    b = codec.encode({**P.default_parameters(), "backgroundHue": 10.0})
    mean = codec.decode((a + b) / 2)["backgroundHue"]
    assert min(mean, 360 - mean) < 1.0  # red, not cyan


def test_blending_takes_the_short_way_and_clamps():
    start = {**P.default_parameters(), "backgroundHue": 350.0, "fillAlpha": 20.0}
    target = {**P.default_parameters(), "backgroundHue": 10.0, "fillAlpha": 100.0}
    blended = P.blend_parameters(start, target, 0.5)
    assert min(blended["backgroundHue"], 360 - blended["backgroundHue"]) < 1.0
    assert blended["fillAlpha"] == 60.0
    assert P.blend_parameters(None, {"cellSize": 0, "fillAlpha": 1e6}, 0.5)["cellSize"] == 12


def test_canonicalise_understands_the_existing_formats(repo_root):
    nested = {"cellSize": 500, "canvasWidth": 800, "bogus": 1, "palette": [{"h": 400, "s": 50, "b": 60}]}
    canonical, ignored = P.canonicalise_parameters(nested)
    assert canonical["cellSize"] == 72  # clamped
    assert canonical["palette_1_h"] == 40 and canonical["palette_2_h"] == 130  # one colour spread over four
    assert set(ignored) == {"canvasWidth", "bogus"}
    assert len(canonical) == 50

    example = json.loads((repo_root / "examples" / "parameter_metadata_example.json").read_text())
    for parameters in example.values():
        canonical, _ = P.canonicalise_parameters(parameters)
        assert len(canonical) == 50
        assert canonical["palette_1_s"] == parameters["palette_s"]

    spellings, _ = P.canonicalise_parameters({"palette[0].h": 10, "palette-2 s": 33, "palette_3_b": 44})
    assert (spellings["palette_1_h"], spellings["palette_2_s"], spellings["palette_3_b"]) == (10, 33, 44)


def test_sketch_json_layout():
    document = P.to_sketch_json(P.default_parameters())
    assert document["canvasWidth"] == 1080 and document["canvasHeight"] == 640
    assert len(document["palette"]) == 4 and set(document["palette"][0]) == {"h", "s", "b"}
    assert isinstance(document["cellSize"], int)
    json.dumps(document)


def test_sketch_data_files_are_in_step_with_the_schema(repo_root):
    """The Processing sketch reads these files; regenerate them with `python -m src.parameters`."""

    data = repo_root / "scripts" / "processing_templates" / "dynamic_texture_osc" / "data"
    assert json.loads((data / "parameter_schema.json").read_text()) == P.schema_document()
    assert json.loads((data / "dynamic_texture_params.json").read_text()) == P.to_sketch_json(P.default_parameters())
