import json

import numpy as np
import pytest
from PIL import Image

from src import parameters as P
from src import processing_features as pf
from src.brainwave_dataset import BrainwaveProcessingDataset, load_parameter_metadata
from src.simulator import Segment, write_hex_recording


def _flat(colour, size=(120, 90)):
    return Image.new("RGB", size, colour)


def test_analysis_is_deterministic_and_in_range():
    rng = np.random.default_rng(0)
    image = Image.fromarray(rng.integers(0, 255, (90, 120, 3), dtype=np.uint8))
    first = pf.analyse_image(image)
    assert first == pf.analyse_image(image)
    assert list(first) == list(P.PARAMETER_NAMES)
    for spec in P.PARAMETER_SPECS:
        assert spec.low <= first[spec.name] <= spec.high


def test_palette_follows_the_image_colours():
    image = _flat((20, 20, 200))
    pixels = np.asarray(image).copy()
    pixels[:30, :40] = (230, 30, 30)
    parameters = pf.analyse_image(Image.fromarray(pixels))
    assert abs(parameters["backgroundHue"] - 240) < 8  # the large blue area
    hue = parameters["palette_1_h"]
    assert min(hue, 360 - hue) < 8  # the red block
    assert parameters["palette_1_s"] > 70


def test_texture_statistics_respond_to_texture():
    rng = np.random.default_rng(1)
    fine = Image.fromarray(rng.integers(0, 255, (180, 240, 3), dtype=np.uint8))
    coarse = Image.fromarray(np.kron(rng.integers(0, 255, (3, 4, 3), dtype=np.uint8), np.ones((60, 60, 1), dtype=np.uint8)))
    fine_parameters, coarse_parameters = pf.analyse_image(fine), pf.analyse_image(coarse)
    assert fine_parameters["cellSize"] < coarse_parameters["cellSize"]
    assert fine_parameters["noiseScale"] > coarse_parameters["noiseScale"]
    assert pf.analyse_image(_flat((128, 128, 128)))["textureDepth"] < fine_parameters["textureDepth"]


def test_degenerate_images_do_not_break_analysis():
    for image in (_flat((0, 0, 0)), _flat((255, 255, 255)), _flat((10, 200, 30), size=(4, 4))):
        parameters = pf.analyse_image(image)
        assert all(np.isfinite(v) for v in parameters.values())


def _dataset(tmp_path, with_metadata):
    root = tmp_path / "step5"
    (root / "artworks").mkdir(parents=True)
    for index, colour in enumerate([(200, 30, 30), (30, 30, 200), (30, 180, 60)]):
        _flat(colour).save(root / "artworks" / f"art{index}.png")
        for viewer in ("anna", "ben"):
            write_hex_recording(root / "brainwaves" / viewer / f"art{index}.txt", [Segment(10.0)], seed=index * 7 + len(viewer))
    if with_metadata:
        (root / "metadata.json").write_text(
            json.dumps({"art0.png": {"cellSize": 60, "palette": [{"h": 1, "s": 2, "b": 3}]}, "art1": {"cellSize": 20}})
        )
    return root


def test_dataset_layout_and_middle_windows(tmp_path):
    dataset = BrainwaveProcessingDataset(_dataset(tmp_path, False))
    assert len(dataset) == 6 * 5 and dataset.eeg.shape[1:] == (8, 3200)
    assert dataset.targets.shape == (30, P.ParameterCodec().dim)
    assert set(dataset.parameter_sources.values()) == {"analysis"}
    starts = sorted({sample.start_sample for sample in dataset.samples})
    assert starts[0] == 640 * 1.5 and starts[-1] == 640 * 3.5  # around the middle five seconds
    assert len(set(dataset.groups)) == 6


def test_dataset_prefers_curated_metadata(tmp_path):
    dataset = BrainwaveProcessingDataset(_dataset(tmp_path, True))
    assert dataset.parameter_sources == {"art0": "metadata", "art1": "metadata", "art2": "analysis"}
    assert dataset.artwork_parameters["art0"]["cellSize"] == 60
    assert dataset.artwork_parameters["art0"]["palette_1_h"] == 1
    assert load_parameter_metadata(tmp_path / "step5" / "metadata.json").keys() == {"art0", "art1"}


def test_dataset_reports_layout_problems(tmp_path):
    root = _dataset(tmp_path, False)
    write_hex_recording(root / "brainwaves" / "anna" / "stray.txt", [Segment(6.0)], seed=1)
    with pytest.raises(FileNotFoundError, match="stray"):
        BrainwaveProcessingDataset(root)
    (root / "brainwaves" / "anna" / "stray.txt").unlink()
    write_hex_recording(root / "brainwaves" / "anna" / "art0__again.txt", [Segment(2.0)], seed=1)
    with pytest.raises(ValueError, match="at least"):
        BrainwaveProcessingDataset(root)


def test_artwork_cnn_trains_and_predicts(tmp_path):
    from src import train_processing_features as tpf

    root = tmp_path / "art"
    (root / "images").mkdir(parents=True)
    metadata = {}
    for index in range(6):
        _flat((40 * index, 30, 200 - 30 * index), size=(64 + 8 * index, 48)).save(root / "images" / f"a{index}.png")
        metadata[f"a{index}"] = {"cellSize": 12 + 10 * index, "backgroundHue": 60 * index}
    (root / "metadata.json").write_text(json.dumps(metadata))
    weights = tmp_path / "cnn.pt"
    assert tpf.main([str(root), "--output", str(weights), "--epochs", "3", "--resize-long-edge", "64", "--device", "cpu"]) == 0
    parameters = pf.compute_processing_parameters(root / "images" / "a0.png", weights_path=weights)
    assert list(parameters) == list(P.PARAMETER_NAMES)
    resolved, source = pf.resolve_artwork_parameters(root / "images" / "a0.png", weights_path=weights)
    assert source == "cnn" and resolved == parameters
