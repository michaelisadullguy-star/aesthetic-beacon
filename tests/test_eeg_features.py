import numpy as np
import pytest

from src import eeg_features as ef
from src import thinkgear as tg
from src.simulator import Segment, simulate_raw, simulate_stream


def _tone(frequency, seconds=20, rate=640, amplitude=50.0):
    t = np.arange(int(seconds * rate)) / rate
    return amplitude * np.sin(2 * np.pi * frequency * t)


@pytest.mark.parametrize("frequency,band", [(2.0, 0), (6.0, 1), (9.0, 2), (11.5, 3), (16.0, 4), (25.0, 5), (37.0, 6), (54.0, 7)])
def test_each_band_isolates_its_frequencies(frequency, band):
    envelopes = ef.band_envelopes(_tone(frequency) + np.random.default_rng(0).normal(0, 1, 12800))
    means = envelopes[:, 640:-640].mean(axis=1)
    assert int(np.argmax(means)) == band
    assert means[band] == pytest.approx(50.0, rel=0.15)


def test_mains_hum_is_removed():
    hum = _tone(50.0, amplitude=40.0)
    with_notch = ef.band_envelopes(hum)[7].mean()
    without = ef.band_envelopes(hum, ef.FeatureConfig(notch_hz=None))[7].mean()
    assert with_notch < 0.1 * without


def test_window_uses_only_past_samples():
    rng = np.random.default_rng(1)
    signal = rng.normal(0, 20, 640 * 30)
    end = 640 * 12
    reference = ef.extract_band_window(signal, end)
    altered = signal.copy()
    altered[end:] = 500.0  # the future must not influence the window
    assert np.array_equal(reference, ef.extract_band_window(altered, end))
    assert reference.shape == (8, 3200)


def test_live_and_offline_windows_are_identical():
    """A window cut from a recording equals what the live loop computes from its buffer."""

    signal = simulate_raw([Segment(30.0)], seed=3).astype(np.float32)
    config = ef.DEFAULT_FEATURE_CONFIG
    end = 640 * 21
    offline = ef.Recording(kind="raw", config=config, raw=signal).window(end - 3200)
    live_buffer = signal[end - 3200 - config.context_samples : end]
    live = ef.band_envelopes(live_buffer, config)[:, -3200:]
    assert np.allclose(offline, live, rtol=1e-5, atol=1e-5)


def test_sample_rate_is_measured_and_resampled():
    tone = _tone(11.5, seconds=20, rate=512)
    recording = ef.recording_from_stream(tg.encode_stream(tone.round(), 512))
    assert recording.info["measured_sample_rate"] == 512
    assert recording.info["resampled"] is True
    assert recording.n_samples == 20 * 640
    means = recording.band_matrix()[:, 640:-640].mean(axis=1)
    assert int(np.argmax(means)) == 3  # still 11.5 Hz after resampling


def test_wrong_assumed_rate_would_shift_the_bands():
    """Why the rate is measured: treating 512 Hz data as 640 Hz moves 11.5 Hz to 14.4 Hz."""

    tone = _tone(11.5, seconds=20, rate=512)
    misread = ef.band_envelopes(tone)[:, 640:-640].mean(axis=1)
    assert int(np.argmax(misread)) == 4


def test_rate_without_status_packets_falls_back_to_configured():
    stream = tg.encode_stream(np.zeros(3000) + np.arange(3000) % 7, 640, include_power_packets=False)
    recording = ef.recording_from_stream(stream)
    assert recording.info["measured_sample_rate"] is None
    assert recording.info["device_sample_rate"] == 640


def test_quality_gate():
    rng = np.random.default_rng(2)
    clean = rng.normal(0, 20, 3200)
    assert ef.assess_window(clean).ok
    assert ef.assess_window(clean, np.full(3200, 200)).reason == "poor_contact"
    assert ef.assess_window(np.zeros(3200)).reason == "flatline"
    clipped = clean.copy()
    clipped[:400] = 2047
    assert ef.assess_window(clipped).reason == "clipping"
    assert ef.assess_window(rng.normal(0, 600, 3200), reference_sigma=20.0).reason == "artifact"
    assert ef.assess_window(np.full(3200, np.nan)).reason == "non_finite"
    # A couple of eye blinks must not disqualify a window.
    blinks = clean.copy()
    blinks[500:700] += 250
    blinks[2000:2200] += 250
    assert ef.assess_window(blinks, reference_sigma=20.0).ok


def test_normalisation_removes_gain_but_keeps_state():
    quiet = simulate_raw([Segment(60.0, "rest"), Segment(60.0, "creatable")], seed=5).astype(np.float32)
    loud = quiet * 3.0  # e.g. better electrode contact
    config = ef.DEFAULT_FEATURE_CONFIG
    a = ef.Recording(kind="raw", config=config, raw=quiet)
    b = ef.Recording(kind="raw", config=config, raw=loud)
    start = 640 * 90
    difference = np.abs(a.normalised_window(start) - b.normalised_window(start))
    assert np.percentile(difference, 99.9) < 0.05  # identical except where an envelope touches zero
    rest = a.normalised_window(640 * 20)[3].mean()
    active = a.normalised_window(start)[3].mean()
    assert active - rest > 0.8  # alpha2 rises in the simulated creatable state


def test_normalisation_handles_legacy_signed_data_and_rejects_nan():
    signed = np.random.default_rng(0).normal(0, 1, (8, 3200)).astype(np.float32)
    out = ef.normalise_bands(signed)
    assert np.allclose(out.mean(axis=1), 0, atol=1e-4) and np.allclose(out.std(axis=1), 1, atol=1e-3)
    signed[0, 0] = np.nan
    with pytest.raises(ValueError):
        ef.normalise_bands(signed)


def test_running_baseline_matches_tracker():
    signal = simulate_raw([Segment(100.0)], seed=8).astype(np.float32)
    config = ef.DEFAULT_FEATURE_CONFIG
    recording = ef.Recording(kind="raw", config=config, raw=signal)
    tracker = ef.BaselineTracker(config, history_seconds=60.0, min_seconds=20.0)
    bands = recording.band_matrix()
    for start in range(0, 640 * 80, 3200):
        tracker.update(bands[:, start : start + 3200])
    assert tracker.ready
    live = tracker.baseline()
    offline = recording.running_baseline(640 * 80, history_seconds=60.0, min_seconds=20.0)
    assert np.allclose(live.mean, offline.mean, atol=1e-6)
    assert np.allclose(live.std, offline.std, atol=1e-6)


def test_load_recording_variants(tmp_path):
    stream = simulate_stream([Segment(8.0)], seed=1)
    text = tmp_path / "a.txt"
    text.write_text(tg.bytes_to_hex_text(stream))
    binary = tmp_path / "b.bin"
    binary.write_bytes(stream)
    from_text = ef.load_recording(text)
    from_binary = ef.load_recording(binary)
    assert np.array_equal(from_text.raw, from_binary.raw)
    assert from_text.duration == pytest.approx(8.0)

    np.save(tmp_path / "bands.npy", np.abs(np.random.default_rng(0).normal(5, 1, (3200, 8))).astype(np.float32))
    assert ef.load_recording(tmp_path / "bands.npy").band_matrix().shape == (8, 3200)

    with pytest.raises(ef.RecordingError):
        ef.load_recording(tmp_path / "missing.txt")
    (tmp_path / "empty.txt").write_text("")
    with pytest.raises(ef.RecordingError):
        ef.load_recording(tmp_path / "empty.txt")
    np.save(tmp_path / "wrong.npy", np.zeros((5, 100), dtype=np.float32))
    with pytest.raises(ef.RecordingError):
        ef.load_recording(tmp_path / "wrong.npy")
