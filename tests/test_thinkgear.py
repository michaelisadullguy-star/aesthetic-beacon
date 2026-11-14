import numpy as np
import pytest

from src import thinkgear as tg

# Bytes copied from the hex dump shown in the project description.
PDF_DUMP = (
    "AA AA 04 80 02 00 52 2B AA AA 04 80 02 00 47 36 AA AA 04 80 02 00 3C 41 AA AA 04 80 02 00 40 3D\n"
    "AA AA 04 80 02 FF FD 81 AA AA 04 80 02 FF D8 A6 AA AA 04 80 02 FF F2 8C AA AA 04 80 02 00 25 58\n"
)


def test_decodes_the_recorded_format():
    recording = tg.decode_bytes(tg.hex_text_to_bytes(PDF_DUMP))
    assert recording.raw.tolist() == [82, 71, 60, 64, -3, -40, -14, 37]
    assert recording.stats.checksum_errors == 0
    assert recording.stats.bytes_discarded == 0


def test_round_trip_with_status_packets():
    samples = np.random.default_rng(0).integers(-2000, 2000, 512 * 5)
    stream = tg.encode_stream(samples, 512, poor_signal=[0, 0, 200, 200, 0, 0])
    recording = tg.decode_bytes(stream)
    assert np.array_equal(recording.raw, samples.astype(np.int16))
    assert np.diff(recording.power_positions).tolist() == [512] * 5
    assert recording.poor_signal[0] == 0
    assert recording.poor_signal[512 * 2 + 10] == 200
    assert recording.poor_signal[-1] == 0


def test_chunk_boundaries_do_not_matter():
    samples = np.arange(-700, 700)
    stream = tg.encode_stream(samples, 256)
    for size in (1, 3, 7, 64, 1000):
        parser = tg.ThinkGearParser()
        parts = [parser.feed(stream[i : i + size]).raw for i in range(0, len(stream), size)]
        assert np.array_equal(np.concatenate(parts), samples.astype(np.int16)), size


def test_resynchronises_after_corruption():
    samples = np.arange(2000)
    stream = bytearray(tg.encode_stream(samples, 500))
    stream[801] ^= 0x5A  # damage one packet
    del stream[4000:4003]  # lose three bytes mid-stream
    noisy = b"\x13\x37\xaa" + bytes(stream) + b"\xaa\xaa\x04"
    recording = tg.decode_bytes(noisy)
    assert 1990 <= recording.raw.size <= 1999
    # Every decoded value is a real sample, in order.
    assert np.all(np.diff(recording.raw.astype(int)) >= 1)
    assert recording.stats.bytes_discarded > 0


def test_invalid_length_and_triple_sync():
    good = tg.encode_raw_packets([5, 6]).tobytes()
    stream = b"\xaa\xaa\xfa\x01" + b"\xaa" + good
    assert tg.decode_bytes(stream).raw.tolist() == [5, 6]


@pytest.mark.parametrize(
    "text",
    [
        "AA AA 04 80 02 00 52 2B",
        "aa aa 04 80 02 00 52 2b\r\n",
        "AAAA04800200522B",
        "0xAA, 0xAA, 0x04, 0x80, 0x02, 0x00, 0x52, 0x2B",
        "AA-AA-04-80-02-00-52-2B",
    ],
)
def test_hex_text_layouts(text):
    assert tg.decode_bytes(tg.hex_text_to_bytes(text)).raw.tolist() == [82]


def test_streaming_hex_decoder_matches_one_shot():
    stream = tg.encode_stream(np.arange(-300, 300), 100)
    for text in (tg.bytes_to_hex_text(stream), stream.hex(), tg.bytes_to_hex_text(stream).lower()):
        for size in (1, 5, 97, 4096):
            decoder = tg.HexTextStreamDecoder()
            out = b"".join(decoder.feed(text[i : i + size]) for i in range(0, len(text), size))
            assert out + decoder.flush() == stream


def test_hex_text_layout_matches_logs():
    text = tg.bytes_to_hex_text(tg.encode_raw_packets(range(8)).tobytes())
    first = text.splitlines()[0].split()
    assert len(first) == 32 and first[:5] == ["AA", "AA", "04", "80", "02"]


def test_detects_non_headset_data():
    assert tg.looks_like_thinkgear(tg.encode_stream(np.zeros(100), 50))
    assert not tg.looks_like_thinkgear(np.random.default_rng(1).integers(0, 255, 4000, dtype=np.uint8).tobytes())
    assert not tg.looks_like_thinkgear(b"")
