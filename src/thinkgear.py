"""Decoding of the headset's serial stream (NeuroSky ThinkGear protocol).

The single-electrode forehead headset used by Aesthetic Beacon speaks the
ThinkGear protocol.  Recordings are stored as hexadecimal plain text::

    AA AA 04 80 02 00 52 2B AA AA 04 80 02 00 47 36 ...

Every packet is ``AA AA <length> <payload...> <checksum>`` where the checksum is
the inverted low byte of the payload sum.  The payload is a sequence of data
rows; the ones that matter here are:

``0x80``  raw wave sample, two bytes, big-endian signed (one per sample tick)
``0x02``  poor-signal quality, one byte (0 = good contact, 200 = off the head)
``0x83``  ASIC band powers, sent once per second together with ``0x02``

This module has no dependencies beyond NumPy so that recordings can be decoded
and checked without the machine-learning stack.  The parser is incremental: it
can be fed arbitrary chunks from a serial port and resynchronises after
corrupted bytes instead of trusting the stream.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

SYNC = 0xAA
EXCODE = 0x55
MAX_PAYLOAD_LENGTH = 169

CODE_POOR_SIGNAL = 0x02
CODE_ATTENTION = 0x04
CODE_MEDITATION = 0x05
CODE_BLINK = 0x16
CODE_RAW = 0x80
CODE_ASIC_EEG_POWER = 0x83

#: Value used in ``poor_signal`` arrays before the headset reported its contact quality.
POOR_SIGNAL_UNKNOWN = -1

_RAW_PACKET_LENGTH = 8  # AA AA 04 80 02 hi lo checksum
_BULK_LIMIT = 8192  # raw packets examined per vectorised pass


@dataclass
class ParserStats:
    """Running counters describing the health of a decoded stream."""

    bytes_in: int = 0
    packets: int = 0
    raw_samples: int = 0
    power_packets: int = 0
    checksum_errors: int = 0
    bytes_discarded: int = 0

    def as_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class ParsedChunk:
    """Everything decoded from one call to :meth:`ThinkGearParser.feed`."""

    raw: np.ndarray  # int16, shape (k,)
    poor_signal: np.ndarray  # int16, shape (k,), POOR_SIGNAL_UNKNOWN until reported
    #: Absolute sample index at which each once-per-second power packet arrived.
    power_positions: List[int] = field(default_factory=list)
    #: ``(absolute sample index, code, value)`` for attention / meditation / blink rows.
    esense: List[Tuple[int, int, int]] = field(default_factory=list)


@dataclass
class ThinkGearRecording:
    """A fully decoded recording."""

    raw: np.ndarray
    poor_signal: np.ndarray
    power_positions: List[int]
    esense: List[Tuple[int, int, int]]
    stats: ParserStats


class ThinkGearParser:
    """Incremental, self-resynchronising ThinkGear packet parser."""

    def __init__(self) -> None:
        self._buffer = bytearray()
        self._poor_signal = POOR_SIGNAL_UNKNOWN
        self.total_samples = 0
        self.stats = ParserStats()

    @property
    def last_poor_signal(self) -> int:
        """Most recent contact-quality value (``POOR_SIGNAL_UNKNOWN`` if none yet)."""

        return self._poor_signal

    def feed(self, data: bytes | bytearray | memoryview) -> ParsedChunk:
        """Consume ``data`` and return the samples completed by it.

        Incomplete trailing packets are kept until the next call, so the input
        may be split at arbitrary byte boundaries.
        """

        buf = self._buffer
        buf += data
        self.stats.bytes_in += len(data)

        raw_parts: List[np.ndarray] = []
        poor_parts: List[np.ndarray] = []
        pending_raw: List[int] = []
        pending_poor: List[int] = []
        power_positions: List[int] = []
        esense: List[Tuple[int, int, int]] = []

        def flush_pending() -> None:
            if pending_raw:
                raw_parts.append(np.asarray(pending_raw, dtype=np.int16))
                poor_parts.append(np.asarray(pending_poor, dtype=np.int16))
                pending_raw.clear()
                pending_poor.clear()

        n = len(buf)
        i = 0
        while True:
            # Fast path: runs of back-to-back raw-wave packets make up >99% of the stream.
            available = (n - i) // _RAW_PACKET_LENGTH
            if available > 0 and buf[i] == SYNC and buf[i + 1] == SYNC and buf[i + 2] == 0x04:
                count = min(available, _BULK_LIMIT)
                block = np.frombuffer(
                    bytes(buf[i : i + count * _RAW_PACKET_LENGTH]), dtype=np.uint8
                ).reshape(count, _RAW_PACKET_LENGTH)
                checksum = ((0x82 + block[:, 5].astype(np.uint16) + block[:, 6]) & 0xFF) ^ 0xFF
                valid = (
                    (block[:, 0] == SYNC)
                    & (block[:, 1] == SYNC)
                    & (block[:, 2] == 0x04)
                    & (block[:, 3] == CODE_RAW)
                    & (block[:, 4] == 0x02)
                    & (checksum == block[:, 7])
                )
                run = count if bool(valid.all()) else int(np.argmin(valid))
                if run > 0:
                    flush_pending()
                    values = (block[:run, 5].astype(np.uint16) << 8) | block[:run, 6]
                    raw_parts.append(values.astype(np.int16))
                    poor_parts.append(np.full(run, self._poor_signal, dtype=np.int16))
                    self.total_samples += run
                    self.stats.packets += run
                    self.stats.raw_samples += run
                    i += run * _RAW_PACKET_LENGTH
                    continue

            # General path: locate the next sync marker and decode one packet.
            start = buf.find(b"\xaa\xaa", i)
            if start < 0:
                keep = 1 if n > i and buf[n - 1] == SYNC else 0
                self.stats.bytes_discarded += n - i - keep
                i = n - keep
                break
            self.stats.bytes_discarded += start - i
            i = start
            if i + 2 >= n:
                break
            length = buf[i + 2]
            if length == SYNC:
                # Three sync bytes in a row: the packet starts one byte later.
                i += 1
                self.stats.bytes_discarded += 1
                continue
            if length > MAX_PAYLOAD_LENGTH:
                i += 2
                self.stats.bytes_discarded += 2
                continue
            end = i + 3 + length
            if end >= n:
                break
            payload = bytes(buf[i + 3 : end])
            if ((~sum(payload)) & 0xFF) != buf[end]:
                self.stats.checksum_errors += 1
                self.stats.bytes_discarded += 1
                i += 1
                continue

            self.stats.packets += 1
            self._parse_payload(payload, pending_raw, pending_poor, power_positions, esense)
            i = end + 1

        del buf[:i]
        flush_pending()

        if raw_parts:
            raw = np.concatenate(raw_parts)
            poor = np.concatenate(poor_parts)
        else:
            raw = np.empty(0, dtype=np.int16)
            poor = np.empty(0, dtype=np.int16)
        return ParsedChunk(raw=raw, poor_signal=poor, power_positions=power_positions, esense=esense)

    def _parse_payload(
        self,
        payload: bytes,
        pending_raw: List[int],
        pending_poor: List[int],
        power_positions: List[int],
        esense: List[Tuple[int, int, int]],
    ) -> None:
        length = len(payload)
        i = 0
        while i < length:
            while i < length and payload[i] == EXCODE:
                i += 1
            if i >= length:
                break
            code = payload[i]
            i += 1
            if code >= 0x80:
                if i >= length:
                    break
                value_length = payload[i]
                i += 1
            else:
                value_length = 1
            if i + value_length > length:
                break  # truncated row: ignore the remainder of this packet
            value = payload[i : i + value_length]
            i += value_length

            if code == CODE_RAW and value_length == 2:
                pending_raw.append(int.from_bytes(value, "big", signed=True))
                pending_poor.append(self._poor_signal)
                self.total_samples += 1
                self.stats.raw_samples += 1
            elif code == CODE_POOR_SIGNAL:
                self._poor_signal = value[0]
            elif code == CODE_ASIC_EEG_POWER:
                power_positions.append(self.total_samples)
                self.stats.power_packets += 1
            elif code in (CODE_ATTENTION, CODE_MEDITATION, CODE_BLINK):
                esense.append((self.total_samples, code, value[0]))


def decode_bytes(data: bytes | bytearray) -> ThinkGearRecording:
    """Decode a complete ThinkGear byte stream."""

    parser = ThinkGearParser()
    chunk = parser.feed(data)
    return ThinkGearRecording(
        raw=chunk.raw,
        poor_signal=chunk.poor_signal,
        power_positions=chunk.power_positions,
        esense=chunk.esense,
        stats=parser.stats,
    )


def looks_like_thinkgear(data: bytes | bytearray, *, probe_bytes: int = 65536) -> bool:
    """Return ``True`` when ``data`` is predominantly valid ThinkGear packets."""

    probe = bytes(data[:probe_bytes])
    if len(probe) < 2 * _RAW_PACKET_LENGTH:
        return False
    parser = ThinkGearParser()
    parser.feed(probe)
    stats = parser.stats
    if stats.packets < 2:
        return False
    return stats.bytes_discarded <= 0.5 * len(probe)


# ---------------------------------------------------------------------------
# Hexadecimal text <-> bytes
# ---------------------------------------------------------------------------

_HEX_PREFIX = re.compile(r"0[xX]")
_HEX_PAIR = re.compile(r"[0-9A-Fa-f]{2}")
_TRAILING_TOKEN = re.compile(r"[0-9A-Fa-fxX]*$")


def hex_text_to_bytes(text: str | bytes) -> bytes:
    """Convert a hexadecimal text dump into bytes.

    Accepts the layouts produced by common serial tools: space or newline
    separated byte pairs (``AA AA 04``), a continuous stream (``AAAA04``),
    ``0x`` prefixes and comma separators.  Stray non-hex text (timestamps,
    labels) yields a few garbage bytes which the packet parser then skips.
    """

    if isinstance(text, (bytes, bytearray)):
        text = bytes(text).decode("ascii", errors="ignore")
    try:
        return bytes.fromhex(text)  # fast path: plain pairs separated by whitespace
    except ValueError:
        pass
    cleaned = _HEX_PREFIX.sub(" ", text)
    return bytes.fromhex("".join(_HEX_PAIR.findall(cleaned)))


class HexTextStreamDecoder:
    """Incremental :func:`hex_text_to_bytes` for text that arrives in pieces."""

    def __init__(self) -> None:
        self._carry = ""

    def feed(self, text: str) -> bytes:
        text = self._carry + text
        # Hold back an unfinished token: trailing hex digits may continue in the next chunk.
        match = _TRAILING_TOKEN.search(text)
        cut = match.start() if match else len(text)
        tail = text[cut:]
        if len(tail) > 64 and "x" not in tail and "X" not in tail:
            # A long unbroken run means the dump has no separators: emit all complete pairs.
            cut = len(text) - (len(tail) % 2)
        self._carry = text[cut:]
        head = text[:cut]
        return hex_text_to_bytes(head) if head else b""

    def flush(self) -> bytes:
        tail, self._carry = self._carry, ""
        return hex_text_to_bytes(tail) if tail else b""


def bytes_to_hex_text(data: bytes | bytearray, *, bytes_per_line: int = 32) -> str:
    """Format bytes the way the headset logs are stored (``AA AA 04 80 ...``)."""

    hex_string = bytes(data).hex().upper()
    pairs = [hex_string[i : i + 2] for i in range(0, len(hex_string), 2)]
    lines = [
        " ".join(pairs[i : i + bytes_per_line]) for i in range(0, len(pairs), bytes_per_line)
    ]
    return "\n".join(lines) + ("\n" if lines else "")


# ---------------------------------------------------------------------------
# Encoding (used by the simulator and the tests)
# ---------------------------------------------------------------------------


def encode_raw_packets(samples: Sequence[int] | np.ndarray) -> np.ndarray:
    """Encode raw-wave samples as an ``(n, 8)`` array of packet bytes."""

    values = np.clip(np.asarray(samples), -32768, 32767).astype(np.int16).view(np.uint16)
    packets = np.empty((values.size, _RAW_PACKET_LENGTH), dtype=np.uint8)
    packets[:, 0] = SYNC
    packets[:, 1] = SYNC
    packets[:, 2] = 0x04
    packets[:, 3] = CODE_RAW
    packets[:, 4] = 0x02
    packets[:, 5] = (values >> 8).astype(np.uint8)
    packets[:, 6] = (values & 0xFF).astype(np.uint8)
    packets[:, 7] = ((0x82 + packets[:, 5].astype(np.uint16) + packets[:, 6]) & 0xFF) ^ 0xFF
    return packets


def encode_power_packet(
    poor_signal: int = 0,
    band_powers: Optional[Sequence[int]] = None,
    attention: int = 0,
    meditation: int = 0,
) -> bytes:
    """Encode the once-per-second status packet (contact quality + band powers)."""

    powers = list(band_powers) if band_powers is not None else [0] * 8
    if len(powers) != 8:
        raise ValueError("band_powers must contain exactly eight values")
    payload = bytearray([CODE_POOR_SIGNAL, int(poor_signal) & 0xFF, CODE_ASIC_EEG_POWER, 24])
    for power in powers:
        payload += int(max(0, min(int(power), 0xFFFFFF))).to_bytes(3, "big")
    payload += bytes([CODE_ATTENTION, int(attention) & 0xFF, CODE_MEDITATION, int(meditation) & 0xFF])
    checksum = (~sum(payload)) & 0xFF
    return bytes([SYNC, SYNC, len(payload)]) + bytes(payload) + bytes([checksum])


def encode_stream(
    samples: Sequence[int] | np.ndarray,
    sample_rate: int,
    *,
    poor_signal: int | Sequence[int] = 0,
    include_power_packets: bool = True,
) -> bytes:
    """Encode samples as a headset would transmit them.

    ``poor_signal`` is either one value for the whole stream or one value per
    second of signal.
    """

    packets = encode_raw_packets(samples)
    if not include_power_packets:
        return packets.tobytes()

    total = packets.shape[0]
    seconds = total // sample_rate
    if np.isscalar(poor_signal):
        quality = [int(poor_signal)] * (seconds + 1)
    else:
        quality = [int(v) for v in poor_signal]
        if not quality:
            quality = [0]
        while len(quality) < seconds + 1:
            quality.append(quality[-1])

    out = bytearray()
    out += encode_power_packet(quality[0])
    for second in range(seconds):
        out += packets[second * sample_rate : (second + 1) * sample_rate].tobytes()
        out += encode_power_packet(quality[second + 1])
    out += packets[seconds * sample_rate :].tobytes()
    return bytes(out)


__all__ = [
    "CODE_ASIC_EEG_POWER",
    "CODE_POOR_SIGNAL",
    "CODE_RAW",
    "HexTextStreamDecoder",
    "POOR_SIGNAL_UNKNOWN",
    "ParsedChunk",
    "ParserStats",
    "ThinkGearParser",
    "ThinkGearRecording",
    "bytes_to_hex_text",
    "decode_bytes",
    "encode_power_packet",
    "encode_raw_packets",
    "encode_stream",
    "hex_text_to_bytes",
    "looks_like_thinkgear",
]
