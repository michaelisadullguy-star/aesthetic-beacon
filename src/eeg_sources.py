"""Where live EEG bytes come from.

Every source yields raw headset bytes in whatever chunks arrive.  A source
never raises because the device went away: it yields an empty chunk while it
waits and reconnects on its own, so the feedback loop keeps running (and the
display keeps its last parameter set) through a dropped connection.

Source specifications accepted by :func:`open_source`:

``serial:/dev/ttyUSB0``  or  ``serial:COM3@57600``
    The headset's serial port (USB dongle or Bluetooth serial port).
``follow:/path/to/log.txt``
    A hex log that another program is writing right now (like ``tail -f``).
``replay:/path/to/recording.txt``
    A finished recording, replayed at real-time speed (``speed`` option: 0 = as
    fast as possible).
``tcp:host:port``
    A TCP socket delivering the raw byte stream.
``sim``  or  ``sim:<seed>``
    The built-in simulator, for trying the loop without hardware.
"""
from __future__ import annotations

import logging
import socket
import time
from pathlib import Path
from typing import Callable, Iterator, Optional

from .eeg_features import estimate_sample_rate
from .thinkgear import HexTextStreamDecoder, decode_bytes, hex_text_to_bytes

logger = logging.getLogger(__name__)

_IDLE_SLEEP = 0.05
_HEX_CHARACTERS = b"0123456789abcdefABCDEFxX,;:.-[] \t\r\n"


class SourceError(RuntimeError):
    """The source specification is unusable (as opposed to a device being away)."""


class EEGSource:
    """Base class of byte sources."""

    #: ``True`` when bytes arrive at the pace the headset produces them, so the
    #: wall clock can be used to measure the sampling rate and detect stalls.
    is_live = True
    description = "source"

    def chunks(self) -> Iterator[bytes]:
        """Yield byte chunks; an empty chunk means "nothing new yet"."""

        raise NotImplementedError

    def close(self) -> None:
        """Release the underlying device."""


def _looks_like_hex_text(data: bytes) -> bool:
    probe = data[:4096]
    return bool(probe) and sum(1 for byte in probe if byte in _HEX_CHARACTERS) >= 0.95 * len(probe)


class SerialSource(EEGSource):
    """Reads the headset from a serial port and reconnects when it disappears."""

    def __init__(self, port: str, baudrate: int = 57600, *, max_backoff: float = 10.0) -> None:
        try:
            import serial

            del serial
        except ImportError as error:
            raise SourceError(
                "Reading from a serial port needs the 'pyserial' package: pip install pyserial"
            ) from error
        self.port = port
        self.baudrate = int(baudrate)
        self.max_backoff = max_backoff
        self.description = f"serial port {port} @ {baudrate} baud"
        self._device = None
        self._closed = False

    def _open(self):
        import serial

        return serial.Serial(self.port, self.baudrate, timeout=0.25)

    def chunks(self) -> Iterator[bytes]:
        import serial

        backoff = 1.0
        next_attempt = 0.0
        announced = False
        while not self._closed:
            if self._device is None:
                if time.monotonic() < next_attempt:
                    time.sleep(_IDLE_SLEEP)
                    yield b""
                    continue
                try:
                    self._device = self._open()
                    logger.info("Connected to %s", self.description)
                    backoff, announced = 1.0, False
                except (serial.SerialException, OSError) as error:
                    if not announced:
                        logger.warning("Cannot open %s (%s); retrying…", self.port, error)
                        announced = True
                    next_attempt = time.monotonic() + backoff
                    backoff = min(backoff * 2.0, self.max_backoff)
                    yield b""
                    continue
            try:
                waiting = self._device.in_waiting
                data = self._device.read(max(1, min(waiting, 65536)))
                yield bytes(data)
            except (serial.SerialException, OSError) as error:
                logger.warning("Lost %s (%s); reconnecting…", self.port, error)
                self._drop()
                next_attempt = time.monotonic() + 1.0
                yield b""

    def _drop(self) -> None:
        if self._device is not None:
            try:
                self._device.close()
            except Exception:  # pragma: no cover - best effort
                pass
            self._device = None

    def close(self) -> None:
        self._closed = True
        self._drop()


class TcpSource(EEGSource):
    """Reads the raw byte stream from a TCP socket and reconnects when it drops."""

    def __init__(self, host: str, port: int, *, max_backoff: float = 10.0) -> None:
        self.host = host
        self.port = int(port)
        self.max_backoff = max_backoff
        self.description = f"tcp://{host}:{port}"
        self._socket: Optional[socket.socket] = None
        self._closed = False

    def chunks(self) -> Iterator[bytes]:
        backoff = 1.0
        next_attempt = 0.0
        announced = False
        while not self._closed:
            if self._socket is None:
                if time.monotonic() < next_attempt:
                    time.sleep(_IDLE_SLEEP)
                    yield b""
                    continue
                try:
                    self._socket = socket.create_connection((self.host, self.port), timeout=2.0)
                    self._socket.settimeout(0.25)
                    logger.info("Connected to %s", self.description)
                    backoff, announced = 1.0, False
                except OSError as error:
                    if not announced:
                        logger.warning("Cannot connect to %s (%s); retrying…", self.description, error)
                        announced = True
                    next_attempt = time.monotonic() + backoff
                    backoff = min(backoff * 2.0, self.max_backoff)
                    yield b""
                    continue
            try:
                data = self._socket.recv(65536)
                if not data:
                    raise ConnectionError("connection closed by the other side")
                yield data
            except socket.timeout:
                yield b""
            except OSError as error:
                logger.warning("Lost %s (%s); reconnecting…", self.description, error)
                self._drop()
                next_attempt = time.monotonic() + 1.0
                yield b""

    def _drop(self) -> None:
        if self._socket is not None:
            try:
                self._socket.close()
            except OSError:  # pragma: no cover - best effort
                pass
            self._socket = None

    def close(self) -> None:
        self._closed = True
        self._drop()


class FollowSource(EEGSource):
    """Follows a log file that a capture program is still writing.

    Works with hexadecimal text as well as raw binary logs and starts at the
    current end of the file unless ``from_start`` is set.  If the file is
    replaced or truncated it is read from the beginning again.
    """

    def __init__(self, path: Path | str, *, from_start: bool = False) -> None:
        self.path = Path(path)
        self.from_start = from_start
        self.description = f"growing log {self.path}"
        self._closed = False

    def chunks(self) -> Iterator[bytes]:
        position: Optional[int] = None
        decoder: Optional[HexTextStreamDecoder] = None
        is_text: Optional[bool] = None
        announced = False
        while not self._closed:
            try:
                size = self.path.stat().st_size
            except OSError:
                if not announced:
                    logger.warning("Waiting for %s to appear…", self.path)
                    announced = True
                time.sleep(0.25)
                yield b""
                continue
            announced = False
            if position is None:
                position = 0 if self.from_start else size
            if size < position:
                logger.info("%s was truncated; reading it from the start.", self.path)
                position, decoder, is_text = 0, None, None
            if size == position:
                time.sleep(_IDLE_SLEEP)
                yield b""
                continue
            try:
                with open(self.path, "rb") as handle:
                    handle.seek(position)
                    data = handle.read(min(size - position, 1 << 20))
            except OSError as error:
                logger.warning("Could not read %s (%s); retrying…", self.path, error)
                time.sleep(0.25)
                yield b""
                continue
            position += len(data)
            if is_text is None:
                is_text = _looks_like_hex_text(data)
                decoder = HexTextStreamDecoder() if is_text else None
            if decoder is not None:
                yield decoder.feed(data.decode("ascii", errors="ignore"))
            else:
                yield data

    def close(self) -> None:
        self._closed = True


class ReplaySource(EEGSource):
    """Replays a finished recording.

    ``speed`` 1.0 reproduces the original timing, larger values play faster and
    0 delivers the data as fast as it can be processed.  With ``loop`` the
    recording restarts when it ends.
    """

    is_live = False

    def __init__(
        self,
        path: Path | str,
        *,
        speed: float = 1.0,
        loop: bool = False,
        nominal_sample_rate: int = 640,
    ) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise SourceError(f"Recording not found: {self.path}")
        self.speed = float(speed)
        self.loop = loop
        self.description = f"replay of {self.path.name}" + ("" if speed == 1.0 else f" (speed {speed:g})")
        data = self.path.read_bytes()
        self._data = hex_text_to_bytes(data.decode("ascii", errors="ignore")) if _looks_like_hex_text(data) else data
        if not self._data:
            raise SourceError(f"{self.path} is empty.")
        decoded = decode_bytes(self._data)
        if decoded.raw.size == 0:
            raise SourceError(f"{self.path} contains no headset samples.")
        rate = estimate_sample_rate(decoded.power_positions) or nominal_sample_rate
        self._bytes_per_second = len(self._data) / (decoded.raw.size / rate)
        self._closed = False

    def chunks(self) -> Iterator[bytes]:
        step = max(64, int(self._bytes_per_second * 0.1))
        while not self._closed:
            started = time.monotonic()
            for index, offset in enumerate(range(0, len(self._data), step)):
                if self._closed:
                    return
                if self.speed > 0:
                    due = started + index * 0.1 / self.speed
                    delay = due - time.monotonic()
                    if delay > 0:
                        time.sleep(delay)
                yield self._data[offset : offset + step]
            if not self.loop:
                return

    def close(self) -> None:
        self._closed = True


class GeneratorSource(EEGSource):
    """Wraps a function producing successive byte chunks (used by the simulator)."""

    is_live = False

    def __init__(self, generator: Callable[[], Iterator[bytes]], description: str = "generator") -> None:
        self._generator = generator
        self.description = description
        self._closed = False

    def chunks(self) -> Iterator[bytes]:
        for chunk in self._generator():
            if self._closed:
                return
            yield chunk

    def close(self) -> None:
        self._closed = True


def open_source(
    specification: str,
    *,
    speed: float = 1.0,
    loop: bool = False,
    nominal_sample_rate: int = 640,
) -> EEGSource:
    """Create a source from a specification string (see the module documentation)."""

    spec = str(specification).strip()
    if not spec:
        raise SourceError("No EEG source given.")
    kind, _, rest = spec.partition(":")
    kind = kind.lower()

    if kind == "serial":
        if not rest:
            raise SourceError("Give the serial port, e.g. serial:/dev/ttyUSB0 or serial:COM3")
        port, _, baud = rest.partition("@")
        return SerialSource(port, int(baud) if baud else 57600)
    if kind == "tcp":
        host, _, port = rest.rpartition(":")
        if not host or not port.isdigit():
            raise SourceError("Give the address as tcp:host:port")
        return TcpSource(host, int(port))
    if kind == "follow":
        if not rest:
            raise SourceError("Give the log file, e.g. follow:/path/to/log.txt")
        return FollowSource(rest)
    if kind == "replay":
        return ReplaySource(rest, speed=speed, loop=loop, nominal_sample_rate=nominal_sample_rate)
    if kind == "sim":
        from .simulator import simulated_source

        return simulated_source(seed=int(rest) if rest else 0, speed=speed)
    if Path(spec).is_file():
        return ReplaySource(spec, speed=speed, loop=loop, nominal_sample_rate=nominal_sample_rate)
    raise SourceError(
        f"Unrecognised EEG source '{specification}'. Use serial:<port>, follow:<file>, "
        "replay:<file>, tcp:<host>:<port> or sim."
    )


__all__ = [
    "EEGSource",
    "FollowSource",
    "GeneratorSource",
    "ReplaySource",
    "SerialSource",
    "SourceError",
    "TcpSource",
    "open_source",
]
