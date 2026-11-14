"""Sending parameters from Python to the Processing sketch.

Two transports are used side by side:

* **OSC over UDP** — immediate.  A complete parameter set travels in a single
  message, so the sketch never shows a half-applied update::

      /aesthetic_beacon/params  <sequence:int> <name:str> <value> <name:str> <value> ...
      /aesthetic_beacon/state   <creatable:int> <confidence:float> <label:str>

* **A JSON file** — written atomically on every update, in the layout the
  sketch loads.  The sketch watches this file, so the display keeps working if
  UDP is blocked, and the file doubles as a record of what is being shown.

OSC encoding is implemented here directly; no extra package is required.
"""
from __future__ import annotations

import json
import logging
import os
import socket
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .parameters import to_sketch_json, typed_items

logger = logging.getLogger(__name__)

DEFAULT_JSON_PATH = Path("data/step6_realtime/outputs/current_parameters.json")


@dataclass
class OSCConfig:
    """Configuration for OSC communication."""

    host: str = "127.0.0.1"
    port: int = 12000
    address_prefix: str = "/aesthetic_beacon"
    #: Kept for compatibility: a parameter set is always sent as one message.
    batch_send: bool = True
    #: Also write every parameter set to the JSON file.
    fallback_to_json: bool = True


# ---------------------------------------------------------------------------
# Minimal OSC 1.0 encoding / decoding
# ---------------------------------------------------------------------------


def _pad(data: bytes) -> bytes:
    return data + b"\x00" * (4 - len(data) % 4)


def encode_osc_message(address: str, arguments: Sequence[Any] = ()) -> bytes:
    """Encode an OSC message with int32, float32 and string arguments."""

    tags = ","
    body = b""
    for argument in arguments:
        if isinstance(argument, bool):
            tags += "i"
            body += struct.pack(">i", int(argument))
        elif isinstance(argument, int):
            tags += "i"
            body += struct.pack(">i", max(-(2**31), min(2**31 - 1, argument)))
        elif isinstance(argument, float):
            tags += "f"
            body += struct.pack(">f", argument)
        elif isinstance(argument, str):
            tags += "s"
            body += _pad(argument.encode("utf8"))
        else:
            raise TypeError(f"Unsupported OSC argument type: {type(argument).__name__}")
    return _pad(address.encode("ascii")) + _pad(tags.encode("ascii")) + body


def decode_osc_message(packet: bytes) -> Tuple[str, List[Any]]:
    """Decode a message produced by :func:`encode_osc_message`."""

    def read_string(offset: int) -> Tuple[str, int]:
        end = packet.index(b"\x00", offset)
        text = packet[offset:end].decode("utf8")
        return text, offset + (len(packet[offset:end]) // 4 + 1) * 4

    address, offset = read_string(0)
    tags, offset = read_string(offset)
    arguments: List[Any] = []
    for tag in tags[1:]:
        if tag == "i":
            arguments.append(struct.unpack_from(">i", packet, offset)[0])
            offset += 4
        elif tag == "f":
            arguments.append(struct.unpack_from(">f", packet, offset)[0])
            offset += 4
        elif tag == "s":
            text, offset = read_string(offset)
            arguments.append(text)
        else:
            raise ValueError(f"Unsupported OSC type tag '{tag}'.")
    return address, arguments


def _write_json_atomically(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", encoding="utf8") as handle:
        json.dump(document, handle, indent=2)
    os.replace(temporary, path)


# ---------------------------------------------------------------------------
# Bridges
# ---------------------------------------------------------------------------


class JSONOnlyBridge:
    """Writes parameter sets to the JSON file the sketch watches."""

    def __init__(self, json_path: Optional[Path | str] = None, **_ignored: Any) -> None:
        self.json_path = Path(json_path) if json_path else DEFAULT_JSON_PATH
        self.json_path.parent.mkdir(parents=True, exist_ok=True)
        self.sequence = 0
        self._state: Dict[str, Any] = {"creatable": False, "confidence": 0.0, "label": "waiting"}
        self._parameters: Optional[Dict[str, float]] = None
        self._json_failed = False

    # -- public API -------------------------------------------------------------
    def send_parameters(self, parameters: Mapping[str, Any], timestamp: Optional[float] = None) -> bool:
        """Publish a parameter set. Returns ``True`` if at least one transport succeeded."""

        self.sequence += 1
        self._parameters = {name: value for name, value in typed_items(parameters)}
        return self._publish(timestamp)

    def send_state_change(
        self,
        is_creatable: bool,
        confidence: float = 1.0,
        label: Optional[str] = None,
    ) -> bool:
        """Publish the creator state (shown in the sketch's status overlay)."""

        self._state = {
            "creatable": bool(is_creatable),
            "confidence": round(float(confidence), 4),
            "label": label or ("creatable" if is_creatable else "idle"),
        }
        return True

    def close(self) -> None:
        """Release resources."""

    # -- internals --------------------------------------------------------------
    def _publish(self, timestamp: Optional[float]) -> bool:
        return self._write_json(timestamp)

    def _write_json(self, timestamp: Optional[float]) -> bool:
        if self._parameters is None:
            return False
        document: Dict[str, Any] = to_sketch_json(self._parameters)
        document["sequence"] = self.sequence
        document["timestamp"] = time.time() if timestamp is None else timestamp
        document["state"] = dict(self._state)
        try:
            _write_json_atomically(self.json_path, document)
            self._json_failed = False
            return True
        except OSError as error:
            if not self._json_failed:
                logger.error("Could not write %s: %s", self.json_path, error)
            self._json_failed = True
            return False


class ProcessingOSCBridge(JSONOnlyBridge):
    """Sends parameters to the Processing sketch via OSC and mirrors them to JSON.

    Usage::

        bridge = ProcessingOSCBridge()
        bridge.send_parameters({"noiseScale": 0.015, "cellSize": 32, ...})
    """

    def __init__(self, config: Optional[OSCConfig] = None, json_path: Optional[Path | str] = None) -> None:
        super().__init__(json_path)
        self.config = config or OSCConfig()
        self._socket: Optional[socket.socket] = None
        self._osc_failed = False
        try:
            self._address = (socket.gethostbyname(self.config.host), int(self.config.port))
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            logger.info("Sending OSC to %s:%d", self.config.host, self.config.port)
        except OSError as error:
            logger.error(
                "OSC is unavailable (%s:%s — %s); parameters are written to %s only.",
                self.config.host,
                self.config.port,
                error,
                self.json_path,
            )

    def send_state_change(
        self,
        is_creatable: bool,
        confidence: float = 1.0,
        label: Optional[str] = None,
    ) -> bool:
        super().send_state_change(is_creatable, confidence, label)
        return self._send(
            f"{self.config.address_prefix}/state",
            [int(bool(is_creatable)), float(confidence), str(self._state["label"])],
        )

    def _publish(self, timestamp: Optional[float]) -> bool:
        assert self._parameters is not None
        arguments: List[Any] = [self.sequence]
        for name, value in self._parameters.items():
            arguments.append(name)
            arguments.append(value)
        sent = self._send(f"{self.config.address_prefix}/params", arguments)
        written = self._write_json(timestamp) if self.config.fallback_to_json else False
        return sent or written

    def _send(self, address: str, arguments: Sequence[Any]) -> bool:
        if self._socket is None:
            return False
        try:
            self._socket.sendto(encode_osc_message(address, arguments), self._address)
            self._osc_failed = False
            return True
        except OSError as error:
            if not self._osc_failed:
                logger.error("OSC send failed: %s (will keep trying)", error)
            self._osc_failed = True
            return False

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None


def create_bridge(use_osc: bool = True, **kwargs: Any) -> JSONOnlyBridge:
    """Create the bridge to the display.

    Args:
        use_osc: Send OSC in addition to writing the JSON file.
        **kwargs: ``config`` (:class:`OSCConfig`) and ``json_path``.
    """

    if use_osc:
        return ProcessingOSCBridge(**kwargs)
    kwargs.pop("config", None)
    return JSONOnlyBridge(**kwargs)


__all__ = [
    "DEFAULT_JSON_PATH",
    "JSONOnlyBridge",
    "OSCConfig",
    "ProcessingOSCBridge",
    "create_bridge",
    "decode_osc_message",
    "encode_osc_message",
]
