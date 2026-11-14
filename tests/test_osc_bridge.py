import json
import shutil
import socket
import subprocess
import time

import pytest

from src.osc_bridge import (
    JSONOnlyBridge,
    OSCConfig,
    ProcessingOSCBridge,
    create_bridge,
    decode_osc_message,
    encode_osc_message,
)
from src.parameters import PARAMETER_NAMES, default_parameters


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def test_encoding_round_trip():
    arguments = [7, "cellSize", 24, "noiseScale", 0.006, "name", "héllo", True]
    address, decoded = decode_osc_message(encode_osc_message("/aesthetic_beacon/params", arguments))
    assert address == "/aesthetic_beacon/params"
    assert decoded[:4] == [7, "cellSize", 24, "noiseScale"]
    assert decoded[4] == pytest.approx(0.006)
    assert decoded[5:] == ["name", "héllo", 1]


def test_encoding_agrees_with_python_osc():
    builder = pytest.importorskip("pythonosc.osc_message_builder")
    message = builder.OscMessageBuilder(address="/aesthetic_beacon/state")
    for value in (1, 0.5, "creatable"):
        message.add_arg(value)
    assert message.build().dgram == encode_osc_message("/aesthetic_beacon/state", [1, 0.5, "creatable"])


def test_bridge_sends_one_complete_message_and_mirrors_json(tmp_path):
    port = _free_port()
    listener = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    listener.bind(("127.0.0.1", port))
    listener.settimeout(2.0)
    json_path = tmp_path / "out" / "current_parameters.json"
    bridge = ProcessingOSCBridge(OSCConfig(port=port), json_path=json_path)

    bridge.send_state_change(True, 0.82, "creatable")
    address, arguments = decode_osc_message(listener.recv(65536))
    assert address == "/aesthetic_beacon/state" and arguments[0] == 1 and arguments[2] == "creatable"

    parameters = default_parameters()
    assert bridge.send_parameters(parameters)
    packet = listener.recv(65536)
    assert len(packet) < 1472  # fits one unfragmented datagram
    address, arguments = decode_osc_message(packet)
    assert address == "/aesthetic_beacon/params" and arguments[0] == 1
    names, values = arguments[1::2], arguments[2::2]
    assert tuple(names) == PARAMETER_NAMES
    received = dict(zip(names, values))
    assert isinstance(received["cellSize"], int) and isinstance(received["noiseScale"], float)

    document = json.loads(json_path.read_text())
    assert document["sequence"] == 1 and document["cellSize"] == 24
    assert document["state"] == {"creatable": True, "confidence": 0.82, "label": "creatable"}
    assert not list(json_path.parent.glob("*.tmp"))
    bridge.close()
    listener.close()


def test_bridge_survives_an_unreachable_display(tmp_path):
    bridge = ProcessingOSCBridge(OSCConfig(host="no-such-host.invalid", port=1), json_path=tmp_path / "p.json")
    assert bridge.send_parameters(default_parameters())  # the file still gets written
    bridge.send_state_change(False, 0.1)
    assert (tmp_path / "p.json").is_file()

    json_only = create_bridge(use_osc=False, config=OSCConfig(), json_path=tmp_path / "q.json")
    assert isinstance(json_only, JSONOnlyBridge) and not isinstance(json_only, ProcessingOSCBridge)
    assert json_only.send_parameters({"cellSize": 0, "unknown": 5})
    assert json.loads((tmp_path / "q.json").read_text())["cellSize"] == 12


@pytest.mark.skipif(shutil.which("javac") is None, reason="needs a JDK")
def test_sketch_receiver_understands_the_bridge(tmp_path, repo_root):
    """Compile the sketch's own OSC receiver and feed it from the Python bridge."""

    source = repo_root / "scripts" / "processing_templates" / "dynamic_texture_osc" / "OscReceiver.java"
    subprocess.run(["javac", "-d", str(tmp_path), str(source)], check=True, capture_output=True)
    port = _free_port()
    process = subprocess.Popen(
        ["java", "-cp", str(tmp_path), "OscReceiver", str(port), "3"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
    )
    try:
        assert process.stdout.readline().strip() == f"LISTENING {port}"
        bridge = ProcessingOSCBridge(OSCConfig(port=port), json_path=tmp_path / "p.json")
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sender.sendto(b"\x00\x01 not osc", ("127.0.0.1", port))
        bridge.send_state_change(True, 0.75, "creatable")
        bridge.send_parameters(default_parameters())
        time.sleep(0.3)
        output, _ = process.communicate(timeout=10)
    finally:
        process.kill()
    lines = output.strip().splitlines()
    assert lines[0] == "/aesthetic_beacon/state\tInteger:1\tFloat:0.75\tString:creatable"
    fields = lines[1].split("\t")
    assert fields[0] == "/aesthetic_beacon/params" and fields[1] == "Integer:1"
    assert len(fields) == 2 + 2 * 50
    assert "String:cellSize" in fields and fields[fields.index("String:cellSize") + 1] == "Integer:24"
