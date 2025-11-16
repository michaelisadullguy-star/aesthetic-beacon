"""
OSC Bridge for Real-Time Parameter Communication

Sends Processing parameters from Python to Processing sketch via OSC (Open Sound Control).
Requires python-osc library: pip install python-osc
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from pythonosc import udp_client
    from pythonosc.osc_message_builder import OscMessageBuilder
    OSC_AVAILABLE = True
except ImportError:
    OSC_AVAILABLE = False
    print("Warning: python-osc not installed. Run: pip install python-osc")


@dataclass
class OSCConfig:
    """Configuration for OSC communication."""
    host: str = "127.0.0.1"
    port: int = 12000
    address_prefix: str = "/aesthetic_beacon"
    batch_send: bool = True  # Send all params in one bundle
    fallback_to_json: bool = True  # Also save to JSON file


class ProcessingOSCBridge:
    """
    Bridge for sending Processing parameters to a Processing sketch via OSC.

    Usage:
        bridge = ProcessingOSCBridge()
        bridge.send_parameters({"canvasWidth": 800, "noiseScale": 0.015, ...})
    """

    def __init__(self, config: Optional[OSCConfig] = None, json_path: Optional[Path] = None):
        """
        Initialize the OSC bridge.

        Args:
            config: OSC configuration (uses defaults if None)
            json_path: Fallback JSON file path for parameter output
        """
        self.config = config or OSCConfig()
        self.json_path = json_path or Path("data/step6_realtime/outputs/current_parameters.json")

        if OSC_AVAILABLE:
            self.client = udp_client.SimpleUDPClient(self.config.host, self.config.port)
            print(f"OSC client initialized: {self.config.host}:{self.config.port}")
        else:
            self.client = None
            print("OSC not available, will use JSON file fallback only")

        # Ensure JSON output directory exists
        self.json_path.parent.mkdir(parents=True, exist_ok=True)

    def send_parameters(self, parameters: Dict[str, Any], timestamp: Optional[float] = None) -> bool:
        """
        Send Processing parameters via OSC and/or JSON file.

        Args:
            parameters: Dictionary of parameter name -> value
            timestamp: Optional timestamp for this update

        Returns:
            True if successfully sent, False otherwise
        """
        if timestamp is None:
            timestamp = time.time()

        success = True

        # Send via OSC if available
        if self.client is not None:
            try:
                if self.config.batch_send:
                    # Send all parameters in a bundle
                    self._send_bundle(parameters, timestamp)
                else:
                    # Send each parameter individually
                    for param_name, value in parameters.items():
                        self._send_parameter(param_name, value)

                print(f"✓ Sent {len(parameters)} parameters via OSC")
            except Exception as e:
                print(f"✗ OSC send failed: {e}")
                success = False

        # Fallback to JSON file
        if self.config.fallback_to_json or not OSC_AVAILABLE:
            try:
                self._write_json(parameters, timestamp)
                print(f"✓ Wrote parameters to {self.json_path}")
            except Exception as e:
                print(f"✗ JSON write failed: {e}")
                success = False

        return success

    def _send_parameter(self, name: str, value: Any):
        """Send a single parameter via OSC."""
        address = f"{self.config.address_prefix}/{name}"

        # Handle different value types
        if isinstance(value, (int, float, bool)):
            self.client.send_message(address, value)
        elif isinstance(value, str):
            self.client.send_message(address, value)
        elif isinstance(value, (list, tuple)):
            self.client.send_message(address, list(value))
        else:
            # Fallback to string representation
            self.client.send_message(address, str(value))

    def _send_bundle(self, parameters: Dict[str, Any], timestamp: float):
        """Send all parameters in a single OSC bundle."""
        from pythonosc import osc_bundle_builder

        builder = osc_bundle_builder.OscBundleBuilder(timestamp)

        for name, value in parameters.items():
            address = f"{self.config.address_prefix}/{name}"
            msg_builder = OscMessageBuilder(address=address)

            # Add value with appropriate type
            if isinstance(value, bool):
                msg_builder.add_arg(int(value))  # OSC doesn't have bool, use int
            elif isinstance(value, (int, float)):
                msg_builder.add_arg(value)
            elif isinstance(value, str):
                msg_builder.add_arg(value)
            elif isinstance(value, (list, tuple)):
                for v in value:
                    msg_builder.add_arg(v)
            else:
                msg_builder.add_arg(str(value))

            builder.add_content(msg_builder.build())

        bundle = builder.build()
        self.client.send(bundle)

    def _write_json(self, parameters: Dict[str, Any], timestamp: float):
        """Write parameters to JSON file as fallback."""
        output = {
            "timestamp": timestamp,
            "parameters": parameters
        }

        with open(self.json_path, 'w') as f:
            json.dump(output, f, indent=2)

        # Also save a timestamped copy
        history_dir = self.json_path.parent / "parameter_history"
        history_dir.mkdir(exist_ok=True)

        history_file = history_dir / f"params_{int(timestamp)}.json"
        with open(history_file, 'w') as f:
            json.dump(output, f, indent=2)

    def send_state_change(self, is_creatable: bool, confidence: float = 1.0):
        """
        Send creator state change notification.

        Args:
            is_creatable: Whether the creator is in a creatable state
            confidence: Model confidence (0-1)
        """
        if self.client is not None:
            self.client.send_message(
                f"{self.config.address_prefix}/state",
                [int(is_creatable), confidence]
            )
            print(f"State: {'CREATABLE' if is_creatable else 'NOT CREATABLE'} (confidence: {confidence:.2f})")

    def close(self):
        """Clean up resources."""
        if self.client is not None:
            # UDP client doesn't need explicit closing
            pass


class JSONOnlyBridge:
    """
    Simplified bridge that only writes parameters to JSON files.
    Use this if OSC is not needed or unavailable.
    """

    def __init__(self, json_path: Optional[Path] = None):
        """Initialize JSON-only bridge."""
        self.json_path = json_path or Path("data/step6_realtime/outputs/current_parameters.json")
        self.json_path.parent.mkdir(parents=True, exist_ok=True)

    def send_parameters(self, parameters: Dict[str, Any], timestamp: Optional[float] = None) -> bool:
        """Write parameters to JSON file."""
        if timestamp is None:
            timestamp = time.time()

        output = {
            "timestamp": timestamp,
            "parameters": parameters
        }

        try:
            with open(self.json_path, 'w') as f:
                json.dump(output, f, indent=2)

            print(f"✓ Wrote {len(parameters)} parameters to {self.json_path}")
            return True
        except Exception as e:
            print(f"✗ JSON write failed: {e}")
            return False

    def send_state_change(self, is_creatable: bool, confidence: float = 1.0):
        """Log state change (JSON-only mode)."""
        print(f"State: {'CREATABLE' if is_creatable else 'NOT CREATABLE'} (confidence: {confidence:.2f})")


def create_bridge(use_osc: bool = True, **kwargs) -> Any:
    """
    Factory function to create appropriate bridge.

    Args:
        use_osc: If True and python-osc available, use OSC bridge. Otherwise use JSON-only.
        **kwargs: Additional arguments for bridge configuration

    Returns:
        ProcessingOSCBridge or JSONOnlyBridge instance
    """
    if use_osc and OSC_AVAILABLE:
        return ProcessingOSCBridge(**kwargs)
    else:
        return JSONOnlyBridge(**kwargs)


# Example usage
if __name__ == "__main__":
    # Example 1: OSC Bridge
    print("=== OSC Bridge Example ===")
    bridge = create_bridge(use_osc=True)

    example_params = {
        "canvasWidth": 800,
        "canvasHeight": 800,
        "noiseScale": 0.015,
        "colorCycleSpeed": 0.002,
        "cellSize": 40,
        "targetFrameRate": 30
    }

    bridge.send_parameters(example_params)
    bridge.send_state_change(is_creatable=True, confidence=0.87)

    print("\n=== JSON-Only Bridge Example ===")
    json_bridge = JSONOnlyBridge()
    json_bridge.send_parameters(example_params)
