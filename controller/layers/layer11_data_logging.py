"""
Layer 11: Data logging for controller runs.
"""
import json
import os
import threading
import time
from datetime import datetime

import numpy as np


class DataLogger:
    """
    Append structured controller data to a JSON Lines file.

    Each line is a complete JSON object, which keeps logging simple while still
    supporting nested values like feedback packets, obstacle state, and MPC
    trajectories.
    """

    def __init__(self, log_dir="logs", run_name=None, enabled=True):
        self.enabled = enabled
        self.log_dir = log_dir
        self.run_name = run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
        self.path = os.path.join(self.log_dir, f"{self.run_name}.jsonl")
        self._lock = threading.Lock()
        self._file = None
        self.records_written = 0

        if self.enabled:
            os.makedirs(self.log_dir, exist_ok=True)
            self._file = open(self.path, "a", encoding="utf-8")

    def log_metadata(self, **metadata):
        self.log("metadata", metadata=metadata)

    def log_cycle(
        self,
        cycle,
        estimated_state=None,
        true_state=None,
        feedback=None,
        control=None,
        wheel_commands=None,
        reference_traj=None,
        predicted_traj=None,
        obstacle_status=None,
        current_waypoint=None,
        target_waypoint=None,
        sim_time=None,
        extra=None,
    ):
        self.log(
            "cycle",
            cycle=cycle,
            sim_time=sim_time,
            estimated_state=estimated_state,
            true_state=true_state,
            feedback=feedback,
            control=control,
            wheel_commands=wheel_commands,
            reference_traj=reference_traj,
            predicted_traj=predicted_traj,
            obstacle_status=obstacle_status,
            current_waypoint=current_waypoint,
            target_waypoint=target_waypoint,
            extra=extra or {},
        )

    def log(self, event, **payload):
        if not self.enabled:
            return

        record = {
            "event": event,
            "wall_time": time.time(),
            "wall_time_iso": datetime.now().isoformat(timespec="milliseconds"),
            **payload,
        }

        line = json.dumps(self._to_json_safe(record), separators=(",", ":"))

        with self._lock:
            if self._file is None:
                return
            self._file.write(line + "\n")
            self._file.flush()
            self.records_written += 1

    def close(self):
        with self._lock:
            if self._file is not None:
                self._file.flush()
                self._file.close()
                self._file = None

    def _to_json_safe(self, value):
        if isinstance(value, np.ndarray):
            return value.tolist()

        if isinstance(value, np.generic):
            return value.item()

        if isinstance(value, dict):
            return {str(key): self._to_json_safe(val) for key, val in value.items()}

        if isinstance(value, (list, tuple)):
            return [self._to_json_safe(item) for item in value]

        return value
