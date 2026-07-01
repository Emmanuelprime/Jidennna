"""
esp32_interface.py
==================
Serial interface to the diff_drive firmware running on the ESP32.

ODOM packet format (20 Hz):
  ODOM,t_ms,l_spd,r_spd,l_pwm,r_pwm,l_sp,r_sp,v,w,x,y,yaw

Velocity limits (firmware-enforced, right-motor limited):
  V_MAX  ≈ 0.65 m/s
  W_MAX  ≈ 2.49 rad/s
  Any CMD_VEL that would exceed a wheel limit is scaled down
  proportionally by the firmware before being applied.
"""

import serial
import threading
import time
from dataclasses import dataclass, field
from typing import Callable


# ─── Data types ───────────────────────────────────────────────────────────────

@dataclass
class OdomState:
    t_ms:      int   = 0
    l_speed:   float = 0.0
    r_speed:   float = 0.0
    l_pwm:     int   = 0
    r_pwm:     int   = 0
    l_sp:      float = 0.0
    r_sp:      float = 0.0
    v:         float = 0.0
    w:         float = 0.0
    x:         float = 0.0
    y:         float = 0.0
    yaw:       float = 0.0
    timestamp: float = field(default_factory=time.time)


# ─── Interface class ──────────────────────────────────────────────────────────

class ESP32Interface:
    """
    Thread-safe serial interface to the ESP32 diff_drive firmware.

    Usage
    -----
    iface = ESP32Interface("/dev/ttyUSB0")   # or "COM9"
    iface.connect()
    iface.stream_on()
    iface.set_velocity(0.3, 0.0)            # 0.3 m/s straight
    odom = iface.get_odom()
    iface.stop()
    iface.disconnect()
    """

    # Firmware velocity limits (right-motor limited, no load)
    V_MAX = 0.65   # m/s
    W_MAX = 2.49   # rad/s  (= V_MAX / half_track)

    def __init__(self, port: str, baud: int = 115200):
        self.port = port
        self.baud = baud
        self._ser: serial.Serial | None = None
        self._lock = threading.Lock()
        self._odom = OdomState()
        self._odom_callbacks: list[Callable[[OdomState], None]] = []
        self._reader_thread: threading.Thread | None = None
        self._running = False

    # ── Connection ─────────────────────────────────────────────────────────────

    def connect(self, timeout_s: float = 5.0) -> bool:
        """Open serial port and wait for READY from firmware."""
        try:
            self._ser = serial.Serial(
                self.port, self.baud,
                timeout=1.0,
                dsrdtr=False,
            )
            self._ser.dtr = False
            self._ser.rts = False
            time.sleep(0.1)
            self._ser.reset_input_buffer()

            # Ping handshake – firmware may already be running
            deadline = time.time() + timeout_s
            for attempt in range(8):
                if time.time() > deadline:
                    break
                self._ser.write(b"PING\n")
                self._ser.flush()
                ping_deadline = time.time() + 1.2
                while time.time() < ping_deadline:
                    line = self._ser.readline().decode(errors="replace").strip()
                    if line == "READY":
                        self._ser.reset_input_buffer()
                        self._start_reader()
                        return True
                    if line and not line.startswith("ODOM,"):
                        pass  # swallow non-ODOM lines silently during handshake

            # Fallback: trigger DTR reset
            self._ser.dtr = True
            time.sleep(0.15)
            self._ser.dtr = False
            boot_deadline = time.time() + 3.0
            while time.time() < boot_deadline:
                line = self._ser.readline().decode(errors="replace").strip()
                if line == "READY":
                    self._ser.reset_input_buffer()
                    self._start_reader()
                    return True

            return False

        except serial.SerialException as exc:
            raise ConnectionError(f"Cannot open {self.port}: {exc}") from exc

    def disconnect(self):
        """Stop reader thread and close the serial port."""
        self.stop()
        self._running = False
        if self._reader_thread and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=2.0)
        if self._ser and self._ser.is_open:
            self._ser.close()

    # ── Commands ───────────────────────────────────────────────────────────────

    def set_velocity(self, v: float, w: float):
        """
        Send a CMD_VEL command.

        Parameters
        ----------
        v : linear velocity  (m/s)  positive = forward
        w : angular velocity (rad/s) positive = counter-clockwise (left turn)
        """
        self._send(f"CMD_VEL:{v:.4f},{w:.4f}")

    def stop(self):
        """Immediately stop motors and disable streaming."""
        self._send("STOP")

    def stream_on(self):
        """Enable 20 Hz ODOM telemetry from firmware."""
        self._send("STREAM_ON")

    def stream_off(self):
        """Disable telemetry."""
        self._send("STREAM_OFF")

    def reset_odom(self):
        """Reset firmware odometry to (0, 0, 0)."""
        self._send("RESET_ODOM")

    def tune_pid(self, kp: float, ki: float, kd: float,
                 motor: str = "both") -> str:
        """
        Set PID gains.

        Parameters
        ----------
        motor : 'left' | 'right' | 'both'
        """
        cmd_map = {"both": "TUNE", "left": "TUNEL", "right": "TUNER"}
        self._send(f"{cmd_map[motor]}:{kp},{ki},{kd}")

    def get_status(self) -> str:
        """Request a STATUS line from firmware (blocking, up to 1 s)."""
        with self._lock:
            self._ser.write(b"STATUS\n")
            self._ser.flush()
            deadline = time.time() + 1.0
            while time.time() < deadline:
                line = self._ser.readline().decode(errors="replace").strip()
                if line.startswith("STATUS,"):
                    return line
        return ""

    # ── Odometry access ────────────────────────────────────────────────────────

    def get_odom(self) -> OdomState:
        """Return a copy of the most recent odometry state."""
        with self._lock:
            import copy
            return copy.copy(self._odom)

    def register_odom_callback(self, fn: Callable[[OdomState], None]):
        """Register a function called each time a new ODOM packet arrives."""
        self._odom_callbacks.append(fn)

    # ── Internal ───────────────────────────────────────────────────────────────

    def _send(self, cmd: str):
        if self._ser and self._ser.is_open:
            with self._lock:
                self._ser.write(f"{cmd}\n".encode())
                self._ser.flush()

    def _start_reader(self):
        self._running = True
        self._reader_thread = threading.Thread(
            target=self._reader_loop, daemon=True, name="esp32-reader"
        )
        self._reader_thread.start()

    def _reader_loop(self):
        while self._running:
            try:
                raw = self._ser.readline()
                if not raw:
                    continue
                line = raw.decode(errors="replace").strip()
                if line.startswith("ODOM,"):
                    odom = self._parse_odom(line)
                    if odom:
                        with self._lock:
                            self._odom = odom
                        for cb in self._odom_callbacks:
                            try:
                                cb(odom)
                            except Exception:
                                pass
                elif line == "WATCHDOG":
                    # Firmware stopped motors due to CMD_VEL timeout
                    pass
            except (serial.SerialException, OSError):
                break

    @staticmethod
    def _parse_odom(line: str) -> OdomState | None:
        """Parse  ODOM,t_ms,l_spd,r_spd,l_pwm,r_pwm,l_sp,r_sp,v,w,x,y,yaw"""
        parts = line.split(",")
        if len(parts) != 14:
            return None
        try:
            return OdomState(
                t_ms    = int(parts[1]),
                l_speed = float(parts[2]),
                r_speed = float(parts[3]),
                l_pwm   = int(parts[4]),
                r_pwm   = int(parts[5]),
                l_sp    = float(parts[6]),
                r_sp    = float(parts[7]),
                v       = float(parts[8]),
                w       = float(parts[9]),
                x       = float(parts[10]),
                y       = float(parts[11]),
                yaw     = float(parts[12]),
                timestamp = time.time(),
            )
        except (ValueError, IndexError):
            return None


# ─── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, math

    parser = argparse.ArgumentParser(description="Quick diff_drive firmware test")
    parser.add_argument("--port", default="COM9")
    parser.add_argument("--baud", type=int, default=115200)
    args = parser.parse_args()

    iface = ESP32Interface(args.port, args.baud)
    print(f"Connecting to {args.port} ...")
    if not iface.connect():
        print("Failed to connect.")
        raise SystemExit(1)

    print("Connected. Starting stream ...")
    iface.stream_on()
    iface.reset_odom()

    def on_odom(s: OdomState):
        print(f"  v={s.v:+.3f} w={s.w:+.3f}  x={s.x:.3f} y={s.y:.3f} yaw={math.degrees(s.yaw):.1f}°  "
              f"l_pwm={s.l_pwm} r_pwm={s.r_pwm}")

    iface.register_odom_callback(on_odom)

    try:
        print("Driving forward 0.3 m/s for 2 s ...")
        iface.set_velocity(0.3, 0.0)
        time.sleep(2.0)

        print("Turning left (w=1.0 rad/s) for 2 s ...")
        iface.set_velocity(0.0, 1.0)
        time.sleep(2.0)

        print("Stopping ...")
        iface.stop()
        time.sleep(0.5)

    finally:
        iface.disconnect()
        print("Done.")
