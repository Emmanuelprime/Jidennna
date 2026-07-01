"""
Speed Control Test Script
=========================
Tests the PID speed controller on the ESP32 firmware (speed_control.ino).

DATA packet format (12 fields):
  DATA,timestamp_ms,left_pulses,right_pulses,
       left_speed,right_speed,
       left_pwm,right_pwm,
       left_setpoint,right_setpoint,
       left_pid_out,right_pid_out

Motor models (identified from experiments):
  Left:  K = 0.0402 * PWM - 0.3415,  τ = 0.204 s  → settle ~1.0 s
  Right: K = 0.0127 * PWM + 0.0059,  τ = 0.219 s  → settle ~1.1 s
"""

import serial
import time
import csv
import argparse
import sys
import os
from datetime import datetime

try:
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False
    print("Warning: numpy/matplotlib not found – plotting disabled.")


# ─── Constants ───────────────────────────────────────────────────────────────

# Max reachable speed at PWM 60 (from data)
LEFT_MAX_SPEED  = 2.0   # m/s
RIGHT_MAX_SPEED = 0.69  # m/s  (right motor is weaker)

# Motor time constants
LEFT_TAU  = 0.204  # seconds
RIGHT_TAU = 0.219  # seconds


# ─── Serial Connection ────────────────────────────────────────────────────────

class SpeedControlTester:
    def __init__(self, port: str, baud: int = 115200):
        self.port = port
        self.baud = baud
        self.ser: serial.Serial | None = None

    # ------------------------------------------------------------------
    def connect(self) -> bool:
        """
        Open serial port and wait for firmware READY.

        Handles two cases:
          A) ESP32 already running  -> send PING, read READY via readline()
          B) ESP32 just reset       -> drain boot messages, catch READY
        DTR is kept LOW so opening the port does NOT reset the ESP32.
        """
        try:
            print(f"Opening {self.port} @ {self.baud} baud ...")
            self.ser = serial.Serial(
                self.port, self.baud,
                timeout=1.0,    # readline() blocks up to 1 s per call
                dsrdtr=False,
            )
            # Immediately lower DTR/RTS so the ESP32 does NOT reset on port open
            self.ser.dtr = False
            self.ser.rts = False

            # Drain any pending bytes then ping
            time.sleep(0.1)
            self.ser.reset_input_buffer()

            # Case A: already running -- readline() blocks up to timeout,
            # so we don't need to poll in_waiting.
            print("Pinging firmware (already running) ...")
            for attempt in range(5):
                self.ser.write(b"PING\n")
                self.ser.flush()

                # Read lines for up to 1.5 s per attempt
                deadline = time.time() + 1.5
                while time.time() < deadline:
                    line = self.ser.readline().decode(errors="replace").strip()
                    if line == "READY":
                        print("Connected -- firmware is READY.")
                        self.ser.reset_input_buffer()
                        return True
                    # Print anything unexpected (skip DATA lines)
                    if line and not line.startswith("DATA,"):
                        print(f"  Firmware: {line}")

                print(f"  Ping attempt {attempt + 1}/5 ...")

            # Case B: trigger reset and wait for spontaneous READY
            print("No READY -- triggering DTR reset and waiting for boot ...")
            self.ser.dtr = True
            time.sleep(0.15)
            self.ser.dtr = False

            boot_deadline = time.time() + 3.0
            while time.time() < boot_deadline:
                line = self.ser.readline().decode(errors="replace").strip()
                if line == "READY":
                    print("Connected after reset -- firmware is READY.")
                    self.ser.reset_input_buffer()
                    return True

            # One final PING after full boot window
            self.ser.reset_input_buffer()
            self.ser.write(b"PING\n")
            self.ser.flush()
            deadline = time.time() + 1.5
            while time.time() < deadline:
                line = self.ser.readline().decode(errors="replace").strip()
                if line == "READY":
                    print("Connected -- firmware is READY.")
                    self.ser.reset_input_buffer()
                    return True

            print("ERROR: Could not get READY from firmware.")
            return False

        except serial.SerialException as exc:
            print(f"Serial error: {exc}")
            return False

    # ------------------------------------------------------------------
    def disconnect(self):
        if self.ser and self.ser.is_open:
            self._send("s")          # stop motors
            time.sleep(0.2)
            self.ser.close()
        print("Disconnected.")

    # ------------------------------------------------------------------
    def _send(self, cmd: str):
        self.ser.write(f"{cmd}\n".encode())
        self.ser.flush()

    # ------------------------------------------------------------------
    def stop(self):
        self._send("s")

    # ------------------------------------------------------------------
    def tune_pid(self, kp: float, ki: float, kd: float, motor: str = "both"):
        """
        Send PID gains to the firmware.

        Parameters
        ----------
        motor : 'left' | 'right' | 'both'
        """
        cmd_map = {"both": "TUNE", "left": "TUNEL", "right": "TUNER"}
        cmd = cmd_map.get(motor, "TUNE")
        self._send(f"{cmd}:{kp},{ki},{kd}")
        deadline = time.time() + 0.8
        while time.time() < deadline:
            line = self.ser.readline().decode(errors="replace").strip()
            if line and not line.startswith("DATA,"):
                print(f"  Firmware: {line}")
                break

    # ------------------------------------------------------------------
    def _parse_data_line(self, line: str) -> dict | None:
        """Parse a DATA,… line into a dict. Returns None on bad lines."""
        if not line.startswith("DATA,"):
            return None
        parts = line.split(",")
        if len(parts) != 12:
            return None
        try:
            return {
                "t_ms":          int(parts[1]),
                "left_pulses":   int(parts[2]),
                "right_pulses":  int(parts[3]),
                "left_speed":    float(parts[4]),
                "right_speed":   float(parts[5]),
                "left_pwm":      int(parts[6]),
                "right_pwm":     int(parts[7]),
                "left_sp":       float(parts[8]),
                "right_sp":      float(parts[9]),
                "left_pid_out":  float(parts[10]),
                "right_pid_out": float(parts[11]),
            }
        except (ValueError, IndexError):
            return None

    # ------------------------------------------------------------------
    def _collect(self, duration_s: float) -> list[dict]:
        """Collect DATA packets for `duration_s` seconds."""
        records = []
        deadline = time.time() + duration_s
        while time.time() < deadline:
            if self.ser.in_waiting:
                raw = self.ser.readline().decode(errors="replace").strip()
                rec = self._parse_data_line(raw)
                if rec:
                    records.append(rec)
            else:
                time.sleep(0.005)
        return records

    # ------------------------------------------------------------------
    def run_step_test(
        self,
        speeds_ms: list[float] = None,
        hold_time_s: float = 3.0,
    ) -> list[dict]:
        """
        Apply a sequence of speed setpoints and record the response.

        Parameters
        ----------
        speeds_ms   : list of (left_speed, right_speed) tuples in m/s.
                      Pass a list of floats to use the same value for both.
        hold_time_s : seconds to hold each setpoint before switching.

        Returns a flat list of DATA records (dicts) with an added 'phase' key.
        """
        if speeds_ms is None:
            speeds_ms = [0.2, 0.4, 0.6, 0.4, 0.2, 0.0]

        # Normalise to list of (left, right) tuples
        steps: list[tuple[float, float]] = []
        for s in speeds_ms:
            if isinstance(s, (int, float)):
                steps.append((float(s), float(s)))
            else:
                steps.append((float(s[0]), float(s[1])))

        all_records: list[dict] = []

        print(f"\nStep-response test  ({len(steps)} steps × {hold_time_s:.1f} s each)")
        print("─" * 60)

        for phase, (l_sp, r_sp) in enumerate(steps):
            print(f"  Step {phase + 1}/{len(steps)}: "
                  f"left={l_sp:+.3f} m/s  right={r_sp:+.3f} m/s")
            self._send(f"STREAM:{l_sp},{r_sp}")
            time.sleep(0.1)

            # Drain any firmware echo
            while self.ser.in_waiting:
                echo = self.ser.readline().decode(errors="replace").strip()
                if echo:
                    print(f"    Firmware: {echo}")

            records = self._collect(hold_time_s)
            for r in records:
                r["phase"] = phase
            all_records.extend(records)
            print(f"    Collected {len(records)} samples")

        # Stop streaming and motors
        self._send("STOP_STREAM")
        time.sleep(0.1)
        self._send("s")
        print("  Motors stopped.")
        return all_records

    # ------------------------------------------------------------------
    def run_ramp_test(
        self,
        start_ms: float = 0.1,
        end_ms: float = 0.6,
        n_steps: int = 6,
        hold_time_s: float = 3.0,
    ) -> list[dict]:
        """Ramp through evenly spaced speeds then back to zero."""
        import numpy as np  # local import – already guarded at top
        ramp_up   = list(np.linspace(start_ms, end_ms, n_steps))
        ramp_down = list(reversed(ramp_up[:-1])) + [0.0]
        speeds = ramp_up + ramp_down
        return self.run_step_test(speeds_ms=speeds, hold_time_s=hold_time_s)

    # ------------------------------------------------------------------
    def run_symmetric_test(self, hold_time_s: float = 3.0) -> list[dict]:
        """Forward / stop / reverse to check symmetry."""
        # Clamp to right-motor capability (≈ 0.5 m/s)
        limit = min(0.5, RIGHT_MAX_SPEED * 0.7)
        speeds = [limit, 0.0, -limit, 0.0]
        return self.run_step_test(speeds_ms=speeds, hold_time_s=hold_time_s)


# ─── Data Saving ─────────────────────────────────────────────────────────────

def save_csv(records: list[dict], filepath: str):
    if not records:
        print("No data to save.")
        return
    fieldnames = list(records[0].keys())
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved {len(records)} records → {filepath}")


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_step_response(records: list[dict], title: str = "Speed Control Step Response"):
    if not HAS_PLOT:
        print("matplotlib not available – skipping plot.")
        return
    if not records:
        print("No data to plot.")
        return

    t0 = records[0]["t_ms"]
    t  = np.array([(r["t_ms"] - t0) / 1000.0 for r in records])
    l_speed = np.array([r["left_speed"]  for r in records])
    r_speed = np.array([r["right_speed"] for r in records])
    l_sp    = np.array([r["left_sp"]     for r in records])
    r_sp    = np.array([r["right_sp"]    for r in records])
    l_pwm   = np.array([r["left_pwm"]    for r in records])
    r_pwm   = np.array([r["right_pwm"]   for r in records])

    fig = plt.figure(figsize=(14, 9))
    fig.suptitle(title, fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── Left speed ──────────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(t, l_sp,    "k--", lw=1.2, label="Setpoint")
    ax0.plot(t, l_speed, "b",   lw=1.5, label="Measured")
    ax0.set_title("Left Motor Speed")
    ax0.set_ylabel("Speed (m/s)")
    ax0.legend(fontsize=8)
    ax0.grid(True, alpha=0.3)

    # ── Right speed ──────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(t, r_sp,    "k--", lw=1.2, label="Setpoint")
    ax1.plot(t, r_speed, "r",   lw=1.5, label="Measured")
    ax1.set_title("Right Motor Speed")
    ax1.set_ylabel("Speed (m/s)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ── Left PWM ─────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(t, l_pwm, "b", lw=1.2)
    ax2.set_title("Left PWM Output")
    ax2.set_ylabel("PWM (0–60)")
    ax2.set_ylim(-2, 65)
    ax2.grid(True, alpha=0.3)

    # ── Right PWM ────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(t, r_pwm, "r", lw=1.2)
    ax3.set_title("Right PWM Output")
    ax3.set_ylabel("PWM (0–60)")
    ax3.set_ylim(-2, 65)
    ax3.grid(True, alpha=0.3)

    # ── Tracking error ───────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    l_err = l_sp - l_speed
    ax4.plot(t, l_err, "b", lw=1.2)
    ax4.axhline(0, color="k", lw=0.8, ls="--")
    ax4.set_title("Left Tracking Error")
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Error (m/s)")
    ax4.grid(True, alpha=0.3)

    ax5 = fig.add_subplot(gs[2, 1])
    r_err = r_sp - r_speed
    ax5.plot(t, r_err, "r", lw=1.2)
    ax5.axhline(0, color="k", lw=0.8, ls="--")
    ax5.set_title("Right Tracking Error")
    ax5.set_xlabel("Time (s)")
    ax5.set_ylabel("Error (m/s)")
    ax5.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()


def print_performance_summary(records: list[dict]):
    """Print steady-state error and approximate settling time per phase."""
    if not records:
        return

    phases = sorted({r["phase"] for r in records})

    print("\n" + "═" * 60)
    print("PERFORMANCE SUMMARY")
    print("═" * 60)
    fmt = "{:>5}  {:>7}  {:>7}  {:>8}  {:>8}  {:>8}  {:>8}"
    print(fmt.format("Phase", "L_sp", "R_sp",
                     "L_ss_err", "R_ss_err",
                     "L_RMS_e", "R_RMS_e"))
    print("─" * 60)

    for ph in phases:
        recs = [r for r in records if r["phase"] == ph]
        if not recs:
            continue

        l_sp = recs[0]["left_sp"]
        r_sp = recs[0]["right_sp"]

        # Use last 40 % of the hold window for steady-state estimate
        ss_slice = recs[int(len(recs) * 0.6):]
        if not ss_slice:
            ss_slice = recs

        l_errors = [r["left_sp"]  - r["left_speed"]  for r in ss_slice]
        r_errors = [r["right_sp"] - r["right_speed"] for r in ss_slice]

        l_ss   = sum(l_errors) / len(l_errors)
        r_ss   = sum(r_errors) / len(r_errors)
        l_rms  = (sum(e ** 2 for e in l_errors) / len(l_errors)) ** 0.5
        r_rms  = (sum(e ** 2 for e in r_errors) / len(r_errors)) ** 0.5

        print(fmt.format(
            ph + 1,
            f"{l_sp:+.3f}", f"{r_sp:+.3f}",
            f"{l_ss:+.4f}", f"{r_ss:+.4f}",
            f"{l_rms:.4f}", f"{r_rms:.4f}",
        ))

    print("─" * 60)
    print("ss_err = mean steady-state error (m/s), RMS over last 40% of step\n")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Test PID speed control on the ESP32 robot.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--port", default="COM3",
                   help="Serial port (default: COM3 on Windows, /dev/ttyUSB0 on Linux)")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--test", choices=["step", "ramp", "symmetric", "custom"],
                   default="step",
                   help=(
                       "step     – fixed steps at preset speeds (default)\n"
                       "ramp     – ramp up then ramp down\n"
                       "symmetric– forward / stop / reverse\n"
                       "custom   – supply --speeds"
                   ))
    p.add_argument("--speeds", nargs="+", type=float,
                   help="Custom list of speeds in m/s (used with --test custom)")
    p.add_argument("--hold", type=float, default=3.0,
                   help="Hold time per step in seconds (default: 3.0)")
    # ── PID tuning (per-motor or shared) ─────────────────────────────
    g = p.add_argument_group("PID tuning")
    g.add_argument("--kp",       type=float, help="Kp for both motors")
    g.add_argument("--ki",       type=float, help="Ki for both motors")
    g.add_argument("--kd",       type=float, help="Kd for both motors")
    g.add_argument("--kp-left",  type=float, help="Kp for left motor only")
    g.add_argument("--ki-left",  type=float, help="Ki for left motor only")
    g.add_argument("--kd-left",  type=float, help="Kd for left motor only")
    g.add_argument("--kp-right", type=float, help="Kp for right motor only")
    g.add_argument("--ki-right", type=float, help="Ki for right motor only")
    g.add_argument("--kd-right", type=float, help="Kd for right motor only")
    p.add_argument("--save", action="store_true",
                   help="Save collected data to a timestamped CSV file")
    p.add_argument("--no-plot", action="store_true",
                   help="Skip the matplotlib plot")
    return p


def main():
    parser = build_parser()
    args   = parser.parse_args()

    tester = SpeedControlTester(args.port, args.baud)

    if not tester.connect():
        sys.exit(1)

    # Defaults from firmware initialisation
    _DEF = {"kp": 10.0, "ki": 2.0, "kd": 0.1}

    try:
        # ── Shared override (both motors) ────────────────────────────
        if args.kp is not None or args.ki is not None or args.kd is not None:
            kp = args.kp if args.kp is not None else _DEF["kp"]
            ki = args.ki if args.ki is not None else _DEF["ki"]
            kd = args.kd if args.kd is not None else _DEF["kd"]
            print(f"PID both  -- Kp={kp}  Ki={ki}  Kd={kd}")
            tester.tune_pid(kp, ki, kd, motor="both")

        # ── Per-motor overrides ───────────────────────────────────────
        if args.kp_left is not None or args.ki_left is not None or args.kd_left is not None:
            kp = args.kp_left  if args.kp_left  is not None else _DEF["kp"]
            ki = args.ki_left  if args.ki_left  is not None else _DEF["ki"]
            kd = args.kd_left  if args.kd_left  is not None else _DEF["kd"]
            print(f"PID left  -- Kp={kp}  Ki={ki}  Kd={kd}")
            tester.tune_pid(kp, ki, kd, motor="left")

        if args.kp_right is not None or args.ki_right is not None or args.kd_right is not None:
            kp = args.kp_right if args.kp_right is not None else _DEF["kp"]
            ki = args.ki_right if args.ki_right is not None else _DEF["ki"]
            kd = args.kd_right if args.kd_right is not None else _DEF["kd"]
            print(f"PID right -- Kp={kp}  Ki={ki}  Kd={kd}")
            tester.tune_pid(kp, ki, kd, motor="right")

        # Choose test mode
        if args.test == "step":
            # Preset steps within right-motor capability
            steps = [0.2, 0.35, 0.5, 0.35, 0.2, 0.0]
            records = tester.run_step_test(steps, args.hold)

        elif args.test == "ramp":
            records = tester.run_ramp_test(
                start_ms=0.1,
                end_ms=min(0.6, RIGHT_MAX_SPEED * 0.85),
                n_steps=6,
                hold_time_s=args.hold,
            )

        elif args.test == "symmetric":
            records = tester.run_symmetric_test(args.hold)

        elif args.test == "custom":
            if not args.speeds:
                print("ERROR: --test custom requires --speeds")
                sys.exit(1)
            records = tester.run_step_test(args.speeds, args.hold)

        else:
            records = []

    finally:
        tester.disconnect()

    if not records:
        print("No data collected.")
        sys.exit(0)

    print_performance_summary(records)

    if args.save:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(os.path.dirname(__file__), "..", "logs") \
                  if os.path.isdir(os.path.join(os.path.dirname(__file__), "..", "logs")) \
                  else os.path.dirname(__file__)
        out_path = os.path.join(out_dir, f"speed_ctrl_{args.test}_{ts}.csv")
        save_csv(records, out_path)

    if not args.no_plot:
        test_title = f"Speed Control – {args.test} test  (Ts=100 ms)"
        plot_step_response(records, title=test_title)


if __name__ == "__main__":
    main()
