#!/usr/bin/env python3
"""
gui_velocity_test.py
====================
Tkinter GUI for testing the diff_drive firmware over serial.

Run on the Jetson (with X11 forwarding from Windows):
  ssh -X prime@<jetson-ip>
  python3 experiments/gui_velocity_test.py --port /dev/ttyUSB0

Or locally on Windows (if ESP32 is on this machine):
  python experiments\gui_velocity_test.py --port COM9
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
import serial
import threading
import time
import math
import queue
import argparse

V_MAX     = 0.65    # m/s   (right-motor limited)
W_MAX     = 2.49    # rad/s (= V_MAX / half_track)
MAX_TRAIL = 800     # max trajectory points kept


class DiffDriveGUI:
    def __init__(self, root: tk.Tk, default_port: str = "/dev/ttyUSB0"):
        self.root = root
        self.root.title("Diff Drive Tester")

        self.ser: serial.Serial | None = None
        self.connected   = False
        self._running    = False
        self._streaming  = False

        self._data_lock  = threading.Lock()   # protects _odom + _trail
        self._ser_lock   = threading.Lock()   # protects serial writes
        self._rx_thread: threading.Thread | None = None

        self._odom:  dict  = {}
        self._trail: list  = []   # [(x, y), ...]
        self._log_q: queue.Queue = queue.Queue()

        self._v = 0.0
        self._w = 0.0

        self._build_ui(default_port)
        self._poll()

    # ── UI construction ──────────────────────────────────────────────────────

    def _build_ui(self, default_port: str):
        # ── Top bar: connection controls ─────────────────────────────────────
        top = ttk.Frame(self.root, padding=(6, 4))
        top.pack(fill=tk.X)

        ttk.Label(top, text="Port:").pack(side=tk.LEFT)
        self._port_var = tk.StringVar(value=default_port)
        ttk.Entry(top, textvariable=self._port_var, width=16).pack(side=tk.LEFT, padx=3)

        ttk.Label(top, text="Baud:").pack(side=tk.LEFT)
        self._baud_var = tk.StringVar(value="115200")
        ttk.Entry(top, textvariable=self._baud_var, width=8).pack(side=tk.LEFT, padx=3)

        self._conn_btn = ttk.Button(top, text="Connect", command=self._toggle_connect, width=12)
        self._conn_btn.pack(side=tk.LEFT, padx=6)

        self._stream_btn = ttk.Button(top, text="Stream ON", command=self._toggle_stream,
                                       state=tk.DISABLED, width=12)
        self._stream_btn.pack(side=tk.LEFT, padx=3)

        self._status_lbl = ttk.Label(top, text="● Disconnected", foreground="red",
                                      font=("Arial", 9, "bold"))
        self._status_lbl.pack(side=tk.LEFT, padx=10)

        # ── Main area: left = telemetry + canvas, right = controls ───────────
        main = ttk.Frame(self.root, padding=(5, 3))
        main.pack(fill=tk.BOTH, expand=True)

        self._build_left(main)
        self._build_right(main)

        # ── Bottom: log ───────────────────────────────────────────────────────
        log_frm = ttk.LabelFrame(self.root, text="Log", padding=4)
        log_frm.pack(fill=tk.X, padx=5, pady=(0, 5))
        self._log_widget = scrolledtext.ScrolledText(log_frm, height=5,
                                                      state=tk.DISABLED,
                                                      font=("Courier", 9),
                                                      wrap=tk.WORD)
        self._log_widget.pack(fill=tk.X)

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_left(self, parent):
        left = ttk.Frame(parent)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))

        # Telemetry grid
        tele_frm = ttk.LabelFrame(left, text="Telemetry", padding=6)
        tele_frm.pack(fill=tk.X)

        fields = [
            ("v (m/s)",     "v"),
            ("ω (rad/s)",   "w"),
            ("x (m)",       "x"),
            ("y (m)",       "y"),
            ("yaw (°)",     "yaw"),
            ("l_speed",     "l_speed"),
            ("r_speed",     "r_speed"),
            ("l_pwm",       "l_pwm"),
            ("r_pwm",       "r_pwm"),
        ]
        self._tele_vars: dict[str, tk.StringVar] = {}
        for i, (lbl, key) in enumerate(fields):
            row, col = divmod(i, 3)
            ttk.Label(tele_frm, text=f"{lbl}:", anchor=tk.E, width=11).grid(
                row=row, column=col * 2, sticky=tk.E, padx=(4, 0))
            var = tk.StringVar(value="—")
            self._tele_vars[key] = var
            ttk.Label(tele_frm, textvariable=var, width=9,
                      font=("Courier", 10), foreground="#0055cc").grid(
                row=row, column=col * 2 + 1, sticky=tk.W, padx=(2, 8))

        # Canvas
        canvas_frm = ttk.LabelFrame(left, text="Trajectory (top-down)", padding=4)
        canvas_frm.pack(fill=tk.BOTH, expand=True, pady=(6, 0))

        self._canvas = tk.Canvas(canvas_frm, bg="white", relief=tk.SUNKEN, bd=1)
        self._canvas.pack(fill=tk.BOTH, expand=True)

        ttk.Button(canvas_frm, text="Clear Path",
                   command=self._clear_trail).pack(side=tk.RIGHT, pady=3)

    def _build_right(self, parent):
        right = ttk.Frame(parent)
        right.pack(side=tk.LEFT, fill=tk.Y)

        # ── Velocity sliders ──────────────────────────────────────────────────
        ctrl = ttk.LabelFrame(right, text="Velocity Control", padding=8)
        ctrl.pack(fill=tk.X, pady=(0, 6))

        # v slider
        ttk.Label(ctrl, text=f"Linear  v  (m/s)   [{-V_MAX:.2f} … +{V_MAX:.2f}]",
                  font=("Arial", 9)).pack(anchor=tk.W)
        self._v_var = tk.DoubleVar(value=0.0)
        ttk.Scale(ctrl, from_=-V_MAX, to=V_MAX, orient=tk.HORIZONTAL,
                  variable=self._v_var, length=280,
                  command=lambda _: self._on_slider()).pack(fill=tk.X)
        self._v_lbl = ttk.Label(ctrl, text="v =  0.000 m/s",
                                 foreground="#0055cc", font=("Courier", 10))
        self._v_lbl.pack(pady=(0, 4))

        ttk.Separator(ctrl, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=4)

        # w slider
        ttk.Label(ctrl, text=f"Angular ω (rad/s) [{-W_MAX:.2f} … +{W_MAX:.2f}]",
                  font=("Arial", 9)).pack(anchor=tk.W)
        self._w_var = tk.DoubleVar(value=0.0)
        ttk.Scale(ctrl, from_=-W_MAX, to=W_MAX, orient=tk.HORIZONTAL,
                  variable=self._w_var, length=280,
                  command=lambda _: self._on_slider()).pack(fill=tk.X)
        self._w_lbl = ttk.Label(ctrl, text="ω =  0.000 rad/s",
                                 foreground="#0055cc", font=("Courier", 10))
        self._w_lbl.pack(pady=(0, 4))

        ttk.Separator(ctrl, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=4)

        # Action buttons
        stop_btn = tk.Button(ctrl, text="⬛  STOP", bg="#cc0000", fg="white",
                              font=("Arial", 13, "bold"), height=2,
                              activebackground="#ff3333",
                              command=self._emergency_stop)
        stop_btn.pack(fill=tk.X, pady=4)

        btn_row = ttk.Frame(ctrl)
        btn_row.pack(fill=tk.X)
        ttk.Button(btn_row, text="Zero Sliders",
                   command=self._zero_sliders).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        ttk.Button(btn_row, text="Reset Odom",
                   command=self._reset_odom).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        # ── PID tuning ────────────────────────────────────────────────────────
        pid_frm = ttk.LabelFrame(right, text="PID Tuning", padding=6)
        pid_frm.pack(fill=tk.X, pady=(0, 4))

        motor_rows = [
            ("Left",  (10.0, 2.0, 0.1)),
            ("Right", (10.0, 5.0, 0.1)),   # higher Ki — right motor undershoots
            ("Both",  (10.0, 2.0, 0.1)),
        ]
        for motor, (kp0, ki0, kd0) in motor_rows:
            row = ttk.Frame(pid_frm)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=f"{motor}:", width=5).pack(side=tk.LEFT)
            entries = []
            for lbl, default in [("Kp", kp0), ("Ki", ki0), ("Kd", kd0)]:
                ttk.Label(row, text=lbl).pack(side=tk.LEFT)
                e = ttk.Entry(row, width=6)
                e.insert(0, str(default))
                e.pack(side=tk.LEFT, padx=2)
                entries.append(e)
            ttk.Button(row, text="Apply",
                       command=lambda m=motor.lower(), es=entries: self._apply_pid(m, es)
                       ).pack(side=tk.LEFT, padx=4)

    # ── Serial connection ─────────────────────────────────────────────────────

    def _toggle_connect(self):
        if self.connected:
            self._disconnect()
        else:
            self._connect()

    def _connect(self):
        port = self._port_var.get().strip()
        try:
            baud = int(self._baud_var.get().strip())
        except ValueError:
            self._log("Invalid baud rate")
            return
        try:
            ser = serial.Serial(port, baud, timeout=1.0, dsrdtr=False)
            ser.dtr = False
            ser.rts = False
            time.sleep(0.1)
            ser.reset_input_buffer()

            # Ping handshake
            ok = False
            for _ in range(6):
                ser.write(b"PING\n")
                ser.flush()
                deadline = time.time() + 1.2
                while time.time() < deadline:
                    line = ser.readline().decode(errors="replace").strip()
                    if line == "READY":
                        ok = True
                        break
                if ok:
                    break

            if not ok:
                ser.close()
                self._log(f"No READY from {port} — check port and firmware")
                return

            self.ser      = ser
            self.connected = True
            self._running  = True
            self._rx_thread = threading.Thread(target=self._rx_loop, daemon=True,
                                                name="esp32-rx")
            self._rx_thread.start()

            self._conn_btn.config(text="Disconnect")
            self._stream_btn.config(state=tk.NORMAL)
            self._status_lbl.config(text=f"● Connected: {port}", foreground="green")
            self._log(f"Connected to {port} @ {baud}")

        except serial.SerialException as exc:
            self._log(f"Serial error: {exc}")

    def _disconnect(self):
        self._running   = False
        self._streaming = False
        if self.ser:
            try:
                self.ser.write(b"STOP\n")
                self.ser.flush()
                time.sleep(0.1)
                self.ser.close()
            except Exception:
                pass
        self.ser        = None
        self.connected  = False
        self._conn_btn.config(text="Connect")
        self._stream_btn.config(text="Stream ON", state=tk.DISABLED)
        self._status_lbl.config(text="● Disconnected", foreground="red")
        self._log("Disconnected.")

    # ── Serial receiver (background thread) ──────────────────────────────────

    def _rx_loop(self):
        while self._running and self.ser and self.ser.is_open:
            try:
                raw = self.ser.readline()
                if not raw:
                    continue
                line = raw.decode(errors="replace").strip()

                if line.startswith("ODOM,"):
                    parts = line.split(",")
                    if len(parts) == 14:
                        try:
                            odom = {
                                "t_ms":    int(parts[1]),
                                "l_speed": float(parts[2]),
                                "r_speed": float(parts[3]),
                                "l_pwm":   int(parts[4]),
                                "r_pwm":   int(parts[5]),
                                "l_sp":    float(parts[6]),
                                "r_sp":    float(parts[7]),
                                "v":       float(parts[8]),
                                "w":       float(parts[9]),
                                "x":       float(parts[10]),
                                "y":       float(parts[11]),
                                "yaw":     float(parts[12]),
                            }
                            with self._data_lock:
                                self._odom = odom
                                self._trail.append((odom["x"], odom["y"]))
                                if len(self._trail) > MAX_TRAIL:
                                    self._trail.pop(0)
                        except (ValueError, IndexError):
                            pass

                elif line == "WATCHDOG":
                    self._log("⚠ WATCHDOG – firmware stopped motors (no CMD_VEL)")
                elif line:
                    self._log(f"FW: {line}")

            except (serial.SerialException, OSError):
                break

    # ── Commands ──────────────────────────────────────────────────────────────

    def _send(self, cmd: str):
        if self.ser and self.ser.is_open:
            try:
                with self._ser_lock:
                    self.ser.write(f"{cmd}\n".encode())
                    self.ser.flush()
            except Exception as exc:
                self._log(f"Send error: {exc}")

    def _toggle_stream(self):
        if not self.connected:
            return
        if self._streaming:
            self._send("STREAM_OFF")
            self._streaming = False
            self._stream_btn.config(text="Stream ON")
        else:
            self._send("STREAM_ON")
            self._streaming = True
            self._stream_btn.config(text="Stream OFF")

    def _emergency_stop(self):
        self._zero_sliders()
        self._send("STOP")
        self._streaming = False
        self._stream_btn.config(text="Stream ON")
        self._log("EMERGENCY STOP sent.")

    def _zero_sliders(self):
        self._v_var.set(0.0)
        self._w_var.set(0.0)
        self._v = 0.0
        self._w = 0.0
        self._v_lbl.config(text="v =  0.000 m/s")
        self._w_lbl.config(text="ω =  0.000 rad/s")

    def _reset_odom(self):
        self._send("RESET_ODOM")
        with self._data_lock:
            self._trail.clear()
        self._log("Odometry reset.")

    def _clear_trail(self):
        with self._data_lock:
            self._trail.clear()

    def _apply_pid(self, motor: str, entries: list):
        try:
            kp = float(entries[0].get())
            ki = float(entries[1].get())
            kd = float(entries[2].get())
        except ValueError:
            self._log("PID error: enter numeric values")
            return
        cmd_map = {"left": "TUNEL", "right": "TUNER", "both": "TUNE"}
        self._send(f"{cmd_map[motor]}:{kp},{ki},{kd}")
        self._log(f"PID {motor}: Kp={kp}  Ki={ki}  Kd={kd}")

    # ── Slider callback ───────────────────────────────────────────────────────

    def _on_slider(self):
        self._v = round(self._v_var.get(), 3)
        self._w = round(self._w_var.get(), 3)
        self._v_lbl.config(text=f"v = {self._v:+.3f} m/s")
        self._w_lbl.config(text=f"ω = {self._w:+.3f} rad/s")

    # ── Periodic GUI update (main thread, 100 ms) ─────────────────────────────

    def _poll(self):
        # Send CMD_VEL keepalive at 10 Hz when streaming
        # (firmware watchdog triggers after 500 ms silence)
        if self.connected and self._streaming:
            self._send(f"CMD_VEL:{self._v:.4f},{self._w:.4f}")

        self._update_telemetry()
        self._draw_canvas()
        self._flush_log()

        self.root.after(100, self._poll)

    def _update_telemetry(self):
        with self._data_lock:
            odom = dict(self._odom)
        if not odom:
            return
        self._tele_vars["v"].set(       f"{odom['v']:+.3f}")
        self._tele_vars["w"].set(       f"{odom['w']:+.3f}")
        self._tele_vars["x"].set(       f"{odom['x']:+.4f}")
        self._tele_vars["y"].set(       f"{odom['y']:+.4f}")
        self._tele_vars["yaw"].set(     f"{math.degrees(odom['yaw']):+.1f}")
        self._tele_vars["l_speed"].set( f"{odom['l_speed']:+.3f}")
        self._tele_vars["r_speed"].set( f"{odom['r_speed']:+.3f}")
        self._tele_vars["l_pwm"].set(   str(odom['l_pwm']))
        self._tele_vars["r_pwm"].set(   str(odom['r_pwm']))

    def _draw_canvas(self):
        c = self._canvas
        W = c.winfo_width()
        H = c.winfo_height()
        c.delete("all")
        if W < 20 or H < 20:
            return

        cx, cy = W // 2, H // 2

        # Grid lines
        c.create_line(cx, 0, cx, H, fill="#e0e0e0")
        c.create_line(0, cy, W, cy, fill="#e0e0e0")
        c.create_text(cx + 4, 6, text="+Y", fill="#aaa", anchor=tk.NW, font=("Arial", 7))
        c.create_text(W - 4, cy + 4, text="+X", fill="#aaa", anchor=tk.NE, font=("Arial", 7))

        with self._data_lock:
            trail = list(self._trail)
            odom  = dict(self._odom)

        if not trail:
            c.create_text(cx, cy, text="(no data — enable Stream)", fill="#bbb",
                          font=("Arial", 10))
            return

        xs = [p[0] for p in trail]
        ys = [p[1] for p in trail]
        span = max(max(xs) - min(xs), max(ys) - min(ys), 0.5)
        scale = min(W, H) * 0.8 / span

        x_mid = (max(xs) + min(xs)) / 2
        y_mid = (max(ys) + min(ys)) / 2

        def to_px(x, y):
            # Y axis pointing up on canvas (inverted screen coords)
            return (cx + (x - x_mid) * scale,
                    cy - (y - y_mid) * scale)

        # Trail
        if len(trail) > 1:
            pts = []
            for x, y in trail:
                pts += list(to_px(x, y))
            c.create_line(*pts, fill="#4488dd", width=1.5, smooth=True)

        # Start (green dot)
        sx, sy = to_px(trail[0][0], trail[0][1])
        c.create_oval(sx - 5, sy - 5, sx + 5, sy + 5, fill="#22aa44", outline="")

        # Current pose (red dot + heading arrow)
        if odom:
            rx, ry = to_px(odom.get("x", 0), odom.get("y", 0))
            yaw    = odom.get("yaw", 0)
            alen   = 14
            ax     = rx + alen * math.cos(yaw)
            ay     = ry - alen * math.sin(yaw)
            c.create_oval(rx - 6, ry - 6, rx + 6, ry + 6,
                          fill="#dd2222", outline="white", width=1)
            c.create_line(rx, ry, ax, ay, fill="#dd2222", width=2, arrow=tk.LAST)

        # Scale bar (0.5 m)
        bar_m   = 0.5
        bar_px  = bar_m * scale
        bx1, by = W - 10 - bar_px, H - 14
        c.create_line(bx1, by, bx1 + bar_px, by, fill="#666", width=2)
        c.create_text(bx1 + bar_px / 2, by - 8, text="0.5 m",
                      fill="#666", font=("Arial", 7))

    # ── Log helpers ───────────────────────────────────────────────────────────

    def _log(self, msg: str):
        self._log_q.put(msg)

    def _flush_log(self):
        while not self._log_q.empty():
            msg = self._log_q.get_nowait()
            self._log_widget.config(state=tk.NORMAL)
            self._log_widget.insert(tk.END, f"{msg}\n")
            self._log_widget.see(tk.END)
            self._log_widget.config(state=tk.DISABLED)

    # ── Window close ─────────────────────────────────────────────────────────

    def _on_close(self):
        self._disconnect()
        self.root.destroy()


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Diff-drive GUI tester")
    parser.add_argument("--port", default="/dev/ttyUSB0",
                        help="Serial port (default: /dev/ttyUSB0)")
    args = parser.parse_args()

    root = tk.Tk()
    root.geometry("900x680")
    root.minsize(720, 560)
    DiffDriveGUI(root, default_port=args.port)
    root.mainloop()


if __name__ == "__main__":
    main()
