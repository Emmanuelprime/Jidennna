import tkinter as tk
from tkinter import ttk, scrolledtext
import serial
import serial.tools.list_ports
import threading
import time
import re
from datetime import datetime

class RobotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Robot Control Panel")
        self.root.geometry("900x700")
        
        # Serial connection variables
        self.serial_port = None
        self.is_connected = False
        self.reading_thread = None
        self.running = False
        
        # Robot state
        self.current_speed = 0.0
        self.current_omega = 0.0
        self.left_speed = 0.0
        self.right_speed = 0.0
        self.left_pwm = 0
        self.right_pwm = 0
        self.actual_linear = 0.0
        self.actual_omega = 0.0
        
        # Create GUI
        self.create_widgets()
        
        # Auto-detect available ports
        self.refresh_ports()
        
    def create_widgets(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Connection Frame
        conn_frame = ttk.LabelFrame(main_frame, text="Connection", padding="5")
        conn_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(conn_frame, text="Port:").grid(row=0, column=0, padx=5)
        self.port_var = tk.StringVar()
        self.port_combo = ttk.Combobox(conn_frame, textvariable=self.port_var, width=15)
        self.port_combo.grid(row=0, column=1, padx=5)
        
        ttk.Label(conn_frame, text="Baud:").grid(row=0, column=2, padx=5)
        self.baud_var = tk.StringVar(value="115200")
        baud_combo = ttk.Combobox(conn_frame, textvariable=self.baud_var, 
                                  values=["9600", "19200", "38400", "57600", "115200"], 
                                  width=10)
        baud_combo.grid(row=0, column=3, padx=5)
        
        self.connect_btn = ttk.Button(conn_frame, text="Connect", command=self.toggle_connection)
        self.connect_btn.grid(row=0, column=4, padx=5)
        
        self.refresh_btn = ttk.Button(conn_frame, text="Refresh", command=self.refresh_ports)
        self.refresh_btn.grid(row=0, column=5, padx=5)
        
        self.status_label = ttk.Label(conn_frame, text="Disconnected", foreground="red")
        self.status_label.grid(row=0, column=6, padx=10)
        
        # Control Frame
        ctrl_frame = ttk.LabelFrame(main_frame, text="Motion Control", padding="10")
        ctrl_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # Forward/Reverse speed control
        ttk.Label(ctrl_frame, text="Speed (m/s):").grid(row=0, column=0, padx=5, pady=5)
        self.speed_var = tk.DoubleVar(value=0.5)
        speed_scale = ttk.Scale(ctrl_frame, from_=0.0, to=1.2, variable=self.speed_var, 
                               orient=tk.HORIZONTAL, length=200)
        speed_scale.grid(row=0, column=1, padx=5, pady=5)
        self.speed_label = ttk.Label(ctrl_frame, text="0.50")
        self.speed_label.grid(row=0, column=2, padx=5)
        speed_scale.configure(command=lambda x: self.speed_label.configure(text=f"{float(x):.2f}"))
        
        # Direction buttons
        btn_frame = ttk.Frame(ctrl_frame)
        btn_frame.grid(row=1, column=0, columnspan=3, pady=10)
        
        ttk.Button(btn_frame, text="⬆ Forward", command=lambda: self.send_command('F'), 
                  width=12).grid(row=0, column=0, padx=5)
        ttk.Button(btn_frame, text="⬇ Reverse", command=lambda: self.send_command('R'), 
                  width=12).grid(row=0, column=1, padx=5)
        ttk.Button(btn_frame, text="↺ Spin Left", command=lambda: self.send_command('L'), 
                  width=12).grid(row=0, column=2, padx=5)
        ttk.Button(btn_frame, text="↻ Spin Right", command=lambda: self.send_command('B'), 
                  width=12).grid(row=0, column=3, padx=5)
        ttk.Button(btn_frame, text="⏹ Stop", command=lambda: self.send_command('s'), 
                  width=12).grid(row=0, column=4, padx=5)
        ttk.Button(btn_frame, text="Zero Encoders", command=lambda: self.send_command('z'), 
                  width=12).grid(row=0, column=5, padx=5)
        
        # Spin speed control
        ttk.Label(ctrl_frame, text="Spin Speed (rad/s):").grid(row=2, column=0, padx=5, pady=5)
        self.omega_var = tk.DoubleVar(value=1.0)
        omega_scale = ttk.Scale(ctrl_frame, from_=0.0, to=2.0, variable=self.omega_var, 
                               orient=tk.HORIZONTAL, length=200)
        omega_scale.grid(row=2, column=1, padx=5, pady=5)
        self.omega_label = ttk.Label(ctrl_frame, text="1.00")
        self.omega_label.grid(row=2, column=2, padx=5)
        omega_scale.configure(command=lambda x: self.omega_label.configure(text=f"{float(x):.2f}"))
        
        # Manual command input
        ttk.Label(ctrl_frame, text="Manual Command:").grid(row=3, column=0, padx=5, pady=5)
        self.cmd_entry = ttk.Entry(ctrl_frame, width=20)
        self.cmd_entry.grid(row=3, column=1, padx=5, pady=5)
        ttk.Button(ctrl_frame, text="Send", command=self.send_manual_command).grid(row=3, column=2, padx=5)
        
        # Status Frame (Right side)
        status_frame = ttk.LabelFrame(main_frame, text="Robot Status", padding="10")
        status_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10), padx=(10, 0))
        
        self.status_text = scrolledtext.ScrolledText(status_frame, width=40, height=12, font=('Courier', 9))
        self.status_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Telemetry Frame
        telemetry_frame = ttk.LabelFrame(main_frame, text="Telemetry", padding="10")
        telemetry_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Create telemetry grid
        telemetry_labels = [
            ("Left Speed:", "left_speed", "0.000"),
            ("Right Speed:", "right_speed", "0.000"),
            ("Left PWM:", "left_pwm", "0"),
            ("Right PWM:", "right_pwm", "0"),
            ("Actual Linear:", "actual_linear", "0.000"),
            ("Actual Omega:", "actual_omega", "0.000")
        ]
        
        self.telemetry_vars = {}
        for i, (label, key, default) in enumerate(telemetry_labels):
            row = i // 3
            col = (i % 3) * 2
            ttk.Label(telemetry_frame, text=label).grid(row=row, column=col, padx=5, pady=2, sticky=tk.E)
            var = tk.StringVar(value=default)
            self.telemetry_vars[key] = var
            ttk.Label(telemetry_frame, textvariable=var, font=('Courier', 10), 
                     width=10).grid(row=row, column=col+1, padx=5, pady=2, sticky=tk.W)
        
        # Console Frame
        console_frame = ttk.LabelFrame(main_frame, text="Console Output", padding="5")
        console_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.console_text = scrolledtext.ScrolledText(console_frame, height=10, font=('Courier', 9))
        self.console_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(3, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
    def refresh_ports(self):
        ports = [port.device for port in serial.tools.list_ports.comports()]
        self.port_combo['values'] = ports
        if ports:
            self.port_combo.set(ports[0])
        
    def toggle_connection(self):
        if not self.is_connected:
            self.connect()
        else:
            self.disconnect()
    
    def connect(self):
        try:
            port = self.port_var.get()
            baud = int(self.baud_var.get())
            
            self.serial_port = serial.Serial(port, baud, timeout=0.1)
            self.is_connected = True
            self.connect_btn.configure(text="Disconnect")
            self.status_label.configure(text="Connected", foreground="green")
            
            # Start reading thread
            self.running = True
            self.reading_thread = threading.Thread(target=self.read_serial, daemon=True)
            self.reading_thread.start()
            
            # Send PING to check connection
            self.send_command("PING")
            
            self.log_console(f"Connected to {port} at {baud} baud")
            
        except Exception as e:
            self.log_console(f"Connection error: {str(e)}")
            self.status_label.configure(text="Error", foreground="red")
    
    def disconnect(self):
        self.running = False
        if self.reading_thread:
            self.reading_thread.join(timeout=1)
        
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
        
        self.is_connected = False
        self.connect_btn.configure(text="Connect")
        self.status_label.configure(text="Disconnected", foreground="red")
        self.log_console("Disconnected")
    
    def send_command(self, cmd):
        if not self.is_connected or not self.serial_port:
            self.log_console("Not connected")
            return
        
        try:
            if cmd in ['F', 'R']:
                speed = self.speed_var.get()
                full_cmd = f"{cmd}{speed:.3f}\n"
            elif cmd in ['L', 'B']:
                omega = self.omega_var.get()
                full_cmd = f"{cmd}{omega:.3f}\n"
            else:
                full_cmd = f"{cmd}\n"
            
            self.serial_port.write(full_cmd.encode())
            self.log_console(f"Sent: {full_cmd.strip()}")
            
        except Exception as e:
            self.log_console(f"Send error: {str(e)}")
            self.disconnect()
    
    def send_manual_command(self):
        cmd = self.cmd_entry.get()
        if cmd:
            self.send_command(cmd)
            self.cmd_entry.delete(0, tk.END)
    
    def read_serial(self):
        buffer = ""
        while self.running:
            try:
                if self.serial_port and self.serial_port.in_waiting:
                    data = self.serial_port.read(self.serial_port.in_waiting)
                    buffer += data.decode('utf-8', errors='ignore')
                    
                    # Process complete lines
                    lines = buffer.split('\n')
                    buffer = lines[-1]  # Keep incomplete line
                    
                    for line in lines[:-1]:
                        line = line.strip()
                        if line:
                            self.process_serial_line(line)
                            
                time.sleep(0.01)
                
            except Exception as e:
                if self.running:
                    self.log_console(f"Read error: {str(e)}")
                    self.running = False
                    self.root.after(0, self.disconnect)
                break
    
    def process_serial_line(self, line):
        # Log to console (except telemetry data)
        if not line.startswith('CNT,'):
            self.log_console(f"< {line}")
        
        # Parse telemetry data
        if line.startswith('CNT,'):
            try:
                parts = line.split(',')
                if len(parts) >= 8:
                    # CNT,time,vL,vR,linear,omega,leftPWM,rightPWM
                    self.left_speed = float(parts[2])
                    self.right_speed = float(parts[3])
                    self.actual_linear = float(parts[4])
                    self.actual_omega = float(parts[5])
                    self.left_pwm = int(parts[6])
                    self.right_pwm = int(parts[7])
                    
                    # Update telemetry display
                    self.root.after(0, self.update_telemetry)
                    
            except (ValueError, IndexError) as e:
                pass
        
        # Check for READY response
        if line == "READY":
            self.log_console("Robot ready")
    
    def update_telemetry(self):
        self.telemetry_vars['left_speed'].set(f"{self.left_speed:.3f}")
        self.telemetry_vars['right_speed'].set(f"{self.right_speed:.3f}")
        self.telemetry_vars['left_pwm'].set(f"{self.left_pwm}")
        self.telemetry_vars['right_pwm'].set(f"{self.right_pwm}")
        self.telemetry_vars['actual_linear'].set(f"{self.actual_linear:.3f}")
        self.telemetry_vars['actual_omega'].set(f"{self.actual_omega:.3f}")
    
    def log_console(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.root.after(0, lambda: self._append_console(f"[{timestamp}] {message}\n"))
    
    def _append_console(self, message):
        self.console_text.insert(tk.END, message)
        self.console_text.see(tk.END)
        # Limit console size
        if int(self.console_text.index('end-1c').split('.')[0]) > 1000:
            self.console_text.delete('1.0', '2.0')
    
    def on_closing(self):
        self.running = False
        self.disconnect()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = RobotGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()