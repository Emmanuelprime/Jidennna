import tkinter as tk
from tkinter import ttk, scrolledtext
import serial
import serial.tools.list_ports
import threading
import time
import sys
import os
from datetime import datetime

class RobotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Robot Control Panel - Cross Platform")
        self.root.geometry("900x700")
        
        # Detect platform
        self.is_windows = sys.platform.startswith('win')
        self.is_linux = sys.platform.startswith('linux')
        self.is_jetson = self.is_linux and os.path.exists('/proc/device-tree/model')
        
        # Platform-specific settings
        if self.is_jetson:
            self.root.title("Robot Control Panel - Jetson Nano")
            self.default_baud = 115200
            self.default_ports = ['/dev/ttyUSB0', '/dev/ttyACM0', '/dev/ttyTHS1']
        elif self.is_windows:
            self.default_baud = 115200
            self.default_ports = ['COM3', 'COM4', 'COM5', 'COM6']
        else:  # Linux
            self.default_baud = 115200
            self.default_ports = ['/dev/ttyUSB0', '/dev/ttyACM0']
        
        # Serial connection variables
        self.serial_port = None
        self.is_connected = False
        self.reading_thread = None
        self.running = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 3
        
        # Robot state
        self.current_speed = 0.0
        self.current_omega = 0.0
        self.left_speed = 0.0
        self.right_speed = 0.0
        self.left_pwm = 0
        self.right_pwm = 0
        self.actual_linear = 0.0
        self.actual_omega = 0.0
        
        # Connection monitoring
        self.last_data_time = time.time()
        self.connection_timeout = 10.0  # Increased timeout
        self.ping_interval = 2.0
        self.last_ping_time = time.time()
        self.ping_retries = 0
        self.max_ping_retries = 3
        
        # Create GUI
        self.create_widgets()
        
        # Auto-detect available ports
        self.refresh_ports()
        
        # Start connection monitor
        self.monitor_connection()
        
    def create_widgets(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Platform info
        platform_text = "Jetson Nano" if self.is_jetson else ("Windows" if self.is_windows else "Linux")
        ttk.Label(main_frame, text=f"Platform: {platform_text}", font=('Arial', 10, 'bold')).grid(
            row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 5))
        
        # Connection Frame
        conn_frame = ttk.LabelFrame(main_frame, text="Connection", padding="5")
        conn_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(conn_frame, text="Port:").grid(row=0, column=0, padx=5)
        self.port_var = tk.StringVar()
        self.port_combo = ttk.Combobox(conn_frame, textvariable=self.port_var, width=20)
        self.port_combo.grid(row=0, column=1, padx=5)
        
        ttk.Label(conn_frame, text="Baud:").grid(row=0, column=2, padx=5)
        self.baud_var = tk.StringVar(value=str(self.default_baud))
        baud_combo = ttk.Combobox(conn_frame, textvariable=self.baud_var, 
                                  values=["9600", "19200", "38400", "57600", "115200", "230400"], 
                                  width=10)
        baud_combo.grid(row=0, column=3, padx=5)
        
        self.connect_btn = ttk.Button(conn_frame, text="Connect", command=self.toggle_connection)
        self.connect_btn.grid(row=0, column=4, padx=5)
        
        self.refresh_btn = ttk.Button(conn_frame, text="Refresh", command=self.refresh_ports)
        self.refresh_btn.grid(row=0, column=5, padx=5)
        
        self.status_label = ttk.Label(conn_frame, text="Disconnected", foreground="red")
        self.status_label.grid(row=0, column=6, padx=10)
        
        # Debug button
        self.debug_btn = ttk.Button(conn_frame, text="Debug", command=self.debug_connection)
        self.debug_btn.grid(row=0, column=7, padx=5)
        
        # Control Frame
        ctrl_frame = ttk.LabelFrame(main_frame, text="Motion Control", padding="10")
        ctrl_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # Forward/Reverse speed control
        ttk.Label(ctrl_frame, text="Speed (m/s):").grid(row=0, column=0, padx=5, pady=5)
        self.speed_var = tk.DoubleVar(value=0.5)
        speed_scale = ttk.Scale(ctrl_frame, from_=0.0, to=1.2, variable=self.speed_var, 
                               orient=tk.HORIZONTAL, length=200)
        speed_scale.grid(row=0, column=1, padx=5, pady=5)
        self.speed_label = ttk.Label(ctrl_frame, text="0.50", width=6)
        self.speed_label.grid(row=0, column=2, padx=5)
        speed_scale.configure(command=lambda x: self.speed_label.configure(text=f"{float(x):.2f}"))
        
        # Quick speed buttons
        speed_btn_frame = ttk.Frame(ctrl_frame)
        speed_btn_frame.grid(row=1, column=0, columnspan=3, pady=2)
        for speed in [0.2, 0.5, 0.8, 1.0, 1.2]:
            ttk.Button(speed_btn_frame, text=f"{speed:.1f}", width=5,
                      command=lambda s=speed: self.set_speed(s)).pack(side=tk.LEFT, padx=2)
        
        # Direction buttons
        btn_frame = ttk.Frame(ctrl_frame)
        btn_frame.grid(row=2, column=0, columnspan=3, pady=10)
        
        ttk.Button(btn_frame, text="Forward", command=lambda: self.send_command('F'), 
                  width=10).grid(row=0, column=0, padx=5)
        ttk.Button(btn_frame, text="Reverse", command=lambda: self.send_command('R'), 
                  width=10).grid(row=0, column=1, padx=5)
        ttk.Button(btn_frame, text="Spin L", command=lambda: self.send_command('L'), 
                  width=10).grid(row=0, column=2, padx=5)
        ttk.Button(btn_frame, text="Spin R", command=lambda: self.send_command('B'), 
                  width=10).grid(row=0, column=3, padx=5)
        ttk.Button(btn_frame, text="Stop", command=lambda: self.send_command('s'), 
                  width=10).grid(row=0, column=4, padx=5)
        ttk.Button(btn_frame, text="Zero", command=lambda: self.send_command('z'), 
                  width=10).grid(row=0, column=5, padx=5)
        
        # Spin speed control
        ttk.Label(ctrl_frame, text="Spin Speed (rad/s):").grid(row=3, column=0, padx=5, pady=5)
        self.omega_var = tk.DoubleVar(value=1.0)
        omega_scale = ttk.Scale(ctrl_frame, from_=0.0, to=2.0, variable=self.omega_var, 
                               orient=tk.HORIZONTAL, length=200)
        omega_scale.grid(row=3, column=1, padx=5, pady=5)
        self.omega_label = ttk.Label(ctrl_frame, text="1.00", width=6)
        self.omega_label.grid(row=3, column=2, padx=5)
        omega_scale.configure(command=lambda x: self.omega_label.configure(text=f"{float(x):.2f}"))
        
        # Manual command input
        ttk.Label(ctrl_frame, text="Manual Command:").grid(row=4, column=0, padx=5, pady=5)
        self.cmd_entry = ttk.Entry(ctrl_frame, width=20)
        self.cmd_entry.grid(row=4, column=1, padx=5, pady=5)
        ttk.Button(ctrl_frame, text="Send", command=self.send_manual_command).grid(row=4, column=2, padx=5)
        
        # Status Frame (Right side)
        status_frame = ttk.LabelFrame(main_frame, text="Robot Status", padding="10")
        status_frame.grid(row=2, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10), padx=(10, 0))
        
        self.status_text = scrolledtext.ScrolledText(status_frame, width=40, height=12, font=('Courier', 9))
        self.status_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Telemetry Frame
        telemetry_frame = ttk.LabelFrame(main_frame, text="Telemetry", padding="10")
        telemetry_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Create telemetry grid
        telemetry_labels = [
            ("Left Speed (m/s):", "left_speed", "0.000"),
            ("Right Speed (m/s):", "right_speed", "0.000"),
            ("Left PWM:", "left_pwm", "0"),
            ("Right PWM:", "right_pwm", "0"),
            ("Actual Linear (m/s):", "actual_linear", "0.000"),
            ("Actual Omega (rad/s):", "actual_omega", "0.000")
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
        console_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.console_text = scrolledtext.ScrolledText(console_frame, height=10, font=('Courier', 9))
        self.console_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(4, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
    def set_speed(self, speed):
        """Set the speed slider to a specific value"""
        self.speed_var.set(speed)
        self.speed_label.configure(text=f"{speed:.2f}")
        
    def refresh_ports(self):
        """Refresh the list of available serial ports"""
        ports = []
        try:
            # Get all available ports
            available_ports = [port.device for port in serial.tools.list_ports.comports()]
            
            # On Jetson/Linux, also check for common USB devices
            if self.is_linux:
                import glob
                usb_ports = glob.glob('/dev/ttyUSB*') + glob.glob('/dev/ttyACM*')
                for port in usb_ports:
                    if port not in available_ports:
                        available_ports.append(port)
            
            ports = available_ports
            
            # If no ports found, use default list
            if not ports:
                ports = self.default_ports
            
            self.port_combo['values'] = ports
            
            # Try to select the first available port or keep current selection
            if self.port_var.get() in ports:
                self.port_combo.set(self.port_var.get())
            elif ports:
                self.port_combo.set(ports[0])
                
        except Exception as e:
            self.log_console(f"Error refreshing ports: {str(e)}")
            # Fallback to default ports
            self.port_combo['values'] = self.default_ports
            if self.default_ports:
                self.port_combo.set(self.default_ports[0])
        
    def toggle_connection(self):
        if not self.is_connected:
            self.connect()
        else:
            self.disconnect()
    
    def connect(self):
        try:
            port = self.port_var.get()
            baud = int(self.baud_var.get())
            
            self.log_console(f"Attempting to connect to {port} at {baud} baud...")
            
            # Platform-specific connection settings
            if self.is_linux:
                # On Linux, we might need to wait for the port to be ready
                time.sleep(0.1)
            
            # Open serial port with more robust settings
            self.serial_port = serial.Serial(
                port=port,
                baudrate=baud,
                timeout=1.0,  # Longer timeout for better reading
                write_timeout=1.0,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                xonxoff=False,
                rtscts=False,
                dsrdtr=False
            )
            
            # Wait for connection to stabilize
            time.sleep(2.0)  # Increased wait time
            
            # Clear any pending data
            self.serial_port.reset_input_buffer()
            self.serial_port.reset_output_buffer()
            
            self.is_connected = True
            self.reconnect_attempts = 0
            self.ping_retries = 0
            self.last_data_time = time.time()
            self.last_ping_time = time.time()
            self.connect_btn.configure(text="Disconnect")
            self.status_label.configure(text="Connected", foreground="green")
            
            # Start reading thread
            self.running = True
            self.reading_thread = threading.Thread(target=self.read_serial, daemon=True)
            self.reading_thread.start()
            
            # Send initial PING to check connection
            time.sleep(0.5)
            self.send_command("PING")
            
            self.log_console(f"Connected to {port} at {baud} baud")
            
        except serial.SerialException as e:
            error_msg = str(e)
            if "Access denied" in error_msg or "Permission denied" in error_msg:
                self.log_console(f"Permission denied. On Linux/Jetson, try: sudo chmod 666 {port}")
                self.log_console(f"Or add user to dialout group: sudo usermod -a -G dialout $USER")
            else:
                self.log_console(f"Connection error: {error_msg}")
            self.status_label.configure(text="Error", foreground="red")
            self.is_connected = False
            
        except Exception as e:
            self.log_console(f"Connection error: {str(e)}")
            self.status_label.configure(text="Error", foreground="red")
            self.is_connected = False
    
    def disconnect(self):
        self.running = False
        if self.reading_thread:
            self.reading_thread.join(timeout=2)
        
        if self.serial_port and self.serial_port.is_open:
            try:
                self.serial_port.close()
            except:
                pass
        
        self.is_connected = False
        self.connect_btn.configure(text="Connect")
        self.status_label.configure(text="Disconnected", foreground="red")
        self.log_console("Disconnected")
    
    def debug_connection(self):
        """Debug function to test serial connection"""
        if not self.is_connected:
            self.log_console("Not connected - click Connect first")
            return
            
        self.log_console("=== DEBUG INFO ===")
        self.log_console(f"Connected: {self.is_connected}")
        self.log_console(f"Port: {self.serial_port.port if self.serial_port else 'None'}")
        self.log_console(f"Baudrate: {self.serial_port.baudrate if self.serial_port else 'None'}")
        self.log_console(f"Port open: {self.serial_port.is_open if self.serial_port else 'False'}")
        self.log_console(f"In waiting: {self.serial_port.in_waiting if self.serial_port else 0}")
        
        # Try to send a PING manually
        self.send_command("PING")
        time.sleep(0.5)
        
        # Check if any data was received
        if self.serial_port and self.serial_port.in_waiting > 0:
            data = self.serial_port.read(self.serial_port.in_waiting)
            self.log_console(f"Data received: {data}")
        else:
            self.log_console("No data received - check Arduino connection and firmware")
    
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
            
            # Ensure the command is sent properly
            bytes_written = self.serial_port.write(full_cmd.encode())
            self.serial_port.flush()
            self.log_console(f"Sent ({bytes_written} bytes): {full_cmd.strip()}")
            
        except serial.SerialException as e:
            self.log_console(f"Send error - connection lost: {str(e)}")
            self.root.after(0, self.disconnect)
            
        except Exception as e:
            self.log_console(f"Send error: {str(e)}")
    
    def send_manual_command(self):
        cmd = self.cmd_entry.get()
        if cmd:
            self.send_command(cmd)
            self.cmd_entry.delete(0, tk.END)
    
    def read_serial(self):
        buffer = ""
        last_read_time = time.time()
        
        while self.running:
            try:
                if self.serial_port and self.serial_port.in_waiting > 0:
                    # Read available data
                    data = self.serial_port.read(self.serial_port.in_waiting)
                    buffer += data.decode('utf-8', errors='ignore')
                    last_read_time = time.time()
                    
                    # Process complete lines
                    lines = buffer.split('\n')
                    buffer = lines[-1]  # Keep incomplete line
                    
                    for line in lines[:-1]:
                        line = line.strip()
                        if line:
                            self.process_serial_line(line)
                            self.last_data_time = time.time()
                
                # Check for timeout - only if we haven't received data for a while
                elif self.is_connected and time.time() - last_read_time > 5.0:
                    # Send a ping to check connection
                    if time.time() - self.last_ping_time > self.ping_interval:
                        self.last_ping_time = time.time()
                        self.send_command("PING")
                        self.ping_retries += 1
                        
                        if self.ping_retries > self.max_ping_retries:
                            self.log_console("Max ping retries exceeded - connection lost")
                            self.running = False
                            self.root.after(0, self.disconnect)
                            break
                    
                    # Check for overall connection timeout
                    if time.time() - self.last_data_time > self.connection_timeout:
                        self.log_console("Connection timeout - no data received")
                        self.running = False
                        self.root.after(0, self.disconnect)
                        break
                
                time.sleep(0.05)  # Small delay to prevent CPU hogging
                
            except serial.SerialException as e:
                if self.running:
                    self.log_console(f"Serial error: {str(e)}")
                    self.running = False
                    self.root.after(0, self.disconnect)
                break
                
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
            
            # Reset ping retries on any response
            self.ping_retries = 0
        
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
            self.log_console("Robot ready - communication established")
            self.ping_retries = 0
    
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
        # Limit console size to prevent memory issues
        if int(self.console_text.index('end-1c').split('.')[0]) > 1000:
            self.console_text.delete('1.0', '2.0')
    
    def monitor_connection(self):
        """Periodically check connection health"""
        if self.is_connected and self.serial_port:
            # Check if port is still open
            if not self.serial_port.is_open:
                self.log_console("Serial port closed unexpectedly")
                self.root.after(0, self.disconnect)
        
        # Schedule next check
        self.root.after(3000, self.monitor_connection)
    
    def on_closing(self):
        self.running = False
        self.disconnect()
        self.root.destroy()

if __name__ == "__main__":
    # Set up for better cross-platform compatibility
    try:
        root = tk.Tk()
        app = RobotGUI(root)
        root.protocol("WM_DELETE_WINDOW", app.on_closing)
        root.mainloop()
    except Exception as e:
        print(f"Error: {e}")
        if sys.platform.startswith('linux'):
            print("On Linux/Jetson, you may need to:")
            print("  sudo chmod 666 /dev/ttyUSB*")
            print("  or add your user to the dialout group: sudo usermod -a -G dialout $USER")
            print("  Also check if Arduino is connected and running the firmware")