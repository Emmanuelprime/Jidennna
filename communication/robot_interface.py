#!/usr/bin/env python3
"""
Robot Interface Class
=====================
High-level interface for the differential drive robot.
Provides methods for velocity control, odometry, and state management.
Can be used by other modules (MPC controller, path planner, etc.)
"""

import serial
import time
import threading
import math
import json
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass, asdict
from enum import Enum

# ─── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class RobotState:
    """Robot state data"""
    timestamp: int = 0
    vL: float = 0.0          # Left wheel speed (m/s)
    vR: float = 0.0          # Right wheel speed (m/s)
    linear: float = 0.0      # Linear velocity (m/s)
    omega: float = 0.0       # Angular velocity from encoders (rad/s)
    actual_omega: float = 0.0 # Angular velocity from IMU (rad/s)
    yaw: float = 0.0         # Heading (degrees)
    x: float = 0.0           # X position (m)
    y: float = 0.0           # Y position (m)
    left_pwm: int = 0        # Left motor PWM
    right_pwm: int = 0       # Right motor PWM
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())
    
    def get_pose(self) -> Tuple[float, float, float]:
        """Return (x, y, yaw) in radians"""
        return (self.x, self.y, math.radians(self.yaw))

@dataclass
class RobotCommand:
    """Command to send to robot"""
    linear_velocity: float = 0.0   # m/s
    angular_velocity: float = 0.0  # rad/s
    duration: Optional[float] = None  # seconds (None = continuous)
    
    def to_command_string(self) -> str:
        return f"V{self.linear_velocity:.3f},{self.angular_velocity:.3f}"

class RobotMode(Enum):
    """Robot operational modes"""
    IDLE = "IDLE"
    ACCELERATING = "ACCELERATING"
    CRUISING = "CRUISING"
    DECELERATING = "DECELERATING"
    EMERGENCY_STOP = "EMERGENCY_STOP"
    CALIBRATING = "CALIBRATING"

# ─── Main Robot Interface Class ─────────────────────────────────────────────

class RobotInterface:
    """
    High-level interface for robot control and feedback.
    Thread-safe and suitable for use with MPC controllers.
    """
    
    def __init__(self, port: str, baudrate: int = 115200, auto_connect: bool = False):
        self.port = port
        self.baudrate = baudrate
        self.ser: Optional[serial.Serial] = None
        self.connected = False
        self.running = False
        
        # Latest telemetry data
        self._telemetry = RobotState()
        self._mode = RobotMode.IDLE
        
        # Lock for thread-safe access
        self._lock = threading.Lock()
        self._rx_thread: Optional[threading.Thread] = None
        
        # Callbacks for telemetry updates
        self._callbacks: List[callable] = []
        
        # Command history
        self._last_command: Optional[RobotCommand] = None
        self._command_time = 0
        
        if auto_connect:
            self.connect()
    
    # ─── Connection Management ──────────────────────────────────────────────
    
    def connect(self) -> bool:
        """Connect to the robot"""
        if self.connected:
            return True
            
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=0.5)
            time.sleep(2)
            self.ser.reset_input_buffer()
            
            # Ping and wait for READY
            self.ser.write(b'PING\n')
            time.sleep(0.5)
            
            response = ""
            start_time = time.time()
            while time.time() - start_time < 2:
                if self.ser.in_waiting > 0:
                    response = self.ser.readline().decode().strip()
                    if 'READY' in response:
                        break
                time.sleep(0.05)
            
            if 'READY' not in response:
                self.ser.close()
                return False
            
            self.connected = True
            self.running = True
            
            # Start RX thread
            self._rx_thread = threading.Thread(target=self._rx_loop, daemon=True)
            self._rx_thread.start()
            
            print(f"✅ Robot connected on {self.port}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to connect: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from robot"""
        self.running = False
        self.connected = False
        
        if self.ser and self.ser.is_open:
            try:
                self.ser.write(b's\n')
                time.sleep(0.1)
                self.ser.close()
            except:
                pass
        
        print("🔌 Robot disconnected")
    
    def is_connected(self) -> bool:
        """Check if connected to robot"""
        return self.connected and self.ser is not None
    
    # ─── RX Thread ────────────────────────────────────────────────────────────
    
    def _rx_loop(self):
        """Background thread to receive telemetry"""
        while self.running and self.ser and self.ser.is_open:
            try:
                if self.ser.in_waiting > 0:
                    line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                    
                    if line.startswith('CNT,'):
                        state = self._parse_telemetry(line)
                        if state:
                            with self._lock:
                                self._telemetry = state
                            # Notify callbacks
                            self._notify_callbacks(state)
                    
                    elif line.startswith('# STATE:'):
                        # Parse state message
                        parts = line.split(':')
                        if len(parts) > 1:
                            state_str = parts[1].strip()
                            try:
                                self._mode = RobotMode(state_str)
                            except ValueError:
                                pass
                    
                    elif line.startswith('#'):
                        # Log other messages
                        pass
                        
                else:
                    time.sleep(0.005)
            except Exception as e:
                print(f"RX error: {e}")
                break
    
    def _parse_telemetry(self, line: str) -> Optional[RobotState]:
        """Parse CNT telemetry line"""
        parts = line.split(',')
        if len(parts) >= 12:
            try:
                return RobotState(
                    timestamp=int(parts[1]),
                    vL=float(parts[2]),
                    vR=float(parts[3]),
                    linear=float(parts[4]),
                    omega=float(parts[5]),
                    actual_omega=float(parts[6]),
                    yaw=float(parts[7]),
                    x=float(parts[8]),
                    y=float(parts[9]),
                    left_pwm=int(parts[10]),
                    right_pwm=int(parts[11])
                )
            except (ValueError, IndexError):
                pass
        return None
    
    def _notify_callbacks(self, state: RobotState):
        """Notify all registered callbacks"""
        for callback in self._callbacks:
            try:
                callback(state)
            except Exception as e:
                print(f"Callback error: {e}")
    
    # ─── Command Sending ─────────────────────────────────────────────────────
    
    def _send_command(self, cmd: str) -> bool:
        """Send raw command to robot"""
        if not self.is_connected():
            return False
        
        try:
            with self._lock:
                self.ser.write(f"{cmd}\n".encode())
                self.ser.flush()
            return True
        except Exception as e:
            print(f"Send error: {e}")
            return False
    
    def set_velocity(self, v: float, w: float, duration: Optional[float] = None) -> bool:
        """
        Set robot velocity with optional duration.
        
        Args:
            v: Linear velocity (m/s) [-1.2, 1.2]
            w: Angular velocity (rad/s) [-2.0, 2.0]
            duration: Duration in seconds (None for continuous)
        """
        v = max(-1.2, min(1.2, v))
        w = max(-2.0, min(2.0, w))
        
        cmd = RobotCommand(v, w, duration)
        self._last_command = cmd
        self._command_time = time.time()
        
        success = self._send_command(cmd.to_command_string())
        
        # If duration specified, schedule stop
        if success and duration is not None and duration > 0:
            def stop_after_duration():
                time.sleep(duration)
                self.stop()
            threading.Thread(target=stop_after_duration, daemon=True).start()
        
        return success
    
    def move_forward(self, speed: float = 0.3, duration: Optional[float] = None) -> bool:
        """Move forward at given speed"""
        return self.set_velocity(speed, 0, duration)
    
    def move_backward(self, speed: float = 0.3, duration: Optional[float] = None) -> bool:
        """Move backward at given speed"""
        return self.set_velocity(-speed, 0, duration)
    
    def spin_left(self, omega: float = 0.5, duration: Optional[float] = None) -> bool:
        """Spin left at given angular velocity"""
        return self.set_velocity(0, omega, duration)
    
    def spin_right(self, omega: float = 0.5, duration: Optional[float] = None) -> bool:
        """Spin right at given angular velocity"""
        return self.set_velocity(0, -omega, duration)
    
    def stop(self, smooth: bool = True) -> bool:
        """Stop the robot"""
        if smooth:
            return self._send_command("s")
        else:
            return self._send_command("E")
    
    def emergency_stop(self) -> bool:
        """Emergency stop (immediate)"""
        return self._send_command("E")
    
    def zero_odometry(self) -> bool:
        """Zero encoders and odometry"""
        return self._send_command("z")
    
    def calibrate_imu(self) -> bool:
        """Calibrate IMU (keep robot still)"""
        return self._send_command("C")
    
    def get_state(self) -> RobotState:
        """Get current robot state"""
        with self._lock:
            return self._telemetry
    
    def get_pose(self) -> Tuple[float, float, float]:
        """Get current pose (x, y, yaw) in radians"""
        with self._lock:
            return (self._telemetry.x, self._telemetry.y, math.radians(self._telemetry.yaw))
    
    def get_position(self) -> Tuple[float, float]:
        """Get current position (x, y) in meters"""
        with self._lock:
            return (self._telemetry.x, self._telemetry.y)
    
    def get_heading(self) -> float:
        """Get current heading in degrees"""
        with self._lock:
            return self._telemetry.yaw
    
    def get_velocity(self) -> Tuple[float, float]:
        """Get current linear and angular velocity"""
        with self._lock:
            return (self._telemetry.linear, self._telemetry.omega)
    
    def get_speed(self) -> float:
        """Get current linear speed"""
        with self._lock:
            return self._telemetry.linear
    
    def get_mode(self) -> RobotMode:
        """Get current robot mode"""
        return self._mode
    
    # ─── Callbacks ──────────────────────────────────────────────────────────
    
    def register_callback(self, callback: callable):
        """
        Register a callback function for telemetry updates.
        Callback receives RobotState object.
        """
        if callback not in self._callbacks:
            self._callbacks.append(callback)
    
    def unregister_callback(self, callback: callable):
        """Unregister a callback"""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    # ─── High-Level Functions ──────────────────────────────────────────────
    
    def drive_to_pose(self, target_x: float, target_y: float, 
                      target_yaw: Optional[float] = None,
                      speed: float = 0.3, tolerance: float = 0.05) -> bool:
        """
        Drive to a target pose using simple control.
        
        Args:
            target_x: Target X position (m)
            target_y: Target Y position (m)
            target_yaw: Target yaw (radians), None to maintain current
            speed: Maximum speed (m/s)
            tolerance: Position tolerance (m)
        
        Returns:
            True if reached target, False otherwise
        """
        max_iterations = 1000
        iteration = 0
        
        while iteration < max_iterations:
            x, y = self.get_position()
            dx = target_x - x
            dy = target_y - y
            distance = math.sqrt(dx*dx + dy*dy)
            
            if distance < tolerance:
                self.stop()
                return True
            
            # Calculate angle to target
            target_angle = math.atan2(dy, dx) * 180.0 / math.pi
            current_heading = self.get_heading()
            
            # Calculate angle error
            angle_error = target_angle - current_heading
            while angle_error > 180: angle_error -= 360
            while angle_error < -180: angle_error += 360
            
            # Control
            if abs(angle_error) > 10:
                # Turn towards target
                w = 0.3 if angle_error > 0 else -0.3
                self.set_velocity(0, w)
            else:
                # Move forward with slight correction
                v = min(speed, distance * 2.0)
                w = angle_error * 0.01
                self.set_velocity(v, w)
            
            time.sleep(0.05)
            iteration += 1
        
        self.stop()
        return False
    
    def follow_path(self, waypoints: List[Tuple[float, float]], 
                    speed: float = 0.3, tolerance: float = 0.05) -> bool:
        """
        Follow a path defined by waypoints.
        
        Args:
            waypoints: List of (x, y) waypoints in meters
            speed: Maximum speed (m/s)
            tolerance: Waypoint tolerance (m)
        
        Returns:
            True if path completed, False otherwise
        """
        for i, (wx, wy) in enumerate(waypoints):
            print(f"📍 Going to waypoint {i+1}/{len(waypoints)}: ({wx:.2f}, {wy:.2f})")
            if not self.drive_to_pose(wx, wy, None, speed, tolerance):
                print(f"❌ Failed to reach waypoint {i+1}")
                return False
        
        print("✅ Path complete!")
        return True
    
    def record_path(self, duration: float = 10.0, interval: float = 0.1) -> List[Dict]:
        """
        Record robot path for analysis.
        
        Args:
            duration: Recording duration (seconds)
            interval: Recording interval (seconds)
        
        Returns:
            List of path points with x, y, yaw
        """
        path = []
        start = time.time()
        
        while time.time() - start < duration:
            state = self.get_state()
            path.append({
                'time': state.timestamp,
                'x': state.x,
                'y': state.y,
                'yaw': state.yaw,
                'linear': state.linear,
                'omega': state.omega
            })
            time.sleep(interval)
        
        return path
    
    # ─── Context Manager Support ──────────────────────────────────────────
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

# ─── Example Usage ──────────────────────────────────────────────────────────

def main():
    """Example usage of RobotInterface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Robot Interface Test")
    parser.add_argument('--port', '-p', default='/dev/ttyUSB0', help='Serial port')
    args = parser.parse_args()
    
    # Create robot interface
    robot = RobotInterface(args.port)
    
    if not robot.connect():
        print("Failed to connect")
        return
    
    try:
        # Register callback for telemetry
        def on_telemetry(state: RobotState):
            print(f"\r📍 Pos: ({state.x:.2f}, {state.y:.2f})m  "
                  f"Yaw: {state.yaw:.1f}°  "
                  f"v: {state.linear:.2f}m/s  "
                  f"ω: {state.omega:.2f}rad/s", end='')
        
        robot.register_callback(on_telemetry)
        
        print("\n" + "="*60)
        print("Robot Interface Test")
        print("="*60)
        print("\nCommands:")
        print("  f <speed>  - Move forward")
        print("  b <speed>  - Move backward")
        print("  l <omega>  - Spin left")
        print("  r <omega>  - Spin right")
        print("  v <v>,<w>  - Set velocity")
        print("  s          - Stop")
        print("  e          - Emergency stop")
        print("  p          - Print state")
        print("  z          - Zero odometry")
        print("  c          - Calibrate IMU")
        print("  g <x>,<y>  - Drive to position")
        print("  q          - Quit")
        print("="*60 + "\n")
        
        while True:
            cmd = input("> ").strip().lower()
            
            if cmd == 'q':
                break
            elif cmd == 's':
                robot.stop()
                print("\n🛑 Stopped")
            elif cmd == 'e':
                robot.emergency_stop()
                print("\n🛑 Emergency stop")
            elif cmd == 'p':
                state = robot.get_state()
                print(f"\n📊 State:")
                print(f"  Position: ({state.x:.3f}, {state.y:.3f}) m")
                print(f"  Heading: {state.yaw:.1f}°")
                print(f"  Linear: {state.linear:.3f} m/s")
                print(f"  Omega: {state.omega:.3f} rad/s")
                print(f"  PWM: {state.left_pwm}/{state.right_pwm}")
                print(f"  Robot Mode: {robot.get_mode().value}")
            elif cmd == 'z':
                robot.zero_odometry()
                print("\n📏 Odometry zeroed")
            elif cmd == 'c':
                print("\n🧭 Calibrating IMU... Keep robot still!")
                robot.calibrate_imu()
                time.sleep(3)
                print("✅ Calibration complete")
            elif cmd.startswith('f '):
                speed = float(cmd.split()[1])
                robot.move_forward(speed)
                print(f"\n🚀 Moving forward at {speed} m/s")
            elif cmd.startswith('b '):
                speed = float(cmd.split()[1])
                robot.move_backward(speed)
                print(f"\n🔙 Moving backward at {speed} m/s")
            elif cmd.startswith('l '):
                omega = float(cmd.split()[1])
                robot.spin_left(omega)
                print(f"\n🔄 Spinning left at {omega} rad/s")
            elif cmd.startswith('r '):
                omega = float(cmd.split()[1])
                robot.spin_right(omega)
                print(f"\n🔄 Spinning right at {omega} rad/s")
            elif cmd.startswith('v '):
                parts = cmd.split()[1].split(',')
                if len(parts) == 2:
                    v = float(parts[0])
                    w = float(parts[1])
                    robot.set_velocity(v, w)
                    print(f"\n🎯 Set velocity: v={v:.3f}, w={w:.3f}")
            elif cmd.startswith('g '):
                parts = cmd.split()[1].split(',')
                if len(parts) == 2:
                    x = float(parts[0])
                    y = float(parts[1])
                    print(f"\n📍 Driving to ({x:.2f}, {y:.2f})")
                    robot.drive_to_pose(x, y)
                    print("\n✅ Arrived!")
            else:
                print("❌ Unknown command")
                
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        robot.disconnect()

if __name__ == "__main__":
    main()