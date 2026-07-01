"""
Layer 7: Simulation Environment
"""
import numpy as np
import time
from config.robot_params import RobotParams
from layers.layer1_plant_model import DifferentialDriveModel

class SimulationEnvironment:
    """
    Simulation environment that mimics real hardware behavior
    
    This MUST use the EXACT same kinematic equations as the real robot
    and the same MPC controller without modification.
    """
    
    def __init__(self, enable_noise=True, enable_delay=True):
        self.model = DifferentialDriveModel()
        
        # Robot state [x, y, theta]
        self.true_state = np.zeros(3)
        
        # Noise parameters
        self.enable_noise = enable_noise
        self.encoder_noise_std = 0.5  # RPM noise
        self.imu_drift_rate = 0.001   # rad/s drift
        self.imu_noise_std = 0.01     # rad noise
        
        # Delay parameters
        self.enable_delay = enable_delay
        self.comm_delay = 0.02  # 20ms communication delay
        
        # IMU drift state
        self.imu_drift = 0.0
        
        # Command buffer (for simulating delays)
        self.command_buffer = []
        
        # Latest motor RPM
        self.current_rpm = {'left': 0, 'right': 0}

        # Simulated obstacle points for distance sensors
        self.obstacles = []
        self.sensor_angles = {
            'left': np.deg2rad(35),
            'center': 0.0,
            'right': np.deg2rad(-35)
        }
        self.sensor_fov = np.deg2rad(25)
        
        # Simulation time
        self.sim_time = 0.0
        
        print(f"Simulation initialized (Noise: {enable_noise}, Delay: {enable_delay})")
    
    def reset(self, initial_state=None):
        """Reset simulation to initial state"""
        if initial_state is not None:
            self.true_state = np.array(initial_state)
        else:
            self.true_state = np.zeros(3)
        
        self.imu_drift = 0.0
        self.command_buffer = []
        self.current_rpm = {'left': 0, 'right': 0}
        self.sim_time = 0.0
    
    def apply_control(self, rpm_left, rpm_right):
        """
        Apply control commands with delay simulation
        
        Args:
            rpm_left, rpm_right: wheel RPM commands
        """
        # Store command with timestamp for delay simulation
        if self.enable_delay:
            self.command_buffer.append({
                'rpm_left': rpm_left,
                'rpm_right': rpm_right,
                'timestamp': self.sim_time + self.comm_delay
            })
        else:
            # Apply immediately
            self._execute_command(rpm_left, rpm_right)
    
    def step(self, dt=RobotParams.MPC_DT):
        """
        Step simulation forward by dt seconds
        
        Returns:
            feedback: simulated sensor feedback
        """
        self.sim_time += dt
        
        # Process delayed commands
        if self.enable_delay:
            executed_commands = []
            for cmd in self.command_buffer:
                if self.sim_time >= cmd['timestamp']:
                    self._execute_command(cmd['rpm_left'], cmd['rpm_right'])
                    executed_commands.append(cmd)
            
            # Remove executed commands
            for cmd in executed_commands:
                self.command_buffer.remove(cmd)
        
        # Convert current RPM to velocities
        v, omega = self.model.forward_kinematics_from_wheels(
            self.current_rpm['left'], 
            self.current_rpm['right']
        )
        
        # Update true state using EXACT same kinematics as real robot
        self.true_state = self.model.discrete_dynamics(
            self.true_state, 
            np.array([v, omega])
        )
        
        # Generate simulated sensor feedback
        feedback = self._generate_feedback()
        
        return feedback
    
    def _execute_command(self, rpm_left, rpm_right):
        """Execute motor command (simulate motor dynamics)"""
        # Simple first-order motor dynamics with rate limiting
        tau = 0.05  # Motor time constant (faster response)
        alpha = RobotParams.MPC_DT / tau
        alpha = min(alpha, 1.0)  # Limit to prevent overshoot
        
        # Maximum RPM change per step (acceleration limit)
        max_rpm_change = 100  # RPM per step
        
        # Apply rate-limited change
        delta_left = alpha * (rpm_left - self.current_rpm['left'])
        delta_right = alpha * (rpm_right - self.current_rpm['right'])
        
        # Limit rate of change
        delta_left = np.clip(delta_left, -max_rpm_change, max_rpm_change)
        delta_right = np.clip(delta_right, -max_rpm_change, max_rpm_change)
        
        self.current_rpm['left'] += delta_left
        self.current_rpm['right'] += delta_right
    
    def _generate_feedback(self):
        """Generate simulated sensor feedback with noise"""
        
        # True values
        rpm_left_true = self.current_rpm['left']
        rpm_right_true = self.current_rpm['right']
        theta_true = self.true_state[2]
        x_true = self.true_state[0]
        y_true = self.true_state[1]
        
        if self.enable_noise:
            # Add encoder noise
            rpm_left_meas = rpm_left_true + np.random.normal(0, self.encoder_noise_std)
            rpm_right_meas = rpm_right_true + np.random.normal(0, self.encoder_noise_std)
            
            # Update IMU drift
            self.imu_drift += self.imu_drift_rate * RobotParams.MPC_DT
            
            # Add IMU noise and drift
            theta_meas = theta_true + self.imu_drift + np.random.normal(0, self.imu_noise_std)
            
            # Position with odometry noise
            x_meas = x_true + np.random.normal(0, 0.01)  # 1cm std
            y_meas = y_true + np.random.normal(0, 0.01)
        else:
            rpm_left_meas = rpm_left_true
            rpm_right_meas = rpm_right_true
            theta_meas = theta_true
            x_meas = x_true
            y_meas = y_true
        
        # Build feedback packet (same format as hardware)
        feedback = {
            'timestamp': self.sim_time,
            'rpm_left': rpm_left_meas,
            'rpm_right': rpm_right_meas,
            'theta': self._normalize_angle(theta_meas),
            'x': x_meas,
            'y': y_meas,
            **self._generate_distance_sensor_feedback()
        }
        
        return feedback
    
    def _normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        return np.arctan2(np.sin(angle), np.cos(angle))
    
    def get_true_state(self):
        """Get true robot state (for evaluation)"""
        return self.true_state.copy()

    def set_obstacles(self, obstacles):
        """
        Set simulated obstacle points as [(x, y), ...].
        """
        self.obstacles = [np.asarray(obstacle, dtype=float) for obstacle in obstacles]
    
    def set_comm_delay(self, delay):
        """Set communication delay"""
        self.comm_delay = delay
    
    def set_noise_levels(self, encoder_std=None, imu_drift=None, imu_std=None):
        """Adjust noise levels"""
        if encoder_std is not None:
            self.encoder_noise_std = encoder_std
        if imu_drift is not None:
            self.imu_drift_rate = imu_drift
        if imu_std is not None:
            self.imu_noise_std = imu_std

    def _generate_distance_sensor_feedback(self):
        distances = {}
        for name, angle_offset in self.sensor_angles.items():
            distances[f'distance_{name}'] = self._distance_in_sensor_cone(angle_offset)

        return distances

    def _distance_in_sensor_cone(self, angle_offset):
        max_range = RobotParams.OBSTACLE_SENSOR_MAX_RANGE
        if not self.obstacles:
            return max_range

        sensor_angle = self._normalize_angle(self.true_state[2] + angle_offset)
        sensor_direction = np.array([np.cos(sensor_angle), np.sin(sensor_angle)])
        robot_pos = self.true_state[:2]
        nearest = max_range

        for obstacle in self.obstacles:
            offset = obstacle - robot_pos
            distance = np.linalg.norm(offset)
            if distance <= 1e-6 or distance > max_range:
                continue

            bearing_direction = offset / distance
            angle_error = np.arccos(np.clip(np.dot(sensor_direction, bearing_direction), -1.0, 1.0))
            if angle_error <= self.sensor_fov:
                nearest = min(nearest, distance)

        if self.enable_noise and nearest < max_range:
            nearest += np.random.normal(0, 0.02)

        return float(np.clip(nearest, 0.0, max_range))


class HardwareInterface:
    """Real hardware interface (placeholder for actual implementation)"""
    
    def __init__(self, port='/dev/ttyUSB0', baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.is_connected = False
        
    def connect(self):
        # Implement actual serial connection
        print(f"Hardware interface connecting to {self.port}...")
        self.is_connected = True
        return True
    
    def disconnect(self):
        self.is_connected = False
        print("Hardware interface disconnected")
    
    def send_control(self, rpm_left, rpm_right):
        # Implement actual control sending
        pass
    
    def get_feedback(self):
        # Implement actual feedback reception
        return {
            'timestamp': time.time(),
            'rpm_left': 0,
            'rpm_right': 0,
            'theta': 0,
            'x': 0,
            'y': 0,
            'distance_left': RobotParams.OBSTACLE_SENSOR_MAX_RANGE,
            'distance_center': RobotParams.OBSTACLE_SENSOR_MAX_RANGE,
            'distance_right': RobotParams.OBSTACLE_SENSOR_MAX_RANGE
        }
