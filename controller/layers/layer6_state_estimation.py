"""
Layer 6: State Estimation - Extended Kalman Filter for pose estimation
"""
import numpy as np
from config.robot_params import RobotParams

class ExtendedKalmanFilter:
    """
    EKF for fusing wheel odometry and IMU data
    
    State: [x, y, theta]
    Control input: [v, omega] from wheel encoders
    Measurement: [theta_imu, x_odom, y_odom]
    """
    
    def __init__(self, dt=RobotParams.MPC_DT):
        self.dt = dt
        
        # State vector [x, y, theta]
        self.state = np.zeros(3)
        
        # State covariance
        self.P = np.eye(3) * 0.1
        
        # Process noise
        self.Q = np.diag([0.01, 0.01, 0.001])
        
        # Measurement noise - store as separate components for clarity
        self.R_imu = 0.01  # IMU heading measurement noise (scalar)
        self.R_pos = np.diag([0.1, 0.1])  # Position measurement noise (2x2)
        
    def predict(self, v, omega):
        """
        Prediction step using wheel odometry
        
        Args:
            v: linear velocity from wheel encoders
            omega: angular velocity from wheel encoders
        """
        x, y, theta = self.state
        
        # Nonlinear state transition
        x_new = x + v * np.cos(theta) * self.dt
        y_new = y + v * np.sin(theta) * self.dt
        theta_new = theta + omega * self.dt
        
        # Normalize theta
        theta_new = self._normalize_angle(theta_new)
        
        self.state = np.array([x_new, y_new, theta_new])
        
        # Linearize and update covariance
        self.F = np.array([
            [1, 0, -v * np.sin(theta) * self.dt],
            [0, 1,  v * np.cos(theta) * self.dt],
            [0, 0,  1]
        ])
        
        self.P = self.F @ self.P @ self.F.T + self.Q
        
    def update_imu(self, theta_imu):
        """
        Update step using IMU heading measurement
        
        Args:
            theta_imu: heading from IMU (rad)
        """
        # Measurement Jacobian for IMU (1x3)
        H_imu = np.array([[0, 0, 1]])
        
        # Innovation (scalar)
        y = theta_imu - self.state[2]
        y = self._normalize_angle(y)
        
        # Innovation covariance (scalar)
        S = H_imu @ self.P @ H_imu.T + self.R_imu
        S = float(S)  # Ensure it's a scalar
        
        # Kalman gain (3x1)
        K = (self.P @ H_imu.T) / S
        K = K.reshape(3, 1)  # Make it 3x1
        
        # Update state
        self.state = self.state.reshape(3, 1) + K * y
        self.state = self.state.flatten()
        self.state[2] = self._normalize_angle(self.state[2])
        
        # Update covariance using Joseph form
        # I - K*H should be 3x3
        I_KH = np.eye(3) - K @ H_imu  # (3x1) @ (1x3) = (3x3)
        self.P = I_KH @ self.P @ I_KH.T + self.R_imu * (K @ K.T)
        
    def update_odometry(self, x_odom, y_odom):
        """
        Update step using position measurements
        
        Args:
            x_odom, y_odom: position from odometry/wheel encoders
        """
        # Measurement (2,)
        z = np.array([x_odom, y_odom])
        z_pred = self.state[:2]
        
        # Measurement Jacobian for position (2x3)
        H_pos = np.array([[1, 0, 0],
                          [0, 1, 0]])
        
        # Innovation (2,)
        y = z - z_pred
        
        # Innovation covariance (2x2)
        S = H_pos @ self.P @ H_pos.T + self.R_pos
        
        # Kalman gain (3x2)
        K = self.P @ H_pos.T @ np.linalg.inv(S)
        
        # Update state: (3,) = (3,) + (3x2) @ (2,)
        self.state = self.state + K @ y
        self.state = self.state.flatten()
        
        # Update covariance using Joseph form
        I_KH = np.eye(3) - K @ H_pos  # (3x3) - (3x2) @ (2x3) = (3x3)
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R_pos @ K.T
    
    def _normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        return np.arctan2(np.sin(angle), np.cos(angle))
    
    def get_state(self):
        """Get current state estimate"""
        return self.state.copy()
    
    def get_covariance(self):
        """Get state covariance"""
        return self.P.copy()


class StateEstimator:
    """
    High-level state estimation combining multiple sensor sources
    """
    
    def __init__(self, dt=RobotParams.MPC_DT):
        self.dt = dt
        self.ekf = ExtendedKalmanFilter(dt)
        
        # Sensor data storage
        self.latest_imu_heading = None
        self.latest_wheel_rpm = {'left': 0, 'right': 0}
        self.latest_pose = np.zeros(3)
        
        # Wheel odometry model
        from layers.layer1_plant_model import DifferentialDriveModel
        self.model = DifferentialDriveModel(dt)
        
    def update_with_feedback(self, feedback):
        """
        Update state estimate with sensor feedback
        
        Args:
            feedback: dict from communication layer
                - rpm_left, rpm_right
                - theta (IMU heading)
                - x, y (optional position estimates)
        """
        # Extract sensor data
        rpm_left = feedback.get('rpm_left', 0)
        rpm_right = feedback.get('rpm_right', 0)
        imu_theta = feedback.get('theta', None)
        x_meas = feedback.get('x', None)
        y_meas = feedback.get('y', None)
        
        # Convert wheel RPM to velocities
        v, omega = self.model.forward_kinematics_from_wheels(rpm_left, rpm_right)
        
        # EKF prediction step
        self.ekf.predict(v, omega)
        
        # EKF update with IMU
        if imu_theta is not None:
            self.ekf.update_imu(imu_theta)
        
        # EKF update with position measurements
        if x_meas is not None and y_meas is not None:
            self.ekf.update_odometry(x_meas, y_meas)
        
        # Store latest state
        self.latest_pose = self.ekf.get_state()
        
    def get_estimated_state(self):
        """Get current state estimate [x, y, theta]"""
        return self.latest_pose
    
    def reset(self, initial_pose=None):
        """Reset state estimator"""
        if initial_pose is not None:
            self.ekf.state = np.array(initial_pose)
        else:
            self.ekf.state = np.zeros(3)
        self.ekf.P = np.eye(3) * 0.1