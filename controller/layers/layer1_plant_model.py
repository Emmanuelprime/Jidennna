"""
Layer 1: Plant Model - Differential Drive Kinematics
"""
import numpy as np
from config.robot_params import RobotParams

class DifferentialDriveModel:
    """
    Differential drive kinematic model
    
    State: [x, y, theta]
    Control: [v, omega] (linear velocity, angular velocity)
    """
    
    def __init__(self, dt=RobotParams.MPC_DT):
        self.dt = dt
        self.L = RobotParams.WHEEL_SEPARATION
        
    def forward_kinematics(self, state, control):
        """
        Nonlinear continuous-time model
        dx/dt = v*cos(theta)
        dy/dt = v*sin(theta)
        dtheta/dt = omega
        
        Args:
            state: [x, y, theta]
            control: [v, omega]
        Returns:
            state_dot: [dx, dy, dtheta]
        """
        x, y, theta = state
        v, omega = control
        
        dx = v * np.cos(theta)
        dy = v * np.sin(theta)
        dtheta = omega
        
        return np.array([dx, dy, dtheta])
    
    def discrete_dynamics(self, state, control):
        """
        Discrete-time model using Euler integration
        state_{k+1} = state_k + dt * f(state_k, control_k)
        
        Args:
            state: [x, y, theta]
            control: [v, omega]
        Returns:
            next_state: [x, y, theta]
        """
        state_dot = self.forward_kinematics(state, control)
        next_state = state + self.dt * state_dot
        return next_state
    
    def inverse_kinematics(self, v, omega):
        """
        Convert robot velocities to wheel velocities
        v_r = v + (L/2)*omega
        v_l = v - (L/2)*omega
        
        Args:
            v: linear velocity (m/s)
            omega: angular velocity (rad/s)
        Returns:
            (v_left, v_right): wheel velocities (m/s)
        """
        v_right = v + (self.L / 2) * omega
        v_left = v - (self.L / 2) * omega
        return v_left, v_right
    
    def wheel_velocities_to_rpm(self, v_left, v_right):
        """
        Convert wheel velocities to RPM
        RPM = (v / (2*pi*r)) * 60
        """
        wheel_circumference = 2 * np.pi * RobotParams.WHEEL_RADIUS
        rpm_left = (v_left / wheel_circumference) * 60
        rpm_right = (v_right / wheel_circumference) * 60
        return rpm_left, rpm_right
    
    def rpm_to_wheel_velocities(self, rpm_left, rpm_right):
        """
        Convert RPM to wheel velocities
        """
        wheel_circumference = 2 * np.pi * RobotParams.WHEEL_RADIUS
        v_left = (rpm_left / 60) * wheel_circumference
        v_right = (rpm_right / 60) * wheel_circumference
        return v_left, v_right
    
    def forward_kinematics_from_wheels(self, rpm_left, rpm_right):
        """
        Compute robot velocities from wheel RPMs
        v = (v_r + v_l) / 2
        omega = (v_r - v_l) / L
        """
        v_left, v_right = self.rpm_to_wheel_velocities(rpm_left, rpm_right)
        v = (v_right + v_left) / 2
        omega = (v_right - v_left) / self.L
        return v, omega