"""
Layer 2: Linearization - Linearize nonlinear model around current state
"""
import numpy as np
from config.robot_params import RobotParams

class Linearizer:
    """
    Linearize the differential drive model around operating point
    """
    
    def __init__(self, dt=RobotParams.MPC_DT):
        self.dt = dt
        
    def linearize(self, state, control):
        """
        Compute Jacobians A and B for state space model
        A = I + dt * df/dx |_{x,u}
        B = dt * df/du |_{x,u}
        
        where f(x,u) = [v*cos(theta), v*sin(theta), omega]
        
        Args:
            state: [x, y, theta]
            control: [v, omega]
        Returns:
            A: 3x3 state transition matrix
            B: 3x2 input matrix
        """
        x, y, theta = state
        v, omega = control
        
        # Jacobian of f with respect to state
        df_dx = np.array([
            [0, 0, -v * np.sin(theta)],
            [0, 0,  v * np.cos(theta)],
            [0, 0,  0]
        ])
        
        # Jacobian of f with respect to control
        df_du = np.array([
            [np.cos(theta), 0],
            [np.sin(theta), 0],
            [0,             1]
        ])
        
        # Discrete-time linear model
        A = np.eye(3) + self.dt * df_dx
        B = self.dt * df_du
        
        return A, B
    
    def linearize_around_trajectory(self, state_trajectory, control_trajectory):
        """
        Linearize around a nominal trajectory
        
        Args:
            state_trajectory: N x 3 array of states
            control_trajectory: N x 2 array of controls
        Returns:
            A_matrices: list of N 3x3 matrices
            B_matrices: list of N 3x2 matrices
        """
        A_matrices = []
        B_matrices = []
        
        for state, control in zip(state_trajectory, control_trajectory):
            A, B = self.linearize(state, control)
            A_matrices.append(A)
            B_matrices.append(B)
            
        return A_matrices, B_matrices