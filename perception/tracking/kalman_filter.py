"""
Kalman filter implementation for object tracking.
Supports constant velocity and constant acceleration models.
"""

import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class KalmanFilterTracker:
    """Kalman filter for tracking object state"""
    
    def __init__(self, dt: float = 0.033, 
                 process_noise: float = 0.01,
                 measurement_noise: float = 0.1):
        """
        Args:
            dt: Time step
            process_noise: Process noise covariance
            measurement_noise: Measurement noise covariance
        """
        self.dt = dt
        
        # State vector: [x, y, vx, vy, ax, ay]
        self.state_dim = 6
        self.measurement_dim = 2
        
        # State transition matrix (constant acceleration model)
        self.F = np.array([
            [1, 0, dt, 0, 0.5*dt**2, 0],
            [0, 1, 0, dt, 0, 0.5*dt**2],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        
        # Measurement matrix (we only measure position)
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ])
        
        # Process noise covariance
        q = process_noise
        self.Q = q * np.array([
            [dt**4/4, 0, dt**3/2, 0, dt**2/2, 0],
            [0, dt**4/4, 0, dt**3/2, 0, dt**2/2],
            [dt**3/2, 0, dt**2, 0, dt, 0],
            [0, dt**3/2, 0, dt**2, 0, dt],
            [dt**2/2, 0, dt, 0, 1, 0],
            [0, dt**2/2, 0, dt, 0, 1]
        ])
        
        # Measurement noise covariance
        r = measurement_noise
        self.R = r * np.eye(self.measurement_dim)
        
        # State and covariance
        self.x = np.zeros((self.state_dim, 1))
        self.P = np.eye(self.state_dim) * 100
        
        # Track metadata
        self.age = 0
        self.hits = 0
        self.time_since_update = 0
        self.is_initialized = False
        
    def initialize(self, position: Tuple[float, float], 
                  velocity: Optional[Tuple[float, float]] = None):
        """Initialize filter with initial state
        Args:
            position: Initial position (x, y)
            velocity: Initial velocity (vx, vy) or None
        """
        self.x[0] = position[0]
        self.x[1] = position[1]
        
        if velocity is not None:
            self.x[2] = velocity[0]
            self.x[3] = velocity[1]
        
        self.P = np.eye(self.state_dim) * 10
        self.is_initialized = True
        self.age = 0
        self.hits = 1
    
    def predict(self) -> np.ndarray:
        """Predict next state
        Returns:
            Predicted state vector
        """
        if not self.is_initialized:
            return None
        
        # State prediction
        self.x = self.F @ self.x
        
        # Covariance prediction
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        self.age += 1
        self.time_since_update += 1
        
        return self.x
    
    def update(self, measurement: np.ndarray) -> np.ndarray:
        """Update filter with new measurement
        Args:
            measurement: Measurement vector [x, y]
        Returns:
            Updated state vector
        """
        if not self.is_initialized:
            self.initialize(measurement[:2])
            return self.x
        
        # Kalman gain
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state
        y = measurement.reshape(self.measurement_dim, 1) - self.H @ self.x
        self.x = self.x + K @ y
        
        # Update covariance
        I = np.eye(self.state_dim)
        self.P = (I - K @ self.H) @ self.P
        
        self.hits += 1
        self.time_since_update = 0
        
        return self.x
    
    def get_state(self) -> dict:
        """Get current state estimate
        Returns:
            Dictionary with position, velocity, acceleration
        """
        return {
            'position': (float(self.x[0]), float(self.x[1])),
            'velocity': (float(self.x[2]), float(self.x[3])),
            'acceleration': (float(self.x[4]), float(self.x[5])),
            'covariance': self.P[:2, :2].copy()
        }
    
    def predict_future(self, dt: float) -> Tuple[float, float]:
        """Predict future position
        Args:
            dt: Time horizon
        Returns:
            Predicted position (x, y)
        """
        pos = self.x[0] + self.x[2] * dt + 0.5 * self.x[4] * dt**2
        pos_y = self.x[1] + self.x[3] * dt + 0.5 * self.x[5] * dt**2
        return (float(pos), float(pos_y))
    
    def is_stale(self, max_age: int = 30) -> bool:
        """Check if filter is stale
        Args:
            max_age: Maximum allowed age
        Returns:
            bool: True if filter is stale
        """
        return self.time_since_update > max_age
    
    def is_confident(self, min_hits: int = 3) -> bool:
        """Check if filter has enough measurements
        Args:
            min_hits: Minimum number of hits
        Returns:
            bool: True if filter is confident
        """
        return self.hits >= min_hits