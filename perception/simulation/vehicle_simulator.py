"""
Vehicle behavior simulation for realistic traffic scenarios.
"""

import numpy as np
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class VehicleSimulator:
    """Simulates vehicle movement with basic traffic rules"""
    
    def __init__(self, config: dict = None):
        self.vehicles = []
        
        # Vehicle parameters
        self.params = {
            'max_speed': 13.9,      # 50 km/h in m/s
            'max_acceleration': 3.0, # m/s^2
            'max_deceleration': 5.0, # m/s^2
            'vehicle_length': 4.5,   # meters
            'vehicle_width': 1.8,    # meters
            'safe_distance': 2.0,    # seconds gap
            'lane_width': 3.5        # meters
        }
        
        if config:
            self.params.update(config)
    
    def add_vehicle(self, position: Tuple[float, float],
                   velocity: Tuple[float, float] = (0, 0),
                   path: List[Tuple[float, float]] = None) -> int:
        """Add a vehicle to simulation
        
        Args:
            position: Starting position (x, y)
            velocity: Initial velocity (vx, vy)
            path: Optional waypoint path
            
        Returns:
            Vehicle ID
        """
        vehicle = {
            'id': len(self.vehicles),
            'position': np.array(position, dtype=float),
            'velocity': np.array(velocity, dtype=float),
            'acceleration': np.zeros(2),
            'speed': np.linalg.norm(velocity),
            'heading': np.arctan2(velocity[1], velocity[0]) if np.linalg.norm(velocity) > 0 else 0,
            'path': path or [],
            'path_index': 0,
            'color': np.random.randint(0, 255, 3).tolist(),
            'stopped': False
        }
        
        self.vehicles.append(vehicle)
        return vehicle['id']
    
    def update(self, dt: float):
        """Update all vehicles
        
        Args:
            dt: Time step in seconds
        """
        for vehicle in self.vehicles:
            if vehicle['path']:
                self._follow_path(vehicle, dt)
            else:
                self._constant_velocity(vehicle, dt)
            
            # Apply basic physics
            vehicle['position'] += vehicle['velocity'] * dt
            
            # Update heading
            speed = np.linalg.norm(vehicle['velocity'])
            if speed > 0.1:
                vehicle['heading'] = np.arctan2(vehicle['velocity'][1], 
                                                vehicle['velocity'][0])
                vehicle['speed'] = speed
    
    def _follow_path(self, vehicle: dict, dt: float):
        """Make vehicle follow waypoint path"""
        if vehicle['path_index'] >= len(vehicle['path']):
            # Loop back to start
            vehicle['path_index'] = 0
        
        # Get current target
        target = np.array(vehicle['path'][vehicle['path_index']])
        current = vehicle['position']
        
        # Calculate steering
        to_target = target - current
        distance = np.linalg.norm(to_target)
        
        if distance < 1.0:
            # Reached waypoint
            vehicle['path_index'] += 1
            if vehicle['path_index'] < len(vehicle['path']):
                target = np.array(vehicle['path'][vehicle['path_index']])
                to_target = target - current
                distance = np.linalg.norm(to_target)
        
        if distance > 0:
            # PID-like steering
            desired_velocity = (to_target / distance) * self.params['max_speed'] * 0.5
            
            # Acceleration towards desired velocity
            velocity_error = desired_velocity - vehicle['velocity']
            acceleration = np.clip(velocity_error / dt, 
                                 -self.params['max_deceleration'],
                                 self.params['max_acceleration'])
            
            vehicle['velocity'] += acceleration * dt
            
            # Limit speed
            speed = np.linalg.norm(vehicle['velocity'])
            if speed > self.params['max_speed']:
                vehicle['velocity'] = vehicle['velocity'] / speed * self.params['max_speed']
    
    def _constant_velocity(self, vehicle: dict, dt: float):
        """Maintain constant velocity with random variations"""
        # Add small random perturbations
        perturbation = np.random.normal(0, 0.1, 2)
        vehicle['velocity'] += perturbation * dt
        
        # Limit speed
        speed = np.linalg.norm(vehicle['velocity'])
        if speed > self.params['max_speed']:
            vehicle['velocity'] = vehicle['velocity'] / speed * self.params['max_speed']
        elif speed < 0.1 and not vehicle['stopped']:
            vehicle['velocity'] = np.zeros(2)
    
    def get_vehicle_positions(self) -> List[Tuple[float, float]]:
        """Get current positions of all vehicles"""
        return [tuple(v['position']) for v in self.vehicles]
    
    def get_vehicle_states(self) -> List[dict]:
        """Get complete state of all vehicles"""
        return [{
            'id': v['id'],
            'position': tuple(v['position']),
            'velocity': tuple(v['velocity']),
            'speed': v['speed'],
            'heading': v['heading']
        } for v in self.vehicles]