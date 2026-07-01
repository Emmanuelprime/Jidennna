"""
Pedestrian behavior simulation using social force model.
"""

import numpy as np
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class PedestrianSimulator:
    """Simulates realistic pedestrian movement"""
    
    def __init__(self, config: dict = None):
        self.pedestrians = []
        
        # Social force model parameters
        self.params = {
            'desired_speed': 1.4,      # m/s
            'relaxation_time': 0.5,     # seconds
            'repulsion_strength': 2.0,
            'repulsion_range': 2.0,     # meters
            'body_radius': 0.3,         # meters
            'obstacle_repulsion': 5.0,
            'random_force_std': 0.1
        }
        
        if config:
            self.params.update(config)
    
    def add_pedestrian(self, start_pos: Tuple[float, float],
                       goal_pos: Tuple[float, float],
                       speed: float = None) -> int:
        """Add a pedestrian with start and goal positions
        
        Args:
            start_pos: Starting position (x, y)
            goal_pos: Goal position (x, y)
            speed: Walking speed (uses default if None)
            
        Returns:
            Pedestrian ID
        """
        if speed is None:
            speed = self.params['desired_speed'] + np.random.normal(0, 0.2)
        
        pedestrian = {
            'id': len(self.pedestrians),
            'position': np.array(start_pos, dtype=float),
            'goal': np.array(goal_pos, dtype=float),
            'velocity': np.zeros(2),
            'speed': speed,
            'path': [np.array(start_pos)],
            'reached_goal': False,
            'color': np.random.randint(0, 255, 3).tolist()
        }
        
        self.pedestrians.append(pedestrian)
        return pedestrian['id']
    
    def update(self, dt: float, obstacles: List[np.ndarray] = None):
        """Update all pedestrians using social force model
        
        Args:
            dt: Time step in seconds
            obstacles: List of obstacle positions to avoid
        """
        for ped in self.pedestrians:
            if ped['reached_goal']:
                # Assign new random goal
                if np.random.random() < 0.01:  # 1% chance per step
                    ped['goal'] = ped['position'] + np.random.uniform(-10, 10, 2)
                    ped['reached_goal'] = False
                continue
            
            # Calculate forces
            desired_force = self._desired_force(ped)
            social_force = self._social_force(ped)
            obstacle_force = self._obstacle_force(ped, obstacles) if obstacles else np.zeros(2)
            random_force = np.random.normal(0, self.params['random_force_std'], 2)
            
            # Sum forces
            total_force = desired_force + social_force + obstacle_force + random_force
            
            # Update velocity (simplified Euler integration)
            ped['velocity'] = ped['velocity'] + total_force * dt
            
            # Limit speed
            speed = np.linalg.norm(ped['velocity'])
            max_speed = ped['speed'] * 1.3
            if speed > max_speed:
                ped['velocity'] = ped['velocity'] / speed * max_speed
            
            # Update position
            ped['position'] = ped['position'] + ped['velocity'] * dt
            
            # Store path
            ped['path'].append(ped['position'].copy())
            if len(ped['path']) > 1000:
                ped['path'] = ped['path'][-500:]
            
            # Check if reached goal
            distance_to_goal = np.linalg.norm(ped['goal'] - ped['position'])
            if distance_to_goal < 0.5:
                ped['reached_goal'] = True
                ped['velocity'] = np.zeros(2)
    
    def _desired_force(self, ped: dict) -> np.ndarray:
        """Calculate force towards goal"""
        to_goal = ped['goal'] - ped['position']
        distance = np.linalg.norm(to_goal)
        
        if distance < 0.1:
            return np.zeros(2)
        
        desired_direction = to_goal / distance
        desired_velocity = desired_direction * ped['speed']
        
        return (desired_velocity - ped['velocity']) / self.params['relaxation_time']
    
    def _social_force(self, ped: dict) -> np.ndarray:
        """Calculate repulsive force from other pedestrians"""
        force = np.zeros(2)
        
        for other in self.pedestrians:
            if other['id'] == ped['id']:
                continue
            
            to_other = ped['position'] - other['position']
            distance = np.linalg.norm(to_other)
            
            if distance < self.params['repulsion_range'] and distance > 0:
                direction = to_other / distance
                strength = self.params['repulsion_strength'] * \
                          np.exp(-distance / self.params['repulsion_range'])
                force += strength * direction
        
        return force
    
    def _obstacle_force(self, ped: dict, obstacles: List[np.ndarray]) -> np.ndarray:
        """Calculate repulsive force from obstacles"""
        force = np.zeros(2)
        
        for obs in obstacles:
            to_obs = ped['position'] - obs
            distance = np.linalg.norm(to_obs)
            
            if distance < 2.0 and distance > 0:
                direction = to_obs / distance
                strength = self.params['obstacle_repulsion'] / (distance ** 2)
                force += strength * direction
        
        return force
    
    def get_pedestrian_positions(self) -> List[Tuple[float, float]]:
        """Get current positions of all pedestrians"""
        return [tuple(p['position']) for p in self.pedestrians if not p['reached_goal']]
    
    def get_pedestrian_states(self) -> List[dict]:
        """Get complete state of all pedestrians"""
        return [{
            'id': p['id'],
            'position': tuple(p['position']),
            'velocity': tuple(p['velocity']),
            'speed': np.linalg.norm(p['velocity']),
            'reached_goal': p['reached_goal']
        } for p in self.pedestrians]