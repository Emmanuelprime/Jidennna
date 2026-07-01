"""
Dynamic obstacle mapping and management.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import time
import logging
from ..interfaces.core_interfaces import Obstacle, TrackedObject

logger = logging.getLogger(__name__)

class ObstacleMap:
    """Manages dynamic and static obstacles"""
    
    def __init__(self, config=None):
        # Obstacle storage
        self.dynamic_obstacles: Dict[int, Obstacle] = {}  # By track ID
        self.static_obstacles: Dict[int, Obstacle] = {}    # By obstacle ID
        self.obstacle_counter = 0
        
        # Configuration
        self.max_obstacle_age = 1.0  # seconds
        self.static_velocity_threshold = 0.1  # m/s
        self.merge_distance = 0.5  # meters
        
        # History for static obstacle detection
        self.position_history: Dict[int, List[Tuple[float, float]]] = {}
        self.max_history = 50
    
    def update_dynamic_obstacles(self, tracked_objects: List[TrackedObject],
                                timestamp: float):
        """Update dynamic obstacles from tracked objects
        
        Args:
            tracked_objects: List of tracked objects
            timestamp: Current timestamp
        """
        # Update existing obstacles
        for obj in tracked_objects:
            obstacle = Obstacle(
                id=obj.track_id,
                position=obj.position,
                radius=self._get_obstacle_radius(obj.class_name),
                obstacle_type=obj.class_name,
                velocity=obj.velocity,
                confidence=obj.confidence,
                timestamp=timestamp
            )
            
            self.dynamic_obstacles[obj.track_id] = obstacle
            
            # Update history
            if obj.track_id not in self.position_history:
                self.position_history[obj.track_id] = []
            
            self.position_history[obj.track_id].append(obj.position)
            
            if len(self.position_history[obj.track_id]) > self.max_history:
                self.position_history[obj.track_id] = \
                    self.position_history[obj.track_id][-self.max_history//2:]
        
        # Check for static obstacles (previously dynamic, now stopped)
        self._detect_static_obstacles(timestamp)
        
        # Remove old obstacles
        self._cleanup_old_obstacles(timestamp)
    
    def add_static_obstacle(self, position: Tuple[float, float],
                           radius: float = 0.5,
                           obstacle_type: str = 'static'):
        """Add a static obstacle
        
        Args:
            position: (x, y) position in robot frame
            radius: Obstacle radius in meters
            obstacle_type: Type of obstacle
            
        Returns:
            Obstacle ID
        """
        obstacle_id = self.obstacle_counter
        self.obstacle_counter += 1
        
        obstacle = Obstacle(
            id=obstacle_id,
            position=position,
            radius=radius,
            obstacle_type=obstacle_type,
            confidence=1.0,
            timestamp=time.time()
        )
        
        self.static_obstacles[obstacle_id] = obstacle
        return obstacle_id
    
    def _detect_static_obstacles(self, timestamp: float):
        """Detect objects that have become static"""
        for track_id, history in list(self.position_history.items()):
            if len(history) < 10:
                continue
            
            # Check if object has stopped moving
            recent_positions = np.array(history[-10:])
            total_movement = np.sum(np.linalg.norm(np.diff(recent_positions, axis=0), axis=1))
            
            if total_movement < self.static_velocity_threshold * 10:
                # Object is static, move to static obstacles
                if track_id in self.dynamic_obstacles:
                    obstacle = self.dynamic_obstacles[track_id]
                    obstacle.obstacle_type = f"static_{obstacle.obstacle_type}"
                    obstacle.velocity = (0.0, 0.0)
                    
                    self.static_obstacles[track_id] = obstacle
                    del self.dynamic_obstacles[track_id]
    
    def _cleanup_old_obstacles(self, timestamp: float):
        """Remove obstacles that haven't been updated"""
        for track_id in list(self.dynamic_obstacles.keys()):
            obstacle = self.dynamic_obstacles[track_id]
            if timestamp - obstacle.timestamp > self.max_obstacle_age:
                del self.dynamic_obstacles[track_id]
                if track_id in self.position_history:
                    del self.position_history[track_id]
    
    def _get_obstacle_radius(self, class_name: str) -> float:
        """Get obstacle radius based on class"""
        radii = {
            'person': 0.3,
            'bicycle': 0.5,
            'car': 1.5,
            'truck': 2.0,
            'bus': 2.5,
            'motorcycle': 0.4,
            'dog': 0.2,
            'cat': 0.15,
            'static': 0.5
        }
        return radii.get(class_name, 0.5)
    
    def get_all_obstacles(self) -> List[Obstacle]:
        """Get all obstacles (static + dynamic)"""
        return (list(self.dynamic_obstacles.values()) + 
                list(self.static_obstacles.values()))
    
    def get_obstacles_in_radius(self, center: Tuple[float, float],
                               radius: float) -> List[Obstacle]:
        """Get obstacles within radius of a point"""
        nearby = []
        
        for obs in self.get_all_obstacles():
            distance = np.sqrt(
                (obs.position[0] - center[0])**2 +
                (obs.position[1] - center[1])**2
            )
            if distance <= radius:
                nearby.append(obs)
        
        return nearby
    
    def get_nearest_obstacle(self, point: Tuple[float, float]) -> Optional[Obstacle]:
        """Get the nearest obstacle to a point"""
        all_obstacles = self.get_all_obstacles()
        
        if not all_obstacles:
            return None
        
        return min(all_obstacles, key=lambda obs: 
                  np.sqrt((obs.position[0] - point[0])**2 + 
                         (obs.position[1] - point[1])**2))
    
    def clear(self):
        """Clear all obstacles"""
        self.dynamic_obstacles.clear()
        self.static_obstacles.clear()
        self.position_history.clear()
    
    def get_obstacle_count(self) -> Tuple[int, int]:
        """Get count of dynamic and static obstacles"""
        return len(self.dynamic_obstacles), len(self.static_obstacles)