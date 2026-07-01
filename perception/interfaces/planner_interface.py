"""
Interface between perception layer and local planner.
Ensures planners only receive structured data, never raw sensor data.
"""

import numpy as np
from typing import Optional, List, Dict, Any
from collections import deque
import time
import threading
from .core_interfaces import PerceptionOutput, Obstacle, TrackedObject, RobotPose

class PlannerInterface:
    """Controlled interface for planners to consume perception data"""
    
    def __init__(self, max_history: int = 100):
        self.max_history = max_history
        self.latest_output: Optional[PerceptionOutput] = None
        self.output_history = deque(maxlen=max_history)
        self.lock = threading.Lock()
        
        # Callbacks
        self.new_output_callbacks = []
        
        # Statistics
        self.stats = {
            'total_updates': 0,
            'last_update_time': 0,
            'update_rate': 0.0
        }
    
    def update(self, perception_output: PerceptionOutput) -> None:
        """Update with latest perception output (thread-safe)"""
        with self.lock:
            self.latest_output = perception_output
            self.output_history.append(perception_output)
            self.stats['total_updates'] += 1
            self.stats['last_update_time'] = time.time()
            
            # Calculate update rate
            if len(self.output_history) > 1:
                time_span = (self.output_history[-1].timestamp - 
                           self.output_history[0].timestamp)
                if time_span > 0:
                    self.stats['update_rate'] = len(self.output_history) / time_span
            
            # Notify callbacks
            for callback in self.new_output_callbacks:
                try:
                    callback(perception_output)
                except Exception as e:
                    print(f"Callback error: {e}")
    
    def get_latest(self) -> Optional[PerceptionOutput]:
        """Get latest perception output"""
        with self.lock:
            return self.latest_output
    
    def get_obstacles(self, max_age: float = 0.5) -> List[Obstacle]:
        """Get current obstacles (filtered by age)"""
        with self.lock:
            if self.latest_output is None:
                return []
            
            current_time = time.time()
            return [
                obs for obs in self.latest_output.obstacle_list
                if (current_time - obs.timestamp) < max_age
            ]
    
    def get_occupancy_grid(self) -> Optional[np.ndarray]:
        """Get current occupancy grid"""
        with self.lock:
            if self.latest_output:
                return self.latest_output.occupancy_grid.copy()
            return None
    
    def get_robot_pose(self) -> Optional[RobotPose]:
        """Get current robot pose"""
        with self.lock:
            if self.latest_output:
                return self.latest_output.robot_pose
            return None
    
    def get_tracked_objects(self, min_confidence: float = 0.5) -> List[TrackedObject]:
        """Get tracked dynamic objects above confidence threshold"""
        with self.lock:
            if self.latest_output is None:
                return []
            return [
                obj for obj in self.latest_output.tracked_objects
                if obj.confidence >= min_confidence
            ]
    
    def get_nearest_obstacle(self) -> Optional[Obstacle]:
        """Get nearest obstacle to robot"""
        with self.lock:
            if self.latest_output is None or not self.latest_output.obstacle_list:
                return None
            
            return min(
                self.latest_output.obstacle_list,
                key=lambda obs: np.sqrt(obs.position[0]**2 + obs.position[1]**2)
            )
    
    def predict_future_state(self, time_horizon: float) -> Dict[str, Any]:
        """Predict future state for planning"""
        with self.lock:
            if self.latest_output is None:
                return {}
            
            predictions = {
                'timestamp': time.time() + time_horizon,
                'predicted_obstacles': [],
                'predicted_tracks': []
            }
            
            # Simple linear prediction
            for obj in self.latest_output.tracked_objects:
                future_pos = (
                    obj.position[0] + obj.velocity[0] * time_horizon,
                    obj.position[1] + obj.velocity[1] * time_horizon
                )
                predictions['predicted_tracks'].append({
                    'id': obj.track_id,
                    'class': obj.class_name,
                    'future_position': future_pos,
                    'confidence': obj.confidence
                })
            
            return predictions
    
    def register_callback(self, callback) -> None:
        """Register callback for new perception outputs"""
        self.new_output_callbacks.append(callback)
    
    def is_fresh(self, max_age: float = 0.2) -> bool:
        """Check if perception data is fresh"""
        with self.lock:
            if self.latest_output is None:
                return False
            return (time.time() - self.latest_output.timestamp) < max_age