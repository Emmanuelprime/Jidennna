"""
Perception output builder and utilities.
"""

import time
import numpy as np
from typing import List, Optional, Dict, Any
from .core_interfaces import (
    PerceptionOutput, Obstacle, TrackedObject, 
    RobotPose, FrameID
)

class PerceptionOutputBuilder:
    """Builder pattern for constructing PerceptionOutput"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset builder state"""
        self._timestamp = time.time()
        self._robot_pose = None
        self._obstacle_list = []
        self._tracked_objects = []
        self._occupancy_grid = np.zeros((1, 1))
        self._world_model = {}
        self._frame_id = FrameID.ROBOT.value
        self._processing_time_ms = 0.0
        self._detection_threshold = 0.5
    
    def with_timestamp(self, timestamp: float) -> 'PerceptionOutputBuilder':
        self._timestamp = timestamp
        return self
    
    def with_robot_pose(self, pose: RobotPose) -> 'PerceptionOutputBuilder':
        self._robot_pose = pose
        return self
    
    def with_obstacles(self, obstacles: List[Obstacle]) -> 'PerceptionOutputBuilder':
        self._obstacle_list = obstacles
        return self
    
    def with_tracked_objects(self, objects: List[TrackedObject]) -> 'PerceptionOutputBuilder':
        self._tracked_objects = objects
        return self
    
    def with_occupancy_grid(self, grid: np.ndarray) -> 'PerceptionOutputBuilder':
        self._occupancy_grid = grid
        return self
    
    def with_world_model(self, model: Dict[str, Any]) -> 'PerceptionOutputBuilder':
        self._world_model = model
        return self
    
    def with_frame_id(self, frame_id: str) -> 'PerceptionOutputBuilder':
        self._frame_id = frame_id
        return self
    
    def with_processing_time(self, time_ms: float) -> 'PerceptionOutputBuilder':
        self._processing_time_ms = time_ms
        return self
    
    def build(self) -> PerceptionOutput:
        """Build PerceptionOutput with validation"""
        if self._robot_pose is None:
            self._robot_pose = RobotPose(
                x=0.0, y=0.0, theta=0.0,
                timestamp=self._timestamp,
                covariance=np.eye(3) * 0.1
            )
        
        return PerceptionOutput(
            timestamp=self._timestamp,
            robot_pose=self._robot_pose,
            obstacle_list=self._obstacle_list,
            tracked_objects=self._tracked_objects,
            occupancy_grid=self._occupancy_grid,
            world_model=self._world_model,
            frame_id=self._frame_id,
            processing_time_ms=self._processing_time_ms,
            detection_confidence_threshold=self._detection_threshold
        )

def create_empty_output() -> PerceptionOutput:
    """Create empty valid perception output"""
    builder = PerceptionOutputBuilder()
    builder.with_occupancy_grid(np.zeros((200, 200), dtype=np.int8))
    return builder.build()