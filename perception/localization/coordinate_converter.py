"""
Coordinate conversion utilities between different reference frames.
"""

import numpy as np
from typing import Tuple, List, Optional
from scipy.spatial.transform import Rotation
import logging

logger = logging.getLogger(__name__)

class CoordinateConverter:
    """Handles conversions between coordinate frames"""
    
    def __init__(self):
        # Frame transformations
        self.T_map_odom = np.eye(4)
        self.T_odom_base = np.eye(4)
        self.T_base_camera = np.eye(4)
        
        # Frame names
        self.frames = ['map', 'odom', 'base_link', 'camera']
    
    def set_transform(self, parent_frame: str, child_frame: str,
                     translation: Tuple[float, float, float],
                     rotation: Tuple[float, float, float, float]):
        """Set static transform between two frames
        
        Args:
            parent_frame: Parent frame name
            child_frame: Child frame name
            translation: (x, y, z) translation
            rotation: (x, y, z, w) quaternion rotation
        """
        T = np.eye(4)
        T[:3, 3] = translation
        T[:3, :3] = Rotation.from_quat(rotation).as_matrix()
        
        if parent_frame == 'map' and child_frame == 'odom':
            self.T_map_odom = T
        elif parent_frame == 'odom' and child_frame == 'base_link':
            self.T_odom_base = T
        elif parent_frame == 'base_link' and child_frame == 'camera':
            self.T_base_camera = T
        else:
            logger.warning(f"Unknown frame pair: {parent_frame} -> {child_frame}")
    
    def transform_point(self, point: Tuple[float, float, float],
                       from_frame: str, to_frame: str) -> Optional[Tuple[float, float, float]]:
        """Transform a point between coordinate frames
        
        Args:
            point: (x, y, z) point coordinates
            from_frame: Source frame name
            to_frame: Target frame name
            
        Returns:
            Transformed point or None if transformation not available
        """
        # Convert to homogeneous coordinates
        point_h = np.array([*point, 1.0])
        
        # Get transformation chain
        if from_frame == to_frame:
            return point
        
        # Camera to base_link
        if from_frame == 'camera' and to_frame == 'base_link':
            T = self.T_base_camera
            return tuple((T @ point_h)[:3])
        
        # Base link to odom
        elif from_frame == 'base_link' and to_frame == 'odom':
            T = self.T_odom_base
            return tuple((T @ point_h)[:3])
        
        # Odom to map
        elif from_frame == 'odom' and to_frame == 'map':
            T = self.T_map_odom
            return tuple((T @ point_h)[:3])
        
        # Reverse transformations
        elif from_frame == 'base_link' and to_frame == 'camera':
            T = np.linalg.inv(self.T_base_camera)
            return tuple((T @ point_h)[:3])
        
        elif from_frame == 'odom' and to_frame == 'base_link':
            T = np.linalg.inv(self.T_odom_base)
            return tuple((T @ point_h)[:3])
        
        elif from_frame == 'map' and to_frame == 'odom':
            T = np.linalg.inv(self.T_map_odom)
            return tuple((T @ point_h)[:3])
        
        else:
            logger.error(f"Unsupported transformation: {from_frame} -> {to_frame}")
            return None
    
    def transform_points(self, points: List[Tuple[float, float, float]],
                        from_frame: str, to_frame: str) -> List[Tuple[float, float, float]]:
        """Transform multiple points between frames"""
        return [self.transform_point(p, from_frame, to_frame) for p in points]
    
    def update_robot_pose(self, x: float, y: float, theta: float):
        """Update robot pose (base_link in odom frame)"""
        self.T_odom_base = np.eye(4)
        self.T_odom_base[:2, 3] = [x, y]
        
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        self.T_odom_base[:2, :2] = [[cos_t, -sin_t], [sin_t, cos_t]]
    
    def get_transform(self, from_frame: str, to_frame: str) -> Optional[np.ndarray]:
        """Get transformation matrix between frames"""
        if from_frame == 'camera' and to_frame == 'base_link':
            return self.T_base_camera
        elif from_frame == 'base_link' and to_frame == 'camera':
            return np.linalg.inv(self.T_base_camera)
        elif from_frame == 'base_link' and to_frame == 'odom':
            return self.T_odom_base
        elif from_frame == 'odom' and to_frame == 'base_link':
            return np.linalg.inv(self.T_odom_base)
        elif from_frame == 'odom' and to_frame == 'map':
            return self.T_map_odom
        elif from_frame == 'map' and to_frame == 'odom':
            return np.linalg.inv(self.T_map_odom)
        else:
            return None