"""
Coordinate transformations from image space to robot frame.
Supports monocular, stereo, and RGB-D depth estimation.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from scipy.spatial.transform import Rotation
import logging
from ..interfaces.core_interfaces import Detection, TrackedObject, Obstacle

logger = logging.getLogger(__name__)

class PoseTransform:
    """Transforms detections from image coordinates to robot/world frame"""
    
    def __init__(self, camera_config=None):
        """
        Args:
            camera_config: Camera configuration with intrinsics and mounting
        """
        # Camera intrinsics
        if camera_config:
            self.fx = camera_config.fx
            self.fy = camera_config.fy
            self.cx = camera_config.cx
            self.cy = camera_config.cy
        else:
            # Default values
            self.fx = 615.0
            self.fy = 615.0
            self.cx = 320.0
            self.cy = 240.0
        
        # Camera matrix
        self.K = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ])
        
        # Camera mounting parameters
        self.camera_height = getattr(camera_config, 'camera_height', 1.2)
        self.camera_pitch = np.deg2rad(getattr(camera_config, 'camera_pitch', -15.0))
        self.camera_x_offset = getattr(camera_config, 'camera_x_offset', 0.1)
        
        # Transformation matrices
        self.T_camera_robot = self._compute_camera_transform()
        self.T_robot_camera = np.linalg.inv(self.T_camera_robot)
        
        # Ground plane normal in robot frame
        self.ground_normal = np.array([0, 0, 1])
        self.ground_offset = 0.0  # Ground is at z=0
        
        # Depth estimation mode
        self.depth_mode = 'monocular'  # 'monocular', 'stereo', 'rgbd'
        self.depth_estimator = None
    
    def _compute_camera_transform(self) -> np.ndarray:
        """Compute camera to robot transformation matrix"""
        T = np.eye(4)
        
        # Translation
        T[0, 3] = self.camera_x_offset  # Forward offset
        T[1, 3] = 0.0  # Centered laterally
        T[2, 3] = self.camera_height  # Height above ground
        
        # Rotation (pitch down)
        R = Rotation.from_euler('y', self.camera_pitch).as_matrix()
        T[:3, :3] = R
        
        return T
    
    def image_to_ground_plane(self, image_point: Tuple[float, float],
                             depth: Optional[float] = None) -> Optional[np.ndarray]:
        """Convert image point to 3D point on ground plane
        
        Args:
            image_point: (u, v) coordinates in image
            depth: Optional depth measurement
            
        Returns:
            3D point in camera frame or None if point is above horizon
        """
        u, v = image_point
        
        if depth is not None:
            # Use provided depth (from stereo or RGB-D)
            return self._back_project(u, v, depth)
        else:
            # Monocular ground plane assumption
            return self._monocular_ground_plane(u, v)
    
    def _back_project(self, u: float, v: float, depth: float) -> np.ndarray:
        """Back-project pixel to 3D point using depth"""
        # Normalized image coordinates
        x_norm = (u - self.cx) / self.fx
        y_norm = (v - self.cy) / self.fy
        
        # 3D point in camera frame
        X = x_norm * depth
        Y = y_norm * depth
        Z = depth
        
        return np.array([X, Y, Z])
    
    def _monocular_ground_plane(self, u: float, v: float) -> Optional[np.ndarray]:
        """Estimate 3D position using ground plane assumption
        
        Assumes the detected point lies on the ground plane (z=0 in world).
        """
        # Check if point is above horizon
        if v <= self.cy:
            # Point is above horizon, can't be on ground
            return None
        
        # Calculate distance using similar triangles
        # For a point on the ground:
        # depth = camera_height * fy / (v - cy)
        
        # Pixel distance from principal point
        dy_pixels = v - self.cy
        
        if dy_pixels <= 0:
            return None
        
        # Distance to ground intersection
        depth = (self.camera_height * self.fy) / dy_pixels
        
        # Lateral offset
        dx_pixels = u - self.cx
        lateral = (dx_pixels * depth) / self.fx
        
        # Point in camera frame (z-forward, x-right, y-down)
        point_camera = np.array([lateral, -self.camera_height, depth, 1])
        
        # Transform to robot frame
        point_robot = self.T_camera_robot @ point_camera
        
        return point_robot[:3]
    
    def transform_detection(self, detection: Detection,
                          depth: Optional[float] = None) -> Optional[Obstacle]:
        """Transform single detection to robot frame obstacle
        
        Args:
            detection: Detection object with bbox
            depth: Optional depth measurement
            
        Returns:
            Obstacle in robot frame or None
        """
        # Get bottom center of bounding box
        x1, y1, x2, y2 = detection.bbox
        bottom_center_u = (x1 + x2) / 2
        bottom_center_v = y2  # Bottom of box is on ground
        
        # Convert to ground plane
        point_robot = self.image_to_ground_plane(
            (bottom_center_u, bottom_center_v), depth
        )
        
        if point_robot is None:
            return None
        
        # Create obstacle
        obstacle = Obstacle(
            id=0,  # Will be assigned later
            position=(point_robot[0], point_robot[1]),
            radius=self._estimate_radius(detection, point_robot),
            obstacle_type=detection.class_name,
            confidence=detection.confidence,
            timestamp=detection.detection_time
        )
        
        return obstacle
    
    def transform_detections(self, tracked_objects: List[TrackedObject],
                           timestamp: float,
                           depth_map: Optional[np.ndarray] = None) -> List[Obstacle]:
        """Transform multiple tracked objects to obstacles
        
        Args:
            tracked_objects: List of tracked objects
            timestamp: Current timestamp
            depth_map: Optional depth map for stereo/RGB-D
            
        Returns:
            List of obstacles in robot frame
        """
        obstacles = []
        
        for obj in tracked_objects:
            # Get depth if available
            depth = None
            if depth_map is not None:
                depth = self._get_depth_from_map(obj.bbox, depth_map)
            
            # Transform to robot frame
            obstacle = self.transform_detection_from_track(obj, depth)
            
            if obstacle is not None:
                obstacle.id = obj.track_id
                obstacle.timestamp = timestamp
                obstacles.append(obstacle)
        
        return obstacles
    
    def transform_detection_from_track(self, track: TrackedObject,
                                      depth: Optional[float] = None) -> Optional[Obstacle]:
        """Transform tracked object to obstacle"""
        x1, y1, x2, y2 = track.bbox
        bottom_center_u = (x1 + x2) / 2
        bottom_center_v = y2
        
        point_robot = self.image_to_ground_plane(
            (bottom_center_u, bottom_center_v), depth
        )
        
        if point_robot is None:
            return None
        
        # If we have tracked velocity, transform it to robot frame
        velocity_robot = None
        if track.velocity is not None:
            # Simplified: assume velocity in image plane corresponds to robot frame
            # In practice, would need proper coordinate transformation
            velocity_robot = track.velocity
        
        return Obstacle(
            id=track.track_id,
            position=(point_robot[0], point_robot[1]),
            radius=self._estimate_radius_from_track(track, point_robot),
            obstacle_type=track.class_name,
            velocity=velocity_robot,
            confidence=track.confidence,
            timestamp=0.0  # Will be set by caller
        )
    
    def _get_depth_from_map(self, bbox: Tuple[int, int, int, int],
                           depth_map: np.ndarray) -> Optional[float]:
        """Get depth value from depth map at bounding box center"""
        x1, y1, x2, y2 = bbox
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        
        h, w = depth_map.shape
        if 0 <= center_x < w and 0 <= center_y < h:
            depth = depth_map[center_y, center_x]
            if depth > 0:
                return float(depth)
        
        return None
    
    def _estimate_radius(self, detection: Detection, 
                        position: np.ndarray) -> float:
        """Estimate obstacle radius based on class and distance"""
        class_radii = {
            'person': 0.3,
            'bicycle': 0.5,
            'car': 1.5,
            'truck': 2.0,
            'bus': 2.5,
            'motorcycle': 0.4,
            'dog': 0.2,
            'cat': 0.15
        }
        
        base_radius = class_radii.get(detection.class_name, 0.5)
        
        # Scale with distance (objects appear smaller when far)
        distance = np.linalg.norm(position[:2])
        if distance < 5.0:
            return base_radius
        else:
            # Increase uncertainty with distance
            return base_radius * (1.0 + 0.1 * (distance - 5.0))
    
    def _estimate_radius_from_track(self, track: TrackedObject,
                                   position: np.ndarray) -> float:
        """Estimate radius from tracked object"""
        class_radii = {
            'person': 0.3,
            'bicycle': 0.5,
            'car': 1.5,
            'truck': 2.0,
            'bus': 2.5,
            'motorcycle': 0.4,
            'dog': 0.2,
            'cat': 0.15
        }
        
        base_radius = class_radii.get(track.class_name, 0.5)
        distance = np.linalg.norm(position[:2])
        
        # Use covariance to adjust radius
        if track.position_covariance is not None:
            uncertainty = np.sqrt(np.trace(track.position_covariance[:2, :2]))
            return base_radius + uncertainty
        elif distance < 5.0:
            return base_radius
        else:
            return base_radius * (1.0 + 0.1 * (distance - 5.0))
    
    def robot_to_world(self, point_robot: Tuple[float, float],
                      robot_pose: Tuple[float, float, float]) -> Tuple[float, float]:
        """Transform point from robot frame to world frame
        
        Args:
            point_robot: (x, y) in robot frame
            robot_pose: (x, y, theta) robot pose in world
            
        Returns:
            (x, y) in world frame
        """
        x, y = point_robot
        rx, ry, rtheta = robot_pose
        
        # Rotation
        cos_t = np.cos(rtheta)
        sin_t = np.sin(rtheta)
        
        x_world = rx + cos_t * x - sin_t * y
        y_world = ry + sin_t * x + cos_t * y
        
        return (x_world, y_world)
    
    def world_to_robot(self, point_world: Tuple[float, float],
                      robot_pose: Tuple[float, float, float]) -> Tuple[float, float]:
        """Transform point from world frame to robot frame"""
        x, y = point_world
        rx, ry, rtheta = robot_pose
        
        # Inverse rotation and translation
        dx = x - rx
        dy = y - ry
        
        cos_t = np.cos(rtheta)
        sin_t = np.sin(rtheta)
        
        x_robot = cos_t * dx + sin_t * dy
        y_robot = -sin_t * dx + cos_t * dy
        
        return (x_robot, y_robot)
    
    def set_depth_mode(self, mode: str, estimator=None):
        """Set depth estimation mode
        
        Args:
            mode: 'monocular', 'stereo', or 'rgbd'
            estimator: Optional depth estimator object
        """
        self.depth_mode = mode
        self.depth_estimator = estimator
        logger.info(f"Depth mode set to: {mode}")