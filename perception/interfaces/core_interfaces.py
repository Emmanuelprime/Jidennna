"""
Core interfaces and data structures for the perception layer.

This module defines:
1. Data structures used throughout the perception pipeline
2. Abstract base classes for all pluggable components
3. Coordinate frame definitions

All modules in the perception layer should import from here
to maintain loose coupling and enable easy component swapping.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
from enum import Enum

# =================== COORDINATE FRAMES ===================

class FrameID(Enum):
    """Coordinate frame identifiers used throughout the system"""
    ROBOT = "base_link"      # Robot's local frame
    MAP = "map"              # Global map frame
    ODOM = "odom"            # Odometry frame
    CAMERA = "camera_frame"  # Camera optical frame
    WORLD = "world"          # World coordinate frame
    SENSOR = "sensor"        # Generic sensor frame
    
    @classmethod
    def from_string(cls, frame_str: str) -> 'FrameID':
        """Convert string to FrameID enum"""
        frame_map = {
            "base_link": cls.ROBOT,
            "map": cls.MAP,
            "odom": cls.ODOM,
            "camera_frame": cls.CAMERA,
            "camera": cls.CAMERA,
            "world": cls.WORLD,
            "sensor": cls.SENSOR
        }
        return frame_map.get(frame_str.lower(), cls.ROBOT)
    
    def __str__(self) -> str:
        return self.value

# =================== CORE DATA STRUCTURES ===================

@dataclass
class Detection:
    """Raw detection from object detector
    
    Attributes:
        class_name: Object class (e.g., 'person', 'car')
        confidence: Detection confidence [0.0, 1.0]
        bbox: Bounding box (x1, y1, x2, y2) in pixels
        detection_time: Timestamp of detection
        feature_vector: Optional appearance feature vector
    """
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    detection_time: float
    feature_vector: Optional[np.ndarray] = None
    
    @property
    def center(self) -> Tuple[float, float]:
        """Get bounding box center coordinates"""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    @property
    def width(self) -> int:
        """Get bounding box width"""
        return self.bbox[2] - self.bbox[0]
    
    @property
    def height(self) -> int:
        """Get bounding box height"""
        return self.bbox[3] - self.bbox[1]
    
    @property
    def area(self) -> float:
        """Get bounding box area"""
        return self.width * self.height
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'class_name': self.class_name,
            'confidence': self.confidence,
            'bbox': self.bbox,
            'detection_time': self.detection_time
        }

@dataclass
class TrackedObject:
    """Tracked object with temporal information
    
    Attributes:
        track_id: Unique track identifier
        class_name: Object class
        confidence: Tracking confidence
        position: Position (x, y) in robot frame (meters)
        velocity: Velocity (vx, vy) in m/s
        acceleration: Acceleration (ax, ay) in m/s²
        bbox: Current bounding box in image
        age: Number of frames tracked
        time_since_update: Frames since last update
        position_covariance: 2x2 position covariance matrix
    """
    track_id: int
    class_name: str
    confidence: float
    position: Tuple[float, float]  # x, y in robot frame
    velocity: Tuple[float, float]  # vx, vy
    acceleration: Tuple[float, float]  # ax, ay
    bbox: Tuple[int, int, int, int]
    age: int
    time_since_update: int
    position_covariance: np.ndarray  # 2x2 covariance matrix
    
    @property
    def speed(self) -> float:
        """Get current speed magnitude"""
        return np.sqrt(self.velocity[0]**2 + self.velocity[1]**2)
    
    @property
    def heading(self) -> float:
        """Get heading angle from velocity"""
        if self.speed > 0.1:
            return np.arctan2(self.velocity[1], self.velocity[0])
        return 0.0
    
    @property
    def is_static(self) -> bool:
        """Check if object is static"""
        return self.speed < 0.1
    
    def predict_future_position(self, dt: float) -> Tuple[float, float]:
        """Predict future position assuming constant velocity
        
        Args:
            dt: Time horizon in seconds
            
        Returns:
            Predicted position (x, y)
        """
        x = self.position[0] + self.velocity[0] * dt
        y = self.position[1] + self.velocity[1] * dt
        return (x, y)

@dataclass
class Obstacle:
    """Obstacle in robot or global frame
    
    Attributes:
        id: Unique obstacle identifier
        position: Position (x, y) in specified frame (meters)
        radius: Obstacle radius for collision checking (meters)
        obstacle_type: Type classification
        velocity: Optional velocity vector (vx, vy) in m/s
        confidence: Detection confidence
        timestamp: Observation timestamp
        frame_id: Coordinate frame of position
    """
    id: int
    position: Tuple[float, float]  # x, y
    radius: float
    obstacle_type: str  # 'static', 'dynamic', 'person', 'vehicle', etc.
    velocity: Optional[Tuple[float, float]] = None
    confidence: float = 1.0
    timestamp: float = 0.0
    frame_id: str = "base_link"  # Default to robot frame
    
    @property
    def is_dynamic(self) -> bool:
        """Check if obstacle is dynamic"""
        return self.velocity is not None and np.linalg.norm(self.velocity) > 0.1
    
    @property
    def speed(self) -> float:
        """Get speed if dynamic"""
        if self.velocity:
            return np.linalg.norm(self.velocity)
        return 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'id': self.id,
            'position': self.position,
            'radius': self.radius,
            'type': self.obstacle_type,
            'velocity': self.velocity,
            'confidence': self.confidence,
            'timestamp': self.timestamp,
            'frame_id': self.frame_id
        }
    
    def distance_to(self, point: Tuple[float, float]) -> float:
        """Calculate distance to a point"""
        return np.sqrt(
            (self.position[0] - point[0])**2 + 
            (self.position[1] - point[1])**2
        )

@dataclass
class RobotPose:
    """Robot pose estimation
    
    Attributes:
        x, y: Position in specified frame (meters)
        theta: Orientation angle (radians)
        timestamp: Pose timestamp
        covariance: 3x3 pose covariance matrix [x, y, theta]
        frame_id: Coordinate frame of pose
    """
    x: float
    y: float
    theta: float
    timestamp: float
    covariance: np.ndarray  # 3x3 covariance matrix
    frame_id: str = "map"  # Default to map frame
    
    @property
    def position(self) -> Tuple[float, float]:
        """Get position tuple"""
        return (self.x, self.y)
    
    @property
    def heading_vector(self) -> Tuple[float, float]:
        """Get unit heading vector"""
        return (np.cos(self.theta), np.sin(self.theta))
    
    def to_transform_matrix(self) -> np.ndarray:
        """Convert to 3x3 homogeneous transformation matrix"""
        cos_t = np.cos(self.theta)
        sin_t = np.sin(self.theta)
        return np.array([
            [cos_t, -sin_t, self.x],
            [sin_t, cos_t, self.y],
            [0, 0, 1]
        ])
    
    def transform_point(self, point: Tuple[float, float]) -> Tuple[float, float]:
        """Transform point from robot frame to global frame"""
        cos_t = np.cos(self.theta)
        sin_t = np.sin(self.theta)
        x = self.x + cos_t * point[0] - sin_t * point[1]
        y = self.y + sin_t * point[0] + cos_t * point[1]
        return (x, y)

@dataclass
class PerceptionOutput:
    """Standardized output for planner consumption
    
    This is the ONLY data structure that should leave the perception layer.
    Planners must NOT access internal perception components.
    
    Attributes:
        timestamp: Output timestamp
        robot_pose: Current robot pose
        obstacle_list: All detected obstacles
        tracked_objects: Actively tracked dynamic objects
        occupancy_grid: 2D occupancy grid map
        world_model: Additional world state information
        frame_id: Default coordinate frame for positions
        processing_time_ms: Pipeline processing time
        detection_confidence_threshold: Confidence threshold used
    """
    timestamp: float
    robot_pose: RobotPose
    obstacle_list: List[Obstacle]
    tracked_objects: List[TrackedObject]
    occupancy_grid: np.ndarray
    world_model: Dict[str, Any]
    frame_id: str = "base_link"
    processing_time_ms: float = 0.0
    detection_confidence_threshold: float = 0.5
    
    def is_valid(self) -> bool:
        """Check if perception output is valid"""
        return (self.obstacle_list is not None and 
                self.occupancy_grid is not None and
                self.robot_pose is not None and
                self.processing_time_ms >= 0)
    
    def get_obstacles_in_radius(self, radius: float, 
                                center: Optional[Tuple[float, float]] = None) -> List[Obstacle]:
        """Get obstacles within given radius
        
        Args:
            radius: Search radius in meters
            center: Center point (uses robot position if None)
            
        Returns:
            List of obstacles within radius
        """
        if center is None:
            center = (0, 0) if self.frame_id == "base_link" else self.robot_pose.position
        
        return [
            obs for obs in self.obstacle_list
            if obs.distance_to(center) <= radius
        ]
    
    def get_nearest_obstacle(self) -> Optional[Obstacle]:
        """Get nearest obstacle to robot"""
        if not self.obstacle_list:
            return None
        
        return min(self.obstacle_list, 
                  key=lambda obs: obs.distance_to((0, 0)))
    
    def get_dynamic_obstacles(self) -> List[Obstacle]:
        """Get only dynamic obstacles"""
        return [obs for obs in self.obstacle_list if obs.is_dynamic]
    
    def get_static_obstacles(self) -> List[Obstacle]:
        """Get only static obstacles"""
        return [obs for obs in self.obstacle_list if not obs.is_dynamic]
    
    def to_dict(self) -> Dict:
        """Convert to serializable dictionary"""
        return {
            'timestamp': self.timestamp,
            'robot_pose': {
                'x': self.robot_pose.x,
                'y': self.robot_pose.y,
                'theta': self.robot_pose.theta,
                'covariance': self.robot_pose.covariance.tolist()
            },
            'obstacles': [obs.to_dict() for obs in self.obstacle_list],
            'tracked_objects_count': len(self.tracked_objects),
            'occupancy_grid_shape': self.occupancy_grid.shape,
            'frame_id': self.frame_id,
            'processing_time_ms': self.processing_time_ms
        }

# =================== CAMERA INTERFACE ===================

class CameraInterface(ABC):
    """Abstract camera interface
    
    All camera implementations (USB, CSI, simulated) must implement this interface.
    This ensures that the perception pipeline works identically regardless of
    the camera source.
    """
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize camera hardware/simulation
        
        Returns:
            bool: True if initialization successful
        """
        pass
    
    @abstractmethod
    def get_frame(self) -> Tuple[np.ndarray, float]:
        """Get camera frame with timestamp
        
        Returns:
            Tuple containing:
            - frame: numpy array (height, width, 3) BGR format
            - timestamp: float seconds
        """
        pass
    
    @abstractmethod
    def get_intrinsics(self) -> Dict[str, Any]:
        """Get camera intrinsic parameters
        
        Returns:
            Dictionary with camera matrix, distortion coefficients, etc.
        """
        pass
    
    @abstractmethod
    def release(self) -> None:
        """Release camera resources"""
        pass
    
    @abstractmethod
    def is_healthy(self) -> bool:
        """Check camera health status
        
        Returns:
            bool: True if camera is functioning properly
        """
        pass
    
    def set_parameter(self, param_id: int, value: Any) -> bool:
        """Set camera parameter (optional implementation)
        
        Args:
            param_id: Parameter identifier
            value: Parameter value
            
        Returns:
            bool: True if successful
        """
        return False
    
    def get_parameter(self, param_id: int) -> Any:
        """Get camera parameter (optional implementation)
        
        Args:
            param_id: Parameter identifier
            
        Returns:
            Parameter value or None
        """
        return None

# =================== DETECTOR INTERFACE ===================

class ObjectDetectorInterface(ABC):
    """Abstract object detector
    
    All object detectors (YOLO, TensorRT, etc.) must implement this interface.
    This enables hot-swapping detection models without changing other components.
    """
    
    @abstractmethod
    def initialize(self, config: Dict) -> bool:
        """Initialize detection model
        
        Args:
            config: Dictionary with detector configuration
            
        Returns:
            bool: True if initialization successful
        """
        pass
    
    @abstractmethod
    def detect(self, image: np.ndarray) -> List[Detection]:
        """Detect objects in image
        
        Args:
            image: Input image (H, W, 3) BGR format
            
        Returns:
            List of Detection objects
        """
        pass
    
    @abstractmethod
    def get_supported_classes(self) -> List[str]:
        """Get list of supported object classes
        
        Returns:
            List of class name strings
        """
        pass
    
    @abstractmethod
    def shutdown(self) -> None:
        """Clean up detector resources"""
        pass
    
    def warmup(self, num_iterations: int = 3) -> None:
        """Warm up the detector (optional implementation)
        
        Args:
            num_iterations: Number of warmup iterations
        """
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded model (optional implementation)
        
        Returns:
            Dictionary with model metadata
        """
        return {}

# =================== TRACKER INTERFACE ===================

class ObjectTrackerInterface(ABC):
    """Abstract object tracker
    
    All trackers (ByteTrack, DeepSORT, etc.) must implement this interface.
    This enables swapping tracking algorithms without affecting other modules.
    """
    
    @abstractmethod
    def initialize(self, config: Dict) -> bool:
        """Initialize tracker
        
        Args:
            config: Dictionary with tracker configuration
            
        Returns:
            bool: True if initialization successful
        """
        pass
    
    @abstractmethod
    def update(self, detections: List[Detection], 
               timestamp: float) -> List[TrackedObject]:
        """Update tracks with new detections
        
        Args:
            detections: List of new detections
            timestamp: Current timestamp
            
        Returns:
            List of tracked objects
        """
        pass
    
    @abstractmethod
    def get_active_tracks(self) -> List[TrackedObject]:
        """Get currently active tracks
        
        Returns:
            List of active tracked objects
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset all tracks"""
        pass
    
    def get_track_count(self) -> int:
        """Get number of active tracks (optional implementation)
        
        Returns:
            Number of active tracks
        """
        return len(self.get_active_tracks())
    
    def get_track_by_id(self, track_id: int) -> Optional[TrackedObject]:
        """Get specific track by ID (optional implementation)
        
        Args:
            track_id: Track identifier
            
        Returns:
            TrackedObject if found, None otherwise
        """
        for track in self.get_active_tracks():
            if track.track_id == track_id:
                return track
        return None

# =================== MAPPING INTERFACE ===================

class WorldModelInterface(ABC):
    """Abstract world model
    
    All world model implementations must follow this interface.
    This enables different mapping approaches while maintaining
    a consistent interface for planners.
    """
    
    @abstractmethod
    def update(self, tracked_objects: List[TrackedObject],
               robot_pose: RobotPose, 
               sensor_data: Dict) -> None:
        """Update world model with new observations
        
        Args:
            tracked_objects: List of tracked dynamic objects
            robot_pose: Current robot pose
            sensor_data: Additional sensor data (LiDAR, ultrasonic, etc.)
        """
        pass
    
    @abstractmethod
    def get_obstacles(self) -> List[Obstacle]:
        """Get all obstacles in robot frame
        
        Returns:
            List of Obstacle objects
        """
        pass
    
    @abstractmethod
    def get_occupancy_grid(self) -> np.ndarray:
        """Get current occupancy grid
        
        Returns:
            2D numpy array with occupancy probabilities
        """
        pass
    
    @abstractmethod
    def get_world_state(self) -> Dict:
        """Get complete world state
        
        Returns:
            Dictionary with world state information
        """
        pass
    
    def get_costmap(self) -> np.ndarray:
        """Get costmap for planning (optional implementation)
        
        Returns:
            2D numpy array with cost values
        """
        grid = self.get_occupancy_grid()
        costmap = np.zeros_like(grid, dtype=np.uint8)
        costmap[grid >= 80] = 254  # Lethal
        costmap[grid == -1] = 255  # Unknown
        return costmap
    
    def is_collision_free(self, point: Tuple[float, float], 
                         radius: float = 0.3) -> bool:
        """Check if a point is collision-free (optional implementation)
        
        Args:
            point: (x, y) position to check
            radius: Safety radius in meters
            
        Returns:
            bool: True if collision-free
        """
        return True
    
    def clear(self) -> None:
        """Clear the world model (optional implementation)"""
        pass

# =================== UTILITY FUNCTIONS ===================

def create_empty_detection() -> Detection:
    """Create an empty detection placeholder"""
    return Detection(
        class_name="unknown",
        confidence=0.0,
        bbox=(0, 0, 0, 0),
        detection_time=0.0
    )

def create_empty_obstacle() -> Obstacle:
    """Create an empty obstacle placeholder"""
    return Obstacle(
        id=-1,
        position=(0.0, 0.0),
        radius=0.5,
        obstacle_type="unknown"
    )

def create_default_robot_pose() -> RobotPose:
    """Create a default robot pose"""
    import time
    return RobotPose(
        x=0.0,
        y=0.0,
        theta=0.0,
        timestamp=time.time(),
        covariance=np.eye(3) * 0.1
    )

# =================== TYPE ALIASES ===================

# Type aliases for common types
ImageType = np.ndarray  # (H, W, 3) BGR image
DepthMapType = np.ndarray  # (H, W) depth map
PointCloudType = np.ndarray  # (N, 3) point cloud
GridMapType = np.ndarray  # (H, W) grid map
TransformMatrix = np.ndarray  # 4x4 transformation matrix