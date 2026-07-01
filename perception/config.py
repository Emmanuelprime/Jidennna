"""
Global configuration management for perception layer.
Supports YAML loading, environment variables, and runtime overrides.
"""

import os
import yaml
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class CameraConfig:
    """Camera configuration parameters"""
    width: int = 640
    height: int = 480
    fps: int = 30
    use_simulation: bool = False
    camera_type: str = "usb"  # 'usb', 'csi', 'simulation'
    device_id: int = 0
    sensor_id: int = 0
    
    # Camera intrinsics
    fx: float = 615.0
    fy: float = 615.0
    cx: float = 320.0
    cy: float = 240.0
    distortion_coeffs: Optional[List[float]] = None
    
    # Camera mounting parameters
    camera_height: float = 1.2  # meters
    camera_pitch: float = -15.0  # degrees
    camera_x_offset: float = 0.1  # meters forward
    
    def get_intrinsics_matrix(self) -> 'np.ndarray':
        """Get camera intrinsics as numpy array"""
        import numpy as np
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ])

@dataclass
class DetectionConfig:
    """Object detection configuration"""
    model_type: str = "yolov8"  # 'yolov8', 'yolov11', 'tensorrt'
    model_path: str = "models/yolov8n.pt"
    engine_path: Optional[str] = None  # TensorRT engine path
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.45
    input_size: tuple = (640, 640)
    use_tensorrt: bool = False
    device: str = "auto"  # 'cpu', 'cuda', 'auto'
    
    classes_of_interest: List[str] = field(default_factory=lambda: [
        'person', 'bicycle', 'car', 'motorcycle', 'bus', 
        'truck', 'dog', 'cat', 'backpack', 'suitcase'
    ])
    
    # Advanced settings
    use_fp16: bool = False
    batch_size: int = 1
    warmup_iterations: int = 3

@dataclass
class TrackingConfig:
    """Object tracking configuration"""
    tracker_type: str = "bytetrack"  # 'bytetrack', 'deepsort'
    max_age: int = 30
    min_hits: int = 3
    iou_threshold: float = 0.3
    use_kalman_filter: bool = True
    
    # Motion model
    process_noise: float = 0.01
    measurement_noise: float = 0.1
    
    # Track management
    max_tracks: int = 100
    min_confidence: float = 0.3
    
    # DeepSORT specific
    feature_dim: int = 512
    max_feature_distance: float = 0.2

@dataclass
class MappingConfig:
    """Mapping and world model configuration"""
    grid_width: int = 200  # cells
    grid_height: int = 200
    resolution: float = 0.05  # meters per cell
    update_frequency: float = 10.0  # Hz
    obstacle_inflation_radius: float = 0.3  # meters
    
    # Occupancy thresholds
    occupied_threshold: float = 0.65
    free_threshold: float = 0.2
    unknown_value: int = -1
    
    # Map management
    map_size_meters: float = 10.0  # 10x10 meter local map
    enable_global_map: bool = False
    global_map_path: Optional[str] = None

@dataclass
class LocalizationConfig:
    """Localization configuration"""
    use_imu: bool = True
    use_wheel_odometry: bool = True
    use_visual_odometry: bool = False
    
    # Sensor fusion
    ekf_process_noise: float = 0.1
    ekf_measurement_noise: float = 0.05
    
    # IMU settings
    imu_update_rate: float = 100.0  # Hz
    
    # Wheel odometry
    wheel_separation: float = 0.521  # meters
    wheel_radius: float = 0.085  # meters

@dataclass
class VisualizationConfig:
    """Visualization configuration"""
    enabled: bool = True
    show_detections: bool = True
    show_tracks: bool = True
    show_occupancy_grid: bool = True
    show_robot_pose: bool = True
    display_fps: bool = True
    
    # Window settings
    window_name: str = "Perception View"
    window_width: int = 1280
    window_height: int = 720
    
    # Colors (BGR)
    detection_color: tuple = (0, 255, 0)  # Green
    track_color: tuple = (255, 0, 0)  # Blue
    obstacle_color: tuple = (0, 0, 255)  # Red

@dataclass
class LoggingConfig:
    """Logging and data recording configuration"""
    log_level: str = "INFO"
    log_file: Optional[str] = "logs/perception.log"
    
    # Data recording
    record_data: bool = False
    data_path: str = "data/recordings"
    record_images: bool = False
    record_detections: bool = True
    max_recording_size_gb: float = 10.0

@dataclass
class PerceptionConfig:
    """Complete perception system configuration"""
    camera: CameraConfig = field(default_factory=CameraConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    mapping: MappingConfig = field(default_factory=MappingConfig)
    localization: LocalizationConfig = field(default_factory=LocalizationConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Pipeline settings
    pipeline_frequency: float = 30.0  # Hz
    enable_visualization: bool = True
    use_simulation: bool = False
    
    # Robot parameters
    robot_radius: float = 0.3  # meters
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'PerceptionConfig':
        """Load configuration from YAML file"""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(
            camera=CameraConfig(**config_dict.get('camera', {})),
            detection=DetectionConfig(**config_dict.get('detection', {})),
            tracking=TrackingConfig(**config_dict.get('tracking', {})),
            mapping=MappingConfig(**config_dict.get('mapping', {})),
            localization=LocalizationConfig(**config_dict.get('localization', {})),
            visualization=VisualizationConfig(**config_dict.get('visualization', {})),
            logging=LoggingConfig(**config_dict.get('logging', {})),
            **{k: v for k, v in config_dict.items() 
               if k not in ['camera', 'detection', 'tracking', 'mapping', 
                           'localization', 'visualization', 'logging']}
        )
    
    def to_yaml(self, yaml_path: str) -> None:
        """Save configuration to YAML file"""
        import dataclasses
        
        config_dict = {
            'camera': dataclasses.asdict(self.camera),
            'detection': dataclasses.asdict(self.detection),
            'tracking': dataclasses.asdict(self.tracking),
            'mapping': dataclasses.asdict(self.mapping),
            'localization': dataclasses.asdict(self.localization),
            'visualization': dataclasses.asdict(self.visualization),
            'logging': dataclasses.asdict(self.logging),
            'pipeline_frequency': self.pipeline_frequency,
            'enable_visualization': self.enable_visualization,
            'use_simulation': self.use_simulation,
            'robot_radius': self.robot_radius
        }
        
        os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    def validate(self) -> bool:
        """Validate configuration"""
        errors = []
        
        # Camera validation
        if self.camera.width <= 0 or self.camera.height <= 0:
            errors.append("Invalid camera dimensions")
        
        # Detection validation
        if not os.path.exists(self.detection.model_path):
            errors.append(f"Model not found: {self.detection.model_path}")
        
        # Mapping validation
        if self.mapping.resolution <= 0:
            errors.append("Invalid grid resolution")
        
        if errors:
            for error in errors:
                logger.error(f"Config validation error: {error}")
            return False
        
        return True

# Global config instance
_global_config: Optional[PerceptionConfig] = None

def get_config() -> PerceptionConfig:
    """Get global configuration instance"""
    global _global_config
    if _global_config is None:
        _global_config = PerceptionConfig()
    return _global_config

def set_config(config: PerceptionConfig) -> None:
    """Set global configuration instance"""
    global _global_config
    _global_config = config