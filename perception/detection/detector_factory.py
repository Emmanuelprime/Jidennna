"""
Factory pattern for creating detector instances.
"""

import logging
from typing import Dict
from .object_detector import ObjectDetectorInterface
from .yolo_detector import YOLODetector
from .tensorrt_detector import TensorRTDetector

logger = logging.getLogger(__name__)

class DetectorFactory:
    """Factory for creating appropriate detector based on configuration"""
    
    _detector_registry = {
        'yolov8': YOLODetector,
        'yolov11': YOLODetector,
        'tensorrt': TensorRTDetector,
        'yolo_pytorch': YOLODetector
    }
    
    @classmethod
    def register_detector(cls, name: str, detector_class: type):
        """Register a new detector type"""
        if not issubclass(detector_class, ObjectDetectorInterface):
            raise TypeError(f"Detector must implement ObjectDetectorInterface")
        cls._detector_registry[name] = detector_class
        logger.info(f"Registered detector: {name}")
    
    @classmethod
    def create_detector(cls, config) -> ObjectDetectorInterface:
        """Create detector instance based on configuration
        Args:
            config: DetectionConfig or dict with model_type
        Returns:
            Initialized detector instance
        """
        # Get detector type
        if hasattr(config, 'model_type'):
            detector_type = config.model_type
            use_tensorrt = config.use_tensorrt
        else:
            detector_type = config.get('model_type', 'yolov8')
            use_tensorrt = config.get('use_tensorrt', False)
        
        # Override with TensorRT if requested
        if use_tensorrt and 'tensorrt' not in detector_type:
            detector_type = 'tensorrt'
        
        # Get detector class
        detector_class = cls._detector_registry.get(detector_type)
        
        if detector_class is None:
            raise ValueError(f"Unknown detector type: {detector_type}. "
                           f"Available: {list(cls._detector_registry.keys())}")
        
        # Create instance
        detector = detector_class()
        
        # Prepare config dict
        if hasattr(config, '__dataclass_fields__'):
            import dataclasses
            config_dict = dataclasses.asdict(config)
        else:
            config_dict = config
        
        # Initialize
        if not detector.initialize(config_dict):
            raise RuntimeError(f"Failed to initialize {detector_type} detector")
        
        logger.info(f"Created detector: {detector_type}")
        return detector
    
    @classmethod
    def list_available_detectors(cls) -> list:
        """List all registered detector types"""
        return list(cls._detector_registry.keys())