"""
Abstract base class for all object detectors.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np
from ..interfaces.core_interfaces import Detection

class ObjectDetectorInterface(ABC):
    """Abstract interface for object detectors"""
    
    @abstractmethod
    def initialize(self, config: Dict) -> bool:
        """Initialize the detector with configuration
        Args:
            config: Dictionary with detector configuration
        Returns:
            bool: True if initialization successful
        """
        pass
    
    @abstractmethod
    def detect(self, image: np.ndarray) -> List[Detection]:
        """Detect objects in an image
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
    
    @abstractmethod
    def warmup(self, num_iterations: int = 3) -> None:
        """Warm up the detector with dummy inferences"""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model
        Returns:
            Dictionary with model metadata
        """
        pass