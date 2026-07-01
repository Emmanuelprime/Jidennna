"""
Abstract camera interface that all camera implementations must follow.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
import numpy as np

class CameraInterface(ABC):
    """Abstract base class for all camera implementations"""
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize camera hardware
        Returns:
            bool: True if initialization successful
        """
        pass
    
    @abstractmethod
    def get_frame(self) -> Tuple[np.ndarray, float]:
        """Capture a frame from the camera
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
            Dict with camera matrix, distortion coefficients, etc.
        """
        pass
    
    @abstractmethod
    def release(self) -> None:
        """Release camera resources"""
        pass
    
    @abstractmethod
    def is_healthy(self) -> bool:
        """Check if camera is functioning properly
        Returns:
            bool: True if camera is healthy
        """
        pass
    
    @abstractmethod
    def set_parameter(self, param_id: int, value: Any) -> bool:
        """Set camera parameter
        Args:
            param_id: Parameter identifier
            value: Parameter value
        Returns:
            bool: True if successful
        """
        pass
    
    @abstractmethod
    def get_parameter(self, param_id: int) -> Any:
        """Get camera parameter
        Args:
            param_id: Parameter identifier
        Returns:
            Parameter value
        """
        pass