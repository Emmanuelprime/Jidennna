"""
USB Camera implementation using OpenCV.
Supports standard UVC (USB Video Class) cameras.
"""

import cv2
import time
import logging
import numpy as np
from typing import Tuple, Dict, Any, Optional
from .camera_interface import CameraInterface

logger = logging.getLogger(__name__)

class USBCamera(CameraInterface):
    """USB Camera implementation with robust error handling"""
    
    def __init__(self, device_id: int = 0, width: int = 640, 
                 height: int = 480, fps: int = 30):
        """
        Args:
            device_id: Camera device ID (0, 1, 2, ...)
            width: Desired frame width
            height: Desired frame height
            fps: Desired frames per second
        """
        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_initialized = False
        self.frame_count = 0
        self.start_time = time.time()
        
        # Camera parameters
        self.actual_width = width
        self.actual_height = height
        self.actual_fps = fps
        
    def initialize(self) -> bool:
        """Initialize USB camera with retry logic"""
        max_retries = 3
        retry_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Initializing USB camera {self.device_id} "
                          f"(attempt {attempt + 1}/{max_retries})")
                
                # Create VideoCapture object
                self.cap = cv2.VideoCapture(self.device_id)
                
                if not self.cap.isOpened():
                    raise RuntimeError(f"Cannot open camera {self.device_id}")
                
                # Set camera properties
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.cap.set(cv2.CAP_PROP_FPS, self.fps)
                self.cap.set(cv2.CAP_PROP_FOURCC, 
                            cv2.VideoWriter_fourcc(*'MJPG'))
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                # Auto exposure and white balance
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
                self.cap.set(cv2.CAP_PROP_AUTO_WB, 1)
                
                # Read actual parameters
                self.actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
                
                logger.info(f"Camera initialized: {self.actual_width}x{self.actual_height} "
                          f"@{self.actual_fps}fps")
                
                # Warm up camera (discard first few frames)
                for _ in range(10):
                    self.cap.read()
                    time.sleep(0.05)
                
                self.is_initialized = True
                self.frame_count = 0
                self.start_time = time.time()
                return True
                
            except Exception as e:
                logger.error(f"Camera initialization attempt {attempt + 1} failed: {e}")
                if self.cap:
                    self.cap.release()
                    self.cap = None
                
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
        
        logger.error(f"Failed to initialize camera after {max_retries} attempts")
        return False
    
    def get_frame(self) -> Tuple[np.ndarray, float]:
        """Capture frame with error handling and frame drop detection"""
        if not self.is_initialized or self.cap is None:
            raise RuntimeError("Camera not initialized")
        
        # Read frame
        ret, frame = self.cap.read()
        
        if not ret:
            logger.error("Failed to read frame")
            # Try to recover
            if not self._attempt_recovery():
                raise RuntimeError("Camera recovery failed")
            ret, frame = self.cap.read()
            if not ret:
                raise RuntimeError("Failed to read frame after recovery")
        
        timestamp = time.time()
        self.frame_count += 1
        
        # Check frame rate
        elapsed = timestamp - self.start_time
        if elapsed >= 5.0:  # Check every 5 seconds
            actual_fps = self.frame_count / elapsed
            if actual_fps < self.fps * 0.8:
                logger.warning(f"Frame rate low: {actual_fps:.1f} fps "
                             f"(expected {self.fps})")
            self.frame_count = 0
            self.start_time = timestamp
        
        return frame, timestamp
    
    def _attempt_recovery(self) -> bool:
        """Attempt to recover camera after failure"""
        logger.warning("Attempting camera recovery...")
        try:
            self.release()
            time.sleep(1.0)
            return self.initialize()
        except Exception as e:
            logger.error(f"Recovery failed: {e}")
            return False
    
    def get_intrinsics(self) -> Dict[str, Any]:
        """Get camera intrinsic parameters"""
        # Default intrinsics for common USB cameras
        # Should be replaced with actual calibration
        fx = self.actual_width * 0.96  # Approximate focal length
        fy = self.actual_height * 0.96
        cx = self.actual_width / 2.0
        cy = self.actual_height / 2.0
        
        return {
            'width': self.actual_width,
            'height': self.actual_height,
            'fps': self.actual_fps,
            'camera_matrix': np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ]),
            'distortion_coeffs': np.zeros(5),  # Assume no distortion
            'device_id': self.device_id
        }
    
    def set_parameter(self, param_id: int, value: Any) -> bool:
        """Set camera parameter"""
        if self.cap is None:
            return False
        return self.cap.set(param_id, value)
    
    def get_parameter(self, param_id: int) -> Any:
        """Get camera parameter"""
        if self.cap is None:
            return None
        return self.cap.get(param_id)
    
    def release(self) -> None:
        """Release camera resources"""
        if self.cap:
            self.cap.release()
            self.cap = None
        self.is_initialized = False
        logger.info("Camera released")
    
    def is_healthy(self) -> bool:
        """Check camera health"""
        if not self.is_initialized or self.cap is None:
            return False
        
        # Check if camera is still opened
        if not self.cap.isOpened():
            return False
        
        # Try to grab a frame
        ret = self.cap.grab()
        return ret
    
    def __del__(self):
        """Destructor to ensure camera is released"""
        self.release()