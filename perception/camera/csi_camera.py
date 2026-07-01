"""
CSI Camera implementation for Jetson Nano.
Uses GStreamer pipeline for hardware-accelerated capture.
"""

import cv2
import time
import logging
import numpy as np
from typing import Tuple, Dict, Any, Optional
from .camera_interface import CameraInterface

logger = logging.getLogger(__name__)

class CSICamera(CameraInterface):
    """CSI Camera implementation for NVIDIA Jetson platforms"""
    
    def __init__(self, sensor_id: int = 0, width: int = 1280, 
                 height: int = 720, fps: int = 30, flip_method: int = 0):
        """
        Args:
            sensor_id: CSI camera sensor ID (0 or 1)
            width: Frame width
            height: Frame height
            fps: Frames per second
            flip_method: Image flip method (0=none, 1=counterclockwise, 
                        2=rotate-180, 3=clockwise)
        """
        self.sensor_id = sensor_id
        self.width = width
        self.height = height
        self.fps = fps
        self.flip_method = flip_method
        self.pipeline = None
        self.cap = None
        self.is_initialized = False
        
    def initialize(self) -> bool:
        """Initialize CSI camera with GStreamer pipeline"""
        try:
            # Build GStreamer pipeline string
            pipeline_str = (
                f"nvarguscamerasrc sensor_id={self.sensor_id} ! "
                f"video/x-raw(memory:NVMM), "
                f"width=(int){self.width}, height=(int){self.height}, "
                f"format=(string)NV12, framerate=(fraction){self.fps}/1 ! "
                f"nvvidconv flip-method={self.flip_method} ! "
                f"video/x-raw, width=(int){self.width}, "
                f"height=(int){self.height}, format=(string)BGRx ! "
                f"videoconvert ! "
                f"video/x-raw, format=(string)BGR ! "
                f"appsink drop=true max-buffers=2"
            )
            
            logger.info(f"Initializing CSI camera with pipeline: {pipeline_str}")
            
            # Create VideoCapture with GStreamer pipeline
            self.cap = cv2.VideoCapture(pipeline_str, cv2.CAP_GSTREAMER)
            
            if not self.cap.isOpened():
                raise RuntimeError("Failed to open CSI camera")
            
            # Warm up
            for _ in range(10):
                self.cap.read()
                time.sleep(0.05)
            
            self.is_initialized = True
            logger.info(f"CSI camera {self.sensor_id} initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"CSI camera initialization failed: {e}")
            return False
    
    def get_frame(self) -> Tuple[np.ndarray, float]:
        """Capture frame from CSI camera"""
        if not self.is_initialized or self.cap is None:
            raise RuntimeError("CSI camera not initialized")
        
        ret, frame = self.cap.read()
        
        if not ret:
            logger.error("Failed to read CSI frame")
            raise RuntimeError("Frame capture failed")
        
        timestamp = time.time()
        return frame, timestamp
    
    def get_intrinsics(self) -> Dict[str, Any]:
        """Get CSI camera intrinsics"""
        # Typical CSI camera parameters (IMX219 sensor)
        fx = 2.8  # mm focal length
        sensor_width = 3.68  # mm
        pixel_size = sensor_width / self.width
        
        fx_pixels = fx / pixel_size
        fy_pixels = fx_pixels  # Square pixels
        
        return {
            'width': self.width,
            'height': self.height,
            'fps': self.fps,
            'camera_matrix': np.array([
                [fx_pixels, 0, self.width / 2.0],
                [0, fy_pixels, self.height / 2.0],
                [0, 0, 1]
            ]),
            'distortion_coeffs': np.zeros(5),
            'sensor_id': self.sensor_id
        }
    
    def set_parameter(self, param_id: int, value: Any) -> bool:
        """Set CSI camera parameter"""
        if self.cap is None:
            return False
        return self.cap.set(param_id, value)
    
    def get_parameter(self, param_id: int) -> Any:
        """Get CSI camera parameter"""
        if self.cap is None:
            return None
        return self.cap.get(param_id)
    
    def release(self) -> None:
        """Release CSI camera"""
        if self.cap:
            self.cap.release()
            self.cap = None
        self.is_initialized = False
        logger.info("CSI camera released")
    
    def is_healthy(self) -> bool:
        """Check CSI camera health"""
        if not self.is_initialized or self.cap is None:
            return False
        return self.cap.isOpened()