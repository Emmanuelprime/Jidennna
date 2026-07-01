"""
Camera manager for handling multiple camera types and simulation.
"""

import time
import logging
import threading
from typing import Tuple, Optional, Dict, Any
import numpy as np
from .camera_interface import CameraInterface
from .usb_camera import USBCamera
from .csi_camera import CSICamera

logger = logging.getLogger(__name__)

class CameraManager:
    """Manages camera lifecycle with health monitoring and automatic recovery"""
    
    def __init__(self, use_simulation: bool = False, config: Dict = None):
        """
        Args:
            use_simulation: Whether to use simulated camera
            config: Camera configuration dictionary
        """
        self.use_simulation = use_simulation
        self.config = config or {}
        self.camera: Optional[CameraInterface] = None
        self.lock = threading.Lock()
        self.health_status = False
        self.last_healthy_time = time.time()
        self.error_count = 0
        self.max_errors = 5
        
        # Performance metrics
        self.fps_history = []
        self.frame_count = 0
        self.fps_start_time = time.time()
        
        self._initialize_camera()
    
    def _initialize_camera(self):
        """Initialize appropriate camera based on configuration"""
        try:
            if self.use_simulation:
                from simulation.simulated_camera import SimulatedCamera
                self.camera = SimulatedCamera(self.config)
                logger.info("Initialized simulated camera")
            else:
                camera_type = self.config.get('camera_type', 'usb')
                
                if camera_type == 'usb':
                    self.camera = USBCamera(
                        device_id=self.config.get('device_id', 0),
                        width=self.config.get('width', 640),
                        height=self.config.get('height', 480),
                        fps=self.config.get('fps', 30)
                    )
                elif camera_type == 'csi':
                    self.camera = CSICamera(
                        sensor_id=self.config.get('sensor_id', 0),
                        width=self.config.get('width', 1280),
                        height=self.config.get('height', 720),
                        fps=self.config.get('fps', 30)
                    )
                else:
                    raise ValueError(f"Unknown camera type: {camera_type}")
            
            if not self.camera.initialize():
                raise RuntimeError("Camera initialization failed")
            
            self.health_status = True
            self.last_healthy_time = time.time()
            logger.info(f"Camera manager initialized ({camera_type})")
            
        except Exception as e:
            logger.error(f"Camera initialization failed: {e}")
            self.health_status = False
            raise
    
    def get_frame(self) -> Tuple[Optional[np.ndarray], float]:
        """Get frame with health check and error recovery"""
        with self.lock:
            if not self.is_healthy():
                logger.warning("Camera unhealthy, attempting recovery")
                self._recover_camera()
                if not self.is_healthy():
                    return None, time.time()
            
            try:
                frame, timestamp = self.camera.get_frame()
                
                # Update FPS
                self.frame_count += 1
                elapsed = timestamp - self.fps_start_time
                if elapsed >= 1.0:
                    current_fps = self.frame_count / elapsed
                    self.fps_history.append(current_fps)
                    if len(self.fps_history) > 100:
                        self.fps_history.pop(0)
                    self.frame_count = 0
                    self.fps_start_time = timestamp
                
                # Update health status
                self.health_status = True
                self.last_healthy_time = timestamp
                self.error_count = 0
                
                return frame, timestamp
                
            except Exception as e:
                logger.error(f"Frame acquisition error: {e}")
                self.error_count += 1
                if self.error_count > self.max_errors:
                    self.health_status = False
                return None, time.time()
    
    def is_healthy(self) -> bool:
        """Check camera health"""
        if self.camera is None:
            return False
        
        # Check timeout
        if time.time() - self.last_healthy_time > 5.0:
            self.health_status = False
        
        return self.health_status and self.camera.is_healthy()
    
    def _recover_camera(self):
        """Attempt to recover camera"""
        try:
            logger.info("Attempting camera recovery...")
            self.release()
            time.sleep(2.0)
            self._initialize_camera()
            logger.info("Camera recovery successful")
        except Exception as e:
            logger.error(f"Camera recovery failed: {e}")
    
    def get_fps(self) -> float:
        """Get current FPS"""
        if not self.fps_history:
            return 0.0
        return np.mean(self.fps_history[-10:])
    
    def get_intrinsics(self) -> Dict[str, Any]:
        """Get camera intrinsics"""
        if self.camera:
            return self.camera.get_intrinsics()
        return {}
    
    def release(self):
        """Release camera resources"""
        with self.lock:
            if self.camera:
                self.camera.release()
            self.health_status = False
            logger.info("Camera manager released")