"""
Sensor noise models for realistic simulation.
"""

import numpy as np
from typing import Tuple, Optional, List
import cv2

class SensorNoise:
    """Adds realistic sensor noise to simulated data"""
    
    def __init__(self, seed: int = None):
        if seed is not None:
            np.random.seed(seed)
        
        # Noise parameters
        self.position_noise_std = 0.02  # meters
        self.velocity_noise_std = 0.05  # m/s
        self.detection_noise_std = 5.0  # pixels
        
        # Camera noise
        self.gaussian_noise_std = 5.0
        self.salt_pepper_prob = 0.001
        self.blur_kernel_size = 1
        
        # Failure modes
        self.frame_drop_prob = 0.01
        self.detection_miss_prob = 0.05
        self.false_positive_prob = 0.02
    
    def add_position_noise(self, position: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to position"""
        noise = np.random.normal(0, self.position_noise_std, position.shape)
        return position + noise
    
    def add_velocity_noise(self, velocity: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to velocity"""
        noise = np.random.normal(0, self.velocity_noise_std, velocity.shape)
        return velocity + noise
    
    def add_detection_noise(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """Add noise to bounding box coordinates"""
        x1, y1, x2, y2 = bbox
        
        noise = np.random.normal(0, self.detection_noise_std, 4)
        
        x1 = max(0, int(x1 + noise[0]))
        y1 = max(0, int(y1 + noise[1]))
        x2 = max(x1 + 10, int(x2 + noise[2]))
        y2 = max(y1 + 10, int(y2 + noise[3]))
        
        return (x1, y1, x2, y2)
    
    def add_camera_noise(self, image: np.ndarray) -> np.ndarray:
        """Add realistic camera noise to image"""
        noisy = image.copy()
        
        # Gaussian noise
        gaussian = np.random.normal(0, self.gaussian_noise_std, image.shape)
        noisy = noisy + gaussian
        
        # Salt and pepper noise
        mask = np.random.random(image.shape[:2]) < self.salt_pepper_prob
        noisy[mask] = np.random.randint(0, 2, noisy[mask].shape) * 255
        
        # Blur
        if self.blur_kernel_size > 1:
            noisy = cv2.GaussianBlur(noisy, (self.blur_kernel_size, self.blur_kernel_size), 0)
        
        return np.clip(noisy, 0, 255).astype(np.uint8)
    
    def should_drop_frame(self) -> bool:
        """Determine if frame should be dropped"""
        return np.random.random() < self.frame_drop_prob
    
    def should_miss_detection(self) -> bool:
        """Determine if detection should be missed"""
        return np.random.random() < self.detection_miss_prob
    
    def should_add_false_positive(self) -> bool:
        """Determine if false positive should be added"""
        return np.random.random() < self.false_positive_prob
    
    def generate_false_positive(self, image_shape: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
        """Generate a random false positive detection"""
        h, w = image_shape[:2]
        
        # Random position and size
        x1 = np.random.randint(0, w - 50)
        y1 = np.random.randint(0, h - 50)
        width = np.random.randint(20, 100)
        height = np.random.randint(20, 100)
        
        x2 = min(w, x1 + width)
        y2 = min(h, y1 + height)
        
        return (x1, y1, x2, y2)