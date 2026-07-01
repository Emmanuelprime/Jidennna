"""
Tests for camera layer.
"""

import unittest
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from camera.camera_interface import CameraInterface
from camera.usb_camera import USBCamera
from simulation.simulated_camera import SimulatedCamera

class TestCameraInterface(unittest.TestCase):
    """Test camera interface"""
    
    def test_abstract_class(self):
        """Test that CameraInterface cannot be instantiated"""
        with self.assertRaises(TypeError):
            camera = CameraInterface()
    
    def test_simulated_camera_initialization(self):
        """Test simulated camera initialization"""
        camera = SimulatedCamera({})
        self.assertTrue(camera.initialize())
        self.assertTrue(camera.is_healthy())
    
    def test_simulated_camera_frame(self):
        """Test frame acquisition from simulated camera"""
        camera = SimulatedCamera({})
        camera.initialize()
        
        frame, timestamp = camera.get_frame()
        
        self.assertIsNotNone(frame)
        self.assertEqual(len(frame.shape), 3)
        self.assertEqual(frame.shape[2], 3)  # BGR
        self.assertGreater(timestamp, 0)
    
    def test_simulated_camera_intrinsics(self):
        """Test camera intrinsics retrieval"""
        camera = SimulatedCamera({})
        camera.initialize()
        
        intrinsics = camera.get_intrinsics()
        
        self.assertIn('camera_matrix', intrinsics)
        self.assertEqual(intrinsics['camera_matrix'].shape, (3, 3))
    
    def test_camera_health_check(self):
        """Test health monitoring"""
        camera = SimulatedCamera({})
        camera.initialize()
        
        self.assertTrue(camera.is_healthy())
        
        camera.release()
        self.assertFalse(camera.is_healthy())

class TestCameraPerformance(unittest.TestCase):
    """Test camera performance metrics"""
    
    def test_frame_rate(self):
        """Test that camera achieves desired frame rate"""
        camera = SimulatedCamera({'fps': 30})
        camera.initialize()
        
        frames = []
        start_time = time.time()
        
        for _ in range(30):
            frame, _ = camera.get_frame()
            frames.append(frame)
        
        elapsed = time.time() - start_time
        actual_fps = len(frames) / elapsed
        
        # Should be within 20% of target
        self.assertGreater(actual_fps, 24)
        self.assertLess(actual_fps, 36)

if __name__ == '__main__':
    unittest.main()