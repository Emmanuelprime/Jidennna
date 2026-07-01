# tests/test_perception.py

import unittest
import numpy as np
from perception.camera.simulated_camera import SimulatedCamera
from perception.detection.yolo_detector import YOLODetector
from perception.tracking.track_manager import TrackManager
from perception.mapping.occupancy_grid import OccupancyGrid
from perception.mapping.world_model import WorldModel

class TestCameraLayer(unittest.TestCase):
    """Test camera layer functionality"""
    
    def setUp(self):
        self.config = {
            'width': 640,
            'height': 480,
            'fps': 30
        }
    
    def test_simulated_camera_creation(self):
        """Test simulated camera initialization"""
        camera = SimulatedCamera(self.config)
        self.assertTrue(camera.initialize())
        
    def test_frame_acquisition(self):
        """Test frame acquisition"""
        camera = SimulatedCamera(self.config)
        camera.initialize()
        
        frame, timestamp = camera.get_frame()
        
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, (480, 640, 3))
        self.assertGreater(timestamp, 0)
    
    def test_camera_health(self):
        """Test health check"""
        camera = SimulatedCamera(self.config)
        camera.initialize()
        
        self.assertTrue(camera.is_healthy())

class TestDetectionLayer(unittest.TestCase):
    """Test detection functionality"""
    
    def setUp(self):
        self.config = {
            'model_path': 'yolov8n.pt',
            'confidence_threshold': 0.5,
            'device': 'cpu'
        }
    
    def test_detector_initialization(self):
        """Test detector initialization"""
        detector = YOLODetector()
        result = detector.initialize(self.config)
        self.assertTrue(result)
    
    def test_detection_on_blank_image(self):
        """Test detection on blank image"""
        detector = YOLODetector()
        detector.initialize(self.config)
        
        # Create blank test image
        image = np.zeros((640, 640, 3), dtype=np.uint8)
        
        detections = detector.detect(image)
        self.assertEqual(len(detections), 0)  # No detections on blank
    
    def test_detection_format(self):
        """Test detection output format"""
        detector = YOLODetector()
        detector.initialize(self.config)
        
        image = np.ones((640, 640, 3), dtype=np.uint8) * 128
        
        detections = detector.detect(image)
        
        for det in detections:
            self.assertIsInstance(det.class_name, str)
            self.assertGreater(det.confidence, 0)
            self.assertLessEqual(det.confidence, 1)
            self.assertEqual(len(det.bbox), 4)

class TestTrackingLayer(unittest.TestCase):
    """Test tracking functionality"""
    
    def setUp(self):
        self.config = type('Config', (), {
            'tracker_type': 'bytetrack',
            'max_age': 30,
            'min_hits': 3,
            'iou_threshold': 0.3
        })
    
    def test_track_creation(self):
        """Test track creation"""
        tracker = TrackManager(self.config)
        
        # Create sample detections
        detections = [
            Detection('person', 0.9, (100, 100, 200, 300), time.time()),
            Detection('car', 0.8, (300, 200, 400, 350), time.time())
        ]
        
        tracks = tracker.update(detections, time.time())
        self.assertEqual(len(tracks), 2)
    
    def test_track_persistence(self):
        """Test track persistence across frames"""
        tracker = TrackManager(self.config)
        
        # Frame 1
        detections1 = [
            Detection('person', 0.9, (100, 100, 200, 300), time.time())
        ]
        tracks1 = tracker.update(detections1, time.time())
        
        # Frame 2 (same object, slightly moved)
        detections2 = [
            Detection('person', 0.9, (105, 105, 205, 305), time.time() + 0.1)
        ]
        tracks2 = tracker.update(detections2, time.time() + 0.1)
        
        # Should maintain same track ID
        self.assertEqual(tracks1[0].track_id, tracks2[0].track_id)

class TestMappingLayer(unittest.TestCase):
    """Test mapping functionality"""
    
    def setUp(self):
        self.config = type('Config', (), {
            'grid_width': 200,
            'grid_height': 200,
            'resolution': 0.05
        })
    
    def test_grid_creation(self):
        """Test occupancy grid creation"""
        grid = OccupancyGrid(200, 200, 0.05)
        self.assertEqual(grid.grid.shape, (200, 200))
    
    def test_coordinate_conversion(self):
        """Test world to grid coordinate conversion"""
        grid = OccupancyGrid(200, 200, 0.05)
        
        # Test origin
        gx, gy = grid.world_to_grid(0, 0)
        self.assertEqual(gx, 100)  # Center of 200x200 grid
        self.assertEqual(gy, 100)
    
    def test_obstacle_update(self):
        """Test obstacle update in grid"""
        grid = OccupancyGrid(200, 200, 0.05)
        
        obstacle = Obstacle(
            id=1,
            position=(0, 0),
            radius=0.3,
            obstacle_type='static'
        )
        
        grid.update_occupancy([obstacle], (0, 0))
        
        # Center should be occupied
        gx, gy = grid.world_to_grid(0, 0)
        self.assertEqual(grid.grid[gy, gx], 100)

class TestIntegration(unittest.TestCase):
    """Integration tests for full pipeline"""
    
    def test_end_to_end_pipeline(self):
        """Test complete perception pipeline"""
        from perception.perception_node import PerceptionNode
        from perception.config import PerceptionConfig
        
        config = PerceptionConfig()
        config.camera_config.use_simulation = True
        
        perception = PerceptionNode(config, use_simulation=True)
        perception.start()
        
        # Wait for pipeline to process
        import time
        time.sleep(3)
        
        output = perception.get_perception_output()
        
        self.assertIsNotNone(output)
        self.assertTrue(output.is_valid())
        
        perception.stop()
    
    def test_configuration_loading(self):
        """Test configuration loading"""
        from perception.config import PerceptionConfig
        
        config = PerceptionConfig()
        
        # Test default values
        self.assertEqual(config.camera_config.width, 640)
        self.assertEqual(config.camera_config.height, 480)
        self.assertEqual(config.detection_config.confidence_threshold, 0.5)

if __name__ == '__main__':
    unittest.main()