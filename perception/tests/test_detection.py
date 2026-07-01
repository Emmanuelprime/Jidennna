"""
Tests for detection layer.
"""

import unittest
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from interfaces.core_interfaces import Detection
from detection.detection_utils import ImagePreprocessor, NMSProcessor, DetectionUtils

class TestDetectionUtils(unittest.TestCase):
    """Test detection utilities"""
    
    def test_nms(self):
        """Test NMS implementation"""
        boxes = np.array([
            [100, 100, 200, 200],
            [105, 105, 205, 205],  # Highly overlapping with first
            [300, 300, 400, 400]   # Separate box
        ])
        scores = np.array([0.9, 0.8, 0.7])
        
        keep_indices = NMSProcessor.nms(boxes, scores, iou_threshold=0.5)
        
        # Should keep box 0 and box 2
        self.assertEqual(len(keep_indices), 2)
        self.assertIn(0, keep_indices)
        self.assertIn(2, keep_indices)
    
    def test_iou_calculation(self):
        """Test IOU calculation"""
        box1 = (0, 0, 100, 100)
        box2 = (50, 50, 150, 150)
        
        iou = DetectionUtils.calculate_iou(box1, box2)
        
        # Expected IOU: intersection=2500, union=17500, iou=0.1428
        self.assertAlmostEqual(iou, 0.1428, places=3)
    
    def test_image_preprocessing(self):
        """Test image preprocessing"""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        processed, scale, pad = ImagePreprocessor.resize_and_pad(
            image, target_size=(640, 640)
        )
        
        self.assertEqual(processed.shape, (640, 640, 3))
        self.assertGreater(scale, 0)
    
    def test_normalization(self):
        """Test image normalization"""
        image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        
        normalized = ImagePreprocessor.normalize(image)
        
        # Check that values are roughly zero-centered
        self.assertTrue(np.all(np.abs(normalized) < 3))

class TestDetectionDataStructures(unittest.TestCase):
    """Test detection data structures"""
    
    def test_detection_creation(self):
        """Test Detection dataclass"""
        det = Detection(
            class_name='person',
            confidence=0.95,
            bbox=(10, 10, 100, 200),
            detection_time=time.time()
        )
        
        self.assertEqual(det.class_name, 'person')
        self.assertEqual(det.confidence, 0.95)
        self.assertEqual(det.center, (55.0, 105.0))
    
    def test_detection_area(self):
        """Test bounding box area calculation"""
        det = Detection(
            class_name='car',
            confidence=0.8,
            bbox=(0, 0, 100, 50),
            detection_time=time.time()
        )
        
        self.assertEqual(det.area, 5000)

if __name__ == '__main__':
    unittest.main()