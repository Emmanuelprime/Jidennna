"""
Integration tests for complete perception pipeline.
"""

import unittest
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from perception_node import PerceptionNode
from config import PerceptionConfig

class TestPerceptionPipeline(unittest.TestCase):
    """Integration tests for perception pipeline"""
    
    def setUp(self):
        """Set up test configuration"""
        self.config = PerceptionConfig()
        self.config.use_simulation = True
        self.config.camera.use_simulation = True
        self.config.detection.confidence_threshold = 0.5
    
    def test_pipeline_initialization(self):
        """Test that pipeline can be created"""
        perception = PerceptionNode(self.config, use_simulation=True)
        self.assertIsNotNone(perception)
    
    def test_pipeline_start_stop(self):
        """Test pipeline start and stop"""
        perception = PerceptionNode(self.config, use_simulation=True)
        
        perception.start()
        time.sleep(2.0)  # Wait for pipeline to initialize
        
        self.assertTrue(perception.is_running)
        
        perception.stop()
        self.assertFalse(perception.is_running)
    
    def test_perception_output(self):
        """Test that pipeline produces valid output"""
        perception = PerceptionNode(self.config, use_simulation=True)
        perception.start()
        
        time.sleep(3.0)  # Wait for processing
        
        output = perception.get_perception_output()
        
        if output is not None:
            self.assertIsNotNone(output.obstacle_list)
            self.assertIsNotNone(output.occupancy_grid)
            self.assertIsNotNone(output.robot_pose)
        
        perception.stop()
    
    def test_configuration_loading(self):
        """Test configuration validation"""
        config = PerceptionConfig()
        
        # Test default values
        self.assertEqual(config.camera.width, 640)
        self.assertEqual(config.camera.height, 480)
        self.assertEqual(config.detection.confidence_threshold, 0.5)
        
        # Test validation
        self.assertTrue(config.validate())

class TestSimulationIntegration(unittest.TestCase):
    """Test simulation integration"""
    
    def test_simulation_switch(self):
        """Test switching between simulation and real hardware"""
        config_sim = PerceptionConfig()
        config_sim.use_simulation = True
        
        config_real = PerceptionConfig()
        config_real.use_simulation = False
        
        # Both should be valid configurations
        self.assertTrue(config_sim.validate())
        self.assertTrue(config_real.validate())
        
        # Simulation mode should use simulated camera
        perception_sim = PerceptionNode(config_sim, use_simulation=True)
        self.assertTrue(perception_sim.use_simulation)

if __name__ == '__main__':
    unittest.main()