"""
Main perception orchestrator node.
"""

import time
import threading
import logging
from typing import Optional
from queue import Queue
import numpy as np

from camera.camera_manager import CameraManager
from detection.detector_factory import DetectorFactory
from tracking.track_manager import TrackManager
from localization.pose_transform import PoseTransform
from mapping.world_model import WorldModel
from interfaces.perception_output import PerceptionOutputBuilder
from interfaces.planner_interface import PlannerInterface
from interfaces.visualization_interface import VisualizationManager
from interfaces.data_logger import DataLogger
from config import PerceptionConfig

logger = logging.getLogger(__name__)

class PerceptionNode:
    """Main perception node orchestrating all subsystems"""
    
    def __init__(self, config: PerceptionConfig, use_simulation: bool = False):
        self.config = config
        self.use_simulation = use_simulation
        self.is_running = False
        
        # Queues
        self.frame_queue = Queue(maxsize=5)
        self.output_queue = Queue(maxsize=10)
        
        # Initialize components
        self._init_components()
        
        # Threading
        self.threads = []
        self.stop_event = threading.Event()
        
        # Performance
        self.processing_times = []
        self.frame_count = 0
        
        logger.info("Perception node initialized")
    
    def _init_components(self):
        """Initialize all perception components"""
        # Camera
        self.camera_manager = CameraManager(
            use_simulation=self.use_simulation,
            config=self.config.camera
        )
        
        # Detector
        self.detector = DetectorFactory.create_detector(self.config.detection)
        
        # Tracker
        self.track_manager = TrackManager(self.config.tracking)
        
        # Pose transform
        self.pose_transform = PoseTransform(self.config.camera)
        
        # World model
        self.world_model = WorldModel(self.config.mapping)
        
        # Interfaces
        self.planner_interface = PlannerInterface()
        
        if self.config.enable_visualization:
            self.visualizer = VisualizationManager()
        else:
            self.visualizer = None
        
        self.data_logger = DataLogger(self.config.logging)
    
    def start(self):
        """Start perception pipeline"""
        self.is_running = True
        
        # Start threads
        self.threads = [
            threading.Thread(target=self._acquisition_loop, name="acquisition"),
            threading.Thread(target=self._processing_loop, name="processing")
        ]
        
        for thread in self.threads:
            thread.daemon = True
            thread.start()
        
        logger.info("Perception pipeline started")
    
    def _acquisition_loop(self):
        """Camera frame acquisition loop"""
        while not self.stop_event.is_set():
            frame, timestamp = self.camera_manager.get_frame()
            
            if frame is not None and not self.frame_queue.full():
                self.frame_queue.put((frame, timestamp))
                self.frame_count += 1
            else:
                time.sleep(0.001)
    
    def _processing_loop(self):
        """Main processing loop"""
        while not self.stop_event.is_set():
            if self.frame_queue.empty():
                time.sleep(0.001)
                continue
            
            frame, timestamp = self.frame_queue.get()
            start_time = time.time()
            
            try:
                # Run pipeline
                output = self._process_frame(frame, timestamp)
                
                if output:
                    self.planner_interface.update(output)
                    
                    if not self.output_queue.full():
                        self.output_queue.put(output)
                    
                    # Visualization
                    if self.visualizer:
                        visuals = self.visualizer.render(output, frame)
                        self.visualizer.show(visuals)
                
                # Track performance
                proc_time = (time.time() - start_time) * 1000
                self.processing_times.append(proc_time)
                
            except Exception as e:
                logger.error(f"Processing error: {e}")
    
    def _process_frame(self, frame: np.ndarray, timestamp: float):
        """Process single frame through pipeline"""
        # Detection
        detections = self.detector.detect(frame)
        
        # Tracking
        tracked = self.track_manager.update(detections, timestamp)
        
        # Coordinate transform
        obstacles = self.pose_transform.transform_detections(tracked, timestamp)
        
        # World model
        self.world_model.update(tracked, obstacles, timestamp)
        
        # Build output
        builder = PerceptionOutputBuilder()
        output = (builder
            .with_timestamp(timestamp)
            .with_obstacles(self.world_model.get_obstacles())
            .with_tracked_objects(tracked)
            .with_occupancy_grid(self.world_model.get_occupancy_grid())
            .with_world_model(self.world_model.get_world_state())
            .with_processing_time(time.time() * 1000 - timestamp * 1000)
            .build())
        
        return output
    
    def get_perception_output(self):
        """Get latest perception output"""
        return self.planner_interface.get_latest()
    
    def stop(self):
        """Stop perception pipeline"""
        logger.info("Stopping perception pipeline...")
        self.stop_event.set()
        
        for thread in self.threads:
            thread.join(timeout=2.0)
        
        self.camera_manager.release()
        self.detector.shutdown()
        
        if self.visualizer:
            self.visualizer.cleanup()
        
        self.is_running = False
        logger.info("Perception pipeline stopped")

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    config = PerceptionConfig()
    config.use_simulation = True
    config.camera.use_simulation = True
    
    perception = PerceptionNode(config, use_simulation=True)
    
    try:
        perception.start()
        print("Running perception pipeline... Press Ctrl+C to stop")
        
        while True:
            time.sleep(1.0)
            output = perception.get_perception_output()
            if output:
                print(f"Obstacles: {len(output.obstacle_list)}, "
                      f"Tracks: {len(output.tracked_objects)}")
    
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        perception.stop()