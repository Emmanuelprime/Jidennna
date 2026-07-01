# example_usage.py

import time
import numpy as np
from perception.perception_node import PerceptionNode
from perception.config import PerceptionConfig
from perception.interfaces.planner_interface import PlannerInterface

def main():
    """Example usage of perception layer"""
    
    # Configuration
    config = PerceptionConfig()
    config.camera_config.use_simulation = False  # Set to True for simulation
    config.camera_config.camera_type = "usb"
    config.camera_config.device_id = 0
    
    config.detection_config.model_type = "yolov8"
    config.detection_config.confidence_threshold = 0.5
    config.detection_config.model_path = "models/yolov8n.pt"
    
    config.tracking_config.tracker_type = "bytetrack"
    config.tracking_config.max_age = 30
    
    # Create perception node
    perception = PerceptionNode(
        config=config,
        use_simulation=config.camera_config.use_simulation
    )
    
    # Start perception pipeline
    perception.start()
    
    # Wait for initialization
    time.sleep(2.0)
    
    # Main loop (would be in ROS node in production)
    try:
        while True:
            # Get latest perception output
            output = perception.get_perception_output()
            
            if output and output.is_valid():
                # This data goes to Local Planner
                print("\n=== Perception Output ===")
                print(f"Timestamp: {output.timestamp:.3f}")
                print(f"Processing Time: {output.processing_time_ms:.1f}ms")
                print(f"Obstacles: {len(output.obstacle_list)}")
                print(f"Tracked Objects: {len(output.tracked_objects)}")
                
                # Example: Send to planner
                # planner.update(output)
                
                # Display obstacle positions
                for obstacle in output.obstacle_list:
                    print(f"  Obstacle {obstacle.id}: {obstacle.obstacle_type} "
                          f"at ({obstacle.position[0]:.2f}, {obstacle.position[1]:.2f})")
                
                # Occupancy grid statistics
                grid = output.occupancy_grid
                occupied_cells = np.sum(grid == 100)
                free_cells = np.sum(grid == 0)
                unknown_cells = np.sum(grid == -1)
                print(f"\nGrid: {occupied_cells} occupied, "
                      f"{free_cells} free, {unknown_cells} unknown")
            
            time.sleep(0.1)  # 10 Hz update rate
            
    except KeyboardInterrupt:
        print("\nStopping perception pipeline...")
    finally:
        perception.stop()

# Example with simulation
def run_simulation_example():
    """Example running perception in simulation"""
    
    config = PerceptionConfig()
    config.camera_config.use_simulation = True
    
    # Create perception node with simulation
    perception = PerceptionNode(
        config=config,
        use_simulation=True
    )
    
    # Create simulated world
    from perception.simulations.simulated_world import SimulatedWorld
    
    world = SimulatedWorld()
    
    # Add moving objects
    world.add_pedestrian((5, 0), (1, 0))
    world.add_pedestrian((3, 2), (-0.5, 0))
    world.add_vehicle((10, 0), (-2, 0))
    
    # Start perception
    perception.start()
    
    try:
        while True:
            # Update simulation
            world.update()
            
            # Get perception output
            output = perception.get_perception_output()
            
            if output:
                # Compare with ground truth
                ground_truth = world.get_ground_truth()
                
                print(f"Tracked: {len(output.tracked_objects)}, "
                      f"GT: {len(ground_truth)}")
            
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        perception.stop()

if __name__ == "__main__":
    main()