#!/usr/bin/env python3
"""
Launch perception system in simulation mode.
"""

import sys
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

from perception_node import PerceptionNode
from config import PerceptionConfig

def main():
    parser = argparse.ArgumentParser(description='Launch Simulation Perception')
    parser.add_argument('--scenario', type=str, default='outdoor_campus',
                       help='Simulation scenario name')
    parser.add_argument('--visualize', action='store_true',
                       help='Enable visualization')
    
    args = parser.parse_args()
    
    # Configure for simulation
    config = PerceptionConfig()
    config.use_simulation = True
    config.camera.use_simulation = True
    config.enable_visualization = args.visualize
    
    # Run perception in simulation
    perception = PerceptionNode(config, use_simulation=True)
    
    try:
        perception.start()
        print(f"Simulation perception running with scenario: {args.scenario}")
        print("Press Ctrl+C to stop.")
        
        import time
        while True:
            time.sleep(1.0)
            
            output = perception.get_perception_output()
            if output:
                print(f"\rObstacles: {len(output.obstacle_list)} | "
                      f"Tracks: {len(output.tracked_objects)} | "
                      f"Grid: {output.occupancy_grid.shape}", end='')
    
    except KeyboardInterrupt:
        print("\nStopping simulation...")
        perception.stop()

if __name__ == '__main__':
    main()