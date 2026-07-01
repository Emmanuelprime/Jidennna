#!/usr/bin/env python3
"""
Main perception launch file.
Launches the complete perception system.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from perception_node import PerceptionNode
from config import PerceptionConfig, set_config

def main():
    parser = argparse.ArgumentParser(description='Launch Perception System')
    parser.add_argument('--config', type=str, default='config/default_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--sim', action='store_true',
                       help='Run in simulation mode')
    parser.add_argument('--visualize', action='store_true',
                       help='Enable visualization')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting perception system...")
    
    # Load configuration
    config_path = Path(__file__).parent.parent / args.config
    if config_path.exists():
        config = PerceptionConfig.from_yaml(str(config_path))
        logger.info(f"Loaded configuration from {config_path}")
    else:
        logger.warning(f"Config file not found: {config_path}")
        config = PerceptionConfig()
    
    # Override with command line arguments
    if args.sim:
        config.use_simulation = True
        config.camera.use_simulation = True
    
    if args.visualize:
        config.enable_visualization = True
        config.visualization.enabled = True
    
    # Set global config
    set_config(config)
    
    # Create and start perception node
    try:
        perception = PerceptionNode(
            config=config,
            use_simulation=config.use_simulation
        )
        
        perception.start()
        logger.info("Perception system running. Press Ctrl+C to stop.")
        
        # Keep running until interrupted
        import time
        try:
            while True:
                time.sleep(1.0)
                
                # Print status
                output = perception.get_perception_output()
                if output:
                    logger.info(f"Obstacles: {len(output.obstacle_list)}, "
                              f"Tracks: {len(output.tracked_objects)}")
        
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        
        finally:
            perception.stop()
            logger.info("Perception system stopped")
    
    except Exception as e:
        logger.error(f"Failed to start perception system: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()