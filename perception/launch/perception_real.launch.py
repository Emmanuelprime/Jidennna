#!/usr/bin/env python3
"""
Launch perception system with real hardware.
"""

import sys
from pathlib import Path
import argparse
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from perception_node import PerceptionNode
from config import PerceptionConfig

def main():
    parser = argparse.ArgumentParser(description='Launch Real Perception')
    parser.add_argument('--camera', type=str, default='usb',
                       choices=['usb', 'csi'],
                       help='Camera type')
    parser.add_argument('--camera-id', type=int, default=0,
                       help='Camera device ID')
    parser.add_argument('--model', type=str, default='models/yolov8n.pt',
                       help='Path to detection model')
    parser.add_argument('--tensorrt', action='store_true',
                       help='Use TensorRT optimization')
    
    args = parser.parse_args()
    
    # Configure for real hardware
    config = PerceptionConfig()
    config.use_simulation = False
    config.camera.use_simulation = False
    config.camera.camera_type = args.camera
    config.camera.device_id = args.camera_id
    config.detection.model_path = args.model
    config.detection.use_tensorrt = args.tensorrt
    
    # Run perception
    perception = PerceptionNode(config, use_simulation=False)
    
    try:
        perception.start()
        print("Real perception system running. Press Ctrl+C to stop.")
        
        import time
        while True:
            time.sleep(1.0)
    
    except KeyboardInterrupt:
        perception.stop()
        print("Perception system stopped")

if __name__ == '__main__':
    main()