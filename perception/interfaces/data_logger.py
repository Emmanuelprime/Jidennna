"""
Data logging and recording for perception debugging.
"""

import os
import time
import json
import pickle
import threading
from typing import Optional, Dict, Any, List
import numpy as np
from datetime import datetime
from pathlib import Path
from .core_interfaces import PerceptionOutput

class DataLogger:
    """Logs perception data for analysis and debugging"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.enabled = self.config.get('enabled', False)
        self.log_dir = Path(self.config.get('data_path', 'data/recordings'))
        self.record_images = self.config.get('record_images', False)
        
        # Current recording session
        self.session_id = None
        self.session_dir = None
        self.is_recording = False
        
        # Buffers
        self.detection_buffer = []
        self.frame_buffer = []
        self.max_buffer_size = 1000
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Statistics
        self.total_recorded_frames = 0
        
        if self.enabled:
            self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def start_recording(self, session_name: str = None) -> str:
        """Start a new recording session"""
        if not self.enabled:
            return None
        
        with self.lock:
            self.session_id = session_name or datetime.now().strftime("%Y%m%d_%H%M%S")
            self.session_dir = self.log_dir / self.session_id
            self.session_dir.mkdir(parents=True, exist_ok=True)
            self.is_recording = True
            self.total_recorded_frames = 0
            
            # Create metadata file
            metadata = {
                'session_id': self.session_id,
                'start_time': time.time(),
                'config': self.config
            }
            
            with open(self.session_dir / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return self.session_id
    
    def stop_recording(self):
        """Stop current recording session"""
        with self.lock:
            if self.is_recording:
                # Save remaining buffers
                self._flush_buffers()
                
                # Update metadata
                metadata_path = self.session_dir / 'metadata.json'
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    metadata['end_time'] = time.time()
                    metadata['total_frames'] = self.total_recorded_frames
                    metadata['duration'] = metadata['end_time'] - metadata['start_time']
                    
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=2)
                
                self.is_recording = False
                self.session_id = None
                self.session_dir = None
    
    def log_frame(self, frame: np.ndarray, perception_output: PerceptionOutput):
        """Log a frame and its perception output"""
        if not self.is_recording:
            return
        
        with self.lock:
            self.total_recorded_frames += 1
            
            # Log perception output
            output_dict = {
                'frame_id': self.total_recorded_frames,
                'timestamp': perception_output.timestamp,
                'robot_pose': {
                    'x': perception_output.robot_pose.x,
                    'y': perception_output.robot_pose.y,
                    'theta': perception_output.robot_pose.theta
                },
                'num_obstacles': len(perception_output.obstacle_list),
                'num_tracks': len(perception_output.tracked_objects)
            }
            
            self.detection_buffer.append(output_dict)
            
            # Optionally store image
            if self.record_images:
                self.frame_buffer.append({
                    'frame_id': self.total_recorded_frames,
                    'timestamp': perception_output.timestamp,
                    'image': frame.copy()
                })
            
            # Flush buffers if needed
            if len(self.detection_buffer) >= self.max_buffer_size:
                self._flush_buffers()
    
    def _flush_buffers(self):
        """Save buffered data to disk"""
        if not self.session_dir:
            return
        
        # Save detection data
        if self.detection_buffer:
            det_file = self.session_dir / f"detections_{int(time.time())}.json"
            with open(det_file, 'w') as f:
                json.dump(self.detection_buffer, f)
            self.detection_buffer = []
        
        # Save images
        if self.frame_buffer and self.record_images:
            img_file = self.session_dir / f"frames_{int(time.time())}.pkl"
            with open(img_file, 'wb') as f:
                pickle.dump(self.frame_buffer, f)
            self.frame_buffer = []
    
    def get_session_stats(self, session_id: str = None) -> Dict:
        """Get statistics for a recording session"""
        if session_id is None:
            session_id = self.session_id
        
        if session_id is None:
            return {}
        
        session_dir = self.log_dir / session_id
        metadata_path = session_dir / 'metadata.json'
        
        if not metadata_path.exists():
            return {}
        
        with open(metadata_path, 'r') as f:
            return json.load(f)