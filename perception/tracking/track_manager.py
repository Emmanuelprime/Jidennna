"""
Track lifecycle manager.
Handles track creation, updating, and removal policies.
"""

import time
import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import logging
from .object_tracker import ObjectTrackerInterface
from .bytetrack_tracker import ByteTrackTracker
from .deepsort_tracker import DeepSORTTracker
from ..interfaces.core_interfaces import Detection, TrackedObject

logger = logging.getLogger(__name__)

class TrackManager:
    """Manages tracking pipeline and track lifecycle"""
    
    def __init__(self, config=None):
        self.config = config
        self.tracker = self._create_tracker()
        
        # Track storage
        self.tracks: Dict[int, TrackedObject] = {}
        self.track_history: Dict[int, List[Dict]] = defaultdict(list)
        self.max_history_length = 1000
        
        # Track statistics
        self.total_tracks_created = 0
        self.total_tracks_lost = 0
        self.current_frame_id = 0
        
        # Track filtering
        self.static_object_threshold = 0.1  # m/s - below this is static
        self.max_track_speed = 10.0  # m/s - above this is likely noise
        
    def _create_tracker(self) -> ObjectTrackerInterface:
        """Create tracker based on configuration"""
        tracker_type = getattr(self.config, 'tracker_type', 'bytetrack')
        
        if tracker_type == 'deepsort':
            tracker = DeepSORTTracker()
        else:
            tracker = ByteTrackTracker()
        
        # Initialize with config
        tracker_config = {
            'max_age': getattr(self.config, 'max_age', 30),
            'min_hits': getattr(self.config, 'min_hits', 3),
            'iou_threshold': getattr(self.config, 'iou_threshold', 0.3)
        }
        
        tracker.initialize(tracker_config)
        return tracker
    
    def update(self, detections: List[Detection], 
               timestamp: float) -> List[TrackedObject]:
        """Update tracks with new detections"""
        self.current_frame_id += 1
        
        # Update tracker
        tracked_objects = self.tracker.update(detections, timestamp)
        
        # Post-process tracks
        tracked_objects = self._post_process_tracks(tracked_objects, timestamp)
        
        # Update track history
        self._update_history(tracked_objects, timestamp)
        
        # Update internal track storage
        for obj in tracked_objects:
            self.tracks[obj.track_id] = obj
        
        # Clean up old tracks
        self._cleanup_old_tracks()
        
        return tracked_objects
    
    def _post_process_tracks(self, tracks: List[TrackedObject],
                            timestamp: float) -> List[TrackedObject]:
        """Apply post-processing filters to tracks"""
        filtered_tracks = []
        
        for track in tracks:
            # Filter by speed
            speed = np.linalg.norm(track.velocity)
            if speed > self.max_track_speed:
                logger.warning(f"Track {track.track_id} has unrealistic speed: {speed:.1f} m/s")
                continue
            
            # Smooth velocity if track has history
            if track.track_id in self.track_history:
                track = self._smooth_track(track)
            
            filtered_tracks.append(track)
        
        return filtered_tracks
    
    def _smooth_track(self, track: TrackedObject) -> TrackedObject:
        """Apply temporal smoothing to track"""
        history = self.track_history[track.track_id]
        
        if len(history) < 2:
            return track
        
        # Exponential moving average for velocity
        alpha = 0.7
        prev_vel = history[-1].get('velocity', (0, 0))
        
        smoothed_vel = (
            alpha * track.velocity[0] + (1 - alpha) * prev_vel[0],
            alpha * track.velocity[1] + (1 - alpha) * prev_vel[1]
        )
        
        # Update track with smoothed velocity
        track.velocity = smoothed_vel
        return track
    
    def _update_history(self, tracks: List[TrackedObject], timestamp: float):
        """Update track history"""
        for track in tracks:
            history_entry = {
                'timestamp': timestamp,
                'position': track.position,
                'velocity': track.velocity,
                'confidence': track.confidence,
                'frame_id': self.current_frame_id
            }
            
            self.track_history[track.track_id].append(history_entry)
            
            # Limit history size
            if len(self.track_history[track.track_id]) > self.max_history_length:
                self.track_history[track.track_id] = \
                    self.track_history[track.track_id][-self.max_history_length//2:]
    
    def _cleanup_old_tracks(self):
        """Remove old tracks that are no longer active"""
        active_ids = set(self.tracker.get_active_tracks())
        
        for track_id in list(self.tracks.keys()):
            if track_id not in active_ids:
                self.total_tracks_lost += 1
                del self.tracks[track_id]
    
    def get_track_statistics(self, track_id: int) -> Dict:
        """Get statistics for a specific track"""
        if track_id not in self.track_history:
            return {}
        
        history = self.track_history[track_id]
        
        if len(history) < 2:
            return {
                'total_distance': 0,
                'average_speed': 0,
                'max_speed': 0,
                'duration': 0,
                'is_static': True
            }
        
        positions = np.array([h['position'] for h in history])
        velocities = np.array([h['velocity'] for h in history])
        speeds = np.linalg.norm(velocities, axis=1)
        
        # Calculate total distance traveled
        distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        total_distance = np.sum(distances)
        
        # Calculate statistics
        avg_speed = np.mean(speeds)
        max_speed = np.max(speeds)
        duration = history[-1]['timestamp'] - history[0]['timestamp']
        
        return {
            'total_distance': total_distance,
            'average_speed': avg_speed,
            'max_speed': max_speed,
            'duration': duration,
            'is_static': avg_speed < self.static_object_threshold,
            'frames_tracked': len(history)
        }
    
    def get_tracks_near_point(self, point: Tuple[float, float],
                              radius: float = 2.0) -> List[TrackedObject]:
        """Get all tracks within radius of a point"""
        nearby_tracks = []
        
        for track in self.tracks.values():
            distance = np.sqrt(
                (track.position[0] - point[0])**2 +
                (track.position[1] - point[1])**2
            )
            if distance <= radius:
                nearby_tracks.append(track)
        
        return nearby_tracks
    
    def predict_track_future(self, track_id: int, 
                            time_horizon: float) -> Optional[List[Tuple[float, float]]]:
        """Predict future positions of a track"""
        if track_id not in self.tracks:
            return None
        
        track = self.tracks[track_id]
        positions = [track.position]
        
        dt = 0.1  # 100ms prediction steps
        steps = int(time_horizon / dt)
        
        pos = np.array(track.position, dtype=float)
        vel = np.array(track.velocity, dtype=float)
        
        for _ in range(steps):
            # Simple constant velocity prediction
            pos = pos + vel * dt
            positions.append(tuple(pos))
        
        return positions
    
    def get_active_track_count(self) -> int:
        """Get number of active tracks"""
        return len(self.tracks)
    
    def reset(self):
        """Reset track manager"""
        self.tracker.reset()
        self.tracks.clear()
        self.track_history.clear()
        self.current_frame_id = 0
        self.total_tracks_created = 0
        self.total_tracks_lost = 0