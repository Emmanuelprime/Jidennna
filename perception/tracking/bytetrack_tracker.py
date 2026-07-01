"""
ByteTrack multi-object tracker implementation.
Simple, fast, and effective tracking by associating every detection box.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import time
from scipy.optimize import linear_sum_assignment
from .object_tracker import ObjectTrackerInterface
from .kalman_filter import KalmanFilterTracker
from ..interfaces.core_interfaces import Detection, TrackedObject

class ByteTrackTracker(ObjectTrackerInterface):
    """ByteTrack: Multi-object tracking by associating every detection box"""
    
    def __init__(self):
        self.tracks: List[KalmanFilterTracker] = []
        self.track_id_counter = 0
        self.max_age = 30
        self.min_hits = 3
        self.iou_threshold = 0.3
        self.det_thresh_high = 0.5
        self.det_thresh_low = 0.1
        self.track_buffer = 30
        
        # Track mapping
        self.track_map: Dict[int, KalmanFilterTracker] = {}
        self.lost_tracks: List[KalmanFilterTracker] = []
        
    def initialize(self, config: Dict) -> bool:
        """Initialize ByteTrack parameters"""
        self.max_age = config.get('max_age', 30)
        self.min_hits = config.get('min_hits', 3)
        self.iou_threshold = config.get('iou_threshold', 0.3)
        self.det_thresh_high = config.get('det_thresh_high', 0.5)
        self.det_thresh_low = config.get('det_thresh_low', 0.1)
        return True
    
    def update(self, detections: List[Detection], 
               timestamp: float) -> List[TrackedObject]:
        """Update tracker with new detections"""
        
        # Split detections by confidence
        dets_high = [d for d in detections if d.confidence >= self.det_thresh_high]
        dets_low = [d for d in detections if self.det_thresh_low <= d.confidence < self.det_thresh_high]
        
        # Get predictions from existing tracks
        for track in self.tracks:
            track.predict()
        
        # First association: high score detections with all tracks
        matched, unmatched_tracks, unmatched_dets = self._associate(
            self.tracks, dets_high, self.iou_threshold
        )
        
        # Update matched tracks
        for track_idx, det_idx in matched:
            track = self.tracks[track_idx]
            det = dets_high[det_idx]
            
            # Get box center as measurement
            x1, y1, x2, y2 = det.bbox
            measurement = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
            
            track.update(measurement)
            
            # Update track map
            if track.track_id not in self.track_map:
                self.track_map[track.track_id] = track
        
        # Second association: low score detections with remaining tracks
        if len(unmatched_tracks) > 0 and len(dets_low) > 0:
            matched_low, unmatched_tracks, _ = self._associate(
                [self.tracks[i] for i in unmatched_tracks],
                dets_low,
                0.5  # Higher IOU threshold for low score detections
            )
            
            for track_idx, det_idx in matched_low:
                track_idx_global = unmatched_tracks[track_idx]
                track = self.tracks[track_idx_global]
                det = dets_low[det_idx]
                
                x1, y1, x2, y2 = det.bbox
                measurement = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
                track.update(measurement)
        
        # Create new tracks for unmatched high score detections
        for det_idx in unmatched_dets:
            det = dets_high[det_idx]
            x1, y1, x2, y2 = det.bbox
            
            new_track = KalmanFilterTracker()
            position = ((x1 + x2) / 2, (y1 + y2) / 2)
            new_track.initialize(position)
            new_track.track_id = self.track_id_counter
            new_track.class_name = det.class_name
            new_track.confidence = det.confidence
            
            self.track_id_counter += 1
            self.tracks.append(new_track)
            self.track_map[new_track.track_id] = new_track
        
        # Handle lost tracks
        self._remove_stale_tracks()
        
        # Return active tracks
        return self.get_active_tracks()
    
    def _associate(self, tracks: List[KalmanFilterTracker],
                   detections: List[Detection],
                   iou_threshold: float) -> Tuple[List, List, List]:
        """Associate tracks with detections using IOU"""
        if len(tracks) == 0 or len(detections) == 0:
            return [], list(range(len(tracks))), list(range(len(detections)))
        
        # Calculate IOU matrix
        iou_matrix = np.zeros((len(tracks), len(detections)))
        
        for t, track in enumerate(tracks):
            track_state = track.get_state()
            track_pos = track_state['position']
            
            # Create pseudo bbox from track position
            track_bbox = self._position_to_bbox(track_pos)
            
            for d, det in enumerate(detections):
                iou_matrix[t, d] = self._calculate_iou(track_bbox, det.bbox)
        
        # Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)
        
        matched = []
        unmatched_tracks = []
        unmatched_dets = []
        
        for t, track in enumerate(tracks):
            if t in row_ind:
                d = col_ind[list(row_ind).index(t)]
                if iou_matrix[t, d] >= iou_threshold:
                    matched.append((t, d))
                else:
                    unmatched_tracks.append(t)
            else:
                unmatched_tracks.append(t)
        
        for d in range(len(detections)):
            if d not in col_ind:
                unmatched_dets.append(d)
        
        return matched, unmatched_tracks, unmatched_dets
    
    def _calculate_iou(self, bbox1: Tuple, bbox2: Tuple) -> float:
        """Calculate IOU between two bounding boxes"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        union = area1 + area2 - inter_area
        
        return inter_area / union if union > 0 else 0
    
    def _position_to_bbox(self, position: Tuple[float, float],
                         size: int = 50) -> Tuple[int, int, int, int]:
        """Convert position to bounding box"""
        x, y = position
        return (int(x - size/2), int(y - size/2), 
                int(x + size/2), int(y + size/2))
    
    def _remove_stale_tracks(self):
        """Remove tracks that haven't been updated"""
        active_tracks = []
        
        for track in self.tracks:
            if not track.is_stale(self.max_age):
                active_tracks.append(track)
            else:
                # Move to lost tracks
                self.lost_tracks.append(track)
                if track.track_id in self.track_map:
                    del self.track_map[track.track_id]
        
        self.tracks = active_tracks
        
        # Limit lost tracks buffer
        if len(self.lost_tracks) > self.track_buffer:
            self.lost_tracks = self.lost_tracks[-self.track_buffer:]
    
    def get_active_tracks(self) -> List[TrackedObject]:
        """Get all active tracks as TrackedObject list"""
        active = []
        
        for track in self.tracks:
            if track.is_confident(self.min_hits):
                state = track.get_state()
                
                tracked_obj = TrackedObject(
                    track_id=track.track_id,
                    class_name=getattr(track, 'class_name', 'unknown'),
                    confidence=getattr(track, 'confidence', 0.5),
                    position=state['position'],
                    velocity=state['velocity'],
                    acceleration=state['acceleration'],
                    bbox=(0, 0, 0, 0),  # Would need to store original bbox
                    age=track.age,
                    time_since_update=track.time_since_update,
                    position_covariance=state['covariance']
                )
                active.append(tracked_obj)
        
        return active
    
    def get_track_by_id(self, track_id: int) -> Optional[TrackedObject]:
        """Get specific track by ID"""
        if track_id in self.track_map:
            track = self.track_map[track_id]
            state = track.get_state()
            
            return TrackedObject(
                track_id=track.track_id,
                class_name=getattr(track, 'class_name', 'unknown'),
                confidence=getattr(track, 'confidence', 0.5),
                position=state['position'],
                velocity=state['velocity'],
                acceleration=state['acceleration'],
                bbox=(0, 0, 0, 0),
                age=track.age,
                time_since_update=track.time_since_update,
                position_covariance=state['covariance']
            )
        return None
    
    def remove_track(self, track_id: int) -> bool:
        """Remove a specific track"""
        for i, track in enumerate(self.tracks):
            if track.track_id == track_id:
                self.tracks.pop(i)
                if track_id in self.track_map:
                    del self.track_map[track_id]
                return True
        return False
    
    def reset(self) -> None:
        """Reset all tracks"""
        self.tracks.clear()
        self.track_map.clear()
        self.lost_tracks.clear()
        self.track_id_counter = 0
    
    def get_track_count(self) -> int:
        """Get number of active tracks"""
        return len([t for t in self.tracks if t.is_confident(self.min_hits)])