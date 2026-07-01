"""
DeepSORT tracker implementation with appearance features.
Uses deep learning features for better association.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import time
from scipy.optimize import linear_sum_assignment
from .object_tracker import ObjectTrackerInterface
from .kalman_filter import KalmanFilterTracker
from ..interfaces.core_interfaces import Detection, TrackedObject

class DeepSORTTracker(ObjectTrackerInterface):
    """DeepSORT: Simple online and realtime tracking with deep association metrics"""
    
    def __init__(self):
        self.tracks: List[KalmanFilterTracker] = []
        self.track_id_counter = 0
        self.max_age = 30
        self.min_hits = 3
        self.iou_threshold = 0.3
        
        # Feature extraction (placeholder - would use actual CNN)
        self.feature_extractor = None
        self.feature_dim = 512
        self.max_feature_distance = 0.2
        
        # Track features
        self.track_features: Dict[int, List[np.ndarray]] = {}
        self.max_features_per_track = 100
        
        # Matching weights
        self.lambda_iou = 0.5  # Weight for IOU distance
        self.lambda_feature = 0.5  # Weight for feature distance
        
    def initialize(self, config: Dict) -> bool:
        """Initialize DeepSORT parameters"""
        self.max_age = config.get('max_age', 30)
        self.min_hits = config.get('min_hits', 3)
        self.iou_threshold = config.get('iou_threshold', 0.3)
        self.max_feature_distance = config.get('max_feature_distance', 0.2)
        
        # Initialize feature extractor if available
        if 'feature_extractor' in config:
            self.feature_extractor = config['feature_extractor']
        
        return True
    
    def update(self, detections: List[Detection], 
               timestamp: float) -> List[TrackedObject]:
        """Update tracker with new detections"""
        
        # Predict all tracks
        for track in self.tracks:
            track.predict()
        
        # Extract features from detections (if feature extractor available)
        detection_features = self._extract_features(detections)
        
        # Perform matching
        if len(self.tracks) > 0 and len(detections) > 0:
            matches, unmatched_tracks, unmatched_dets = self._match_detections(
                self.tracks, detections, detection_features
            )
        else:
            matches = []
            unmatched_tracks = list(range(len(self.tracks)))
            unmatched_dets = list(range(len(detections)))
        
        # Update matched tracks
        for track_idx, det_idx in matches:
            track = self.tracks[track_idx]
            det = detections[det_idx]
            
            # Update Kalman filter
            x1, y1, x2, y2 = det.bbox
            measurement = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
            track.update(measurement)
            
            # Update features
            if detection_features and det_idx < len(detection_features):
                self._update_track_features(track.track_id, detection_features[det_idx])
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            self._create_new_track(det, detection_features[det_idx] if detection_features else None)
        
        # Remove stale tracks
        self._remove_stale_tracks()
        
        return self.get_active_tracks()
    
    def _extract_features(self, detections: List[Detection]) -> Optional[List[np.ndarray]]:
        """Extract appearance features from detections"""
        if self.feature_extractor is None:
            return None
        
        features = []
        for det in detections:
            if det.feature_vector is not None:
                features.append(det.feature_vector)
            else:
                # Extract from image crop (placeholder)
                features.append(np.random.randn(self.feature_dim))
        
        return features
    
    def _match_detections(self, tracks: List[KalmanFilterTracker],
                         detections: List[Detection],
                         features: Optional[List[np.ndarray]] = None) -> Tuple:
        """Match tracks with detections using combined distance"""
        
        # Calculate cost matrix
        cost_matrix = np.zeros((len(tracks), len(detections)))
        
        for t, track in enumerate(tracks):
            track_state = track.get_state()
            track_pos = track_state['position']
            track_bbox = self._position_to_bbox(track_pos)
            
            for d, det in enumerate(detections):
                # IOU distance
                iou_dist = 1 - self._calculate_iou(track_bbox, det.bbox)
                
                # Feature distance
                if features and track.track_id in self.track_features:
                    track_feat = np.mean(self.track_features[track.track_id], axis=0)
                    det_feat = features[d]
                    feat_dist = 1 - np.dot(track_feat, det_feat) / (
                        np.linalg.norm(track_feat) * np.linalg.norm(det_feat) + 1e-6
                    )
                else:
                    feat_dist = 0.5  # Default feature distance
                
                # Combined cost
                cost_matrix[t, d] = (self.lambda_iou * iou_dist + 
                                    self.lambda_feature * feat_dist)
        
        # Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        matches = []
        unmatched_tracks = []
        unmatched_dets = []
        
        for t in range(len(tracks)):
            if t in row_ind:
                d = col_ind[list(row_ind).index(t)]
                if cost_matrix[t, d] < 1.0 - self.iou_threshold:
                    matches.append((t, d))
                else:
                    unmatched_tracks.append(t)
            else:
                unmatched_tracks.append(t)
        
        for d in range(len(detections)):
            if d not in col_ind:
                unmatched_dets.append(d)
        
        return matches, unmatched_tracks, unmatched_dets
    
    def _create_new_track(self, detection: Detection, 
                         feature: Optional[np.ndarray] = None):
        """Create a new track from detection"""
        x1, y1, x2, y2 = detection.bbox
        position = ((x1 + x2) / 2, (y1 + y2) / 2)
        
        new_track = KalmanFilterTracker()
        new_track.initialize(position)
        new_track.track_id = self.track_id_counter
        new_track.class_name = detection.class_name
        new_track.confidence = detection.confidence
        
        self.track_id_counter += 1
        self.tracks.append(new_track)
        
        if feature is not None:
            self.track_features[new_track.track_id] = [feature]
    
    def _update_track_features(self, track_id: int, feature: np.ndarray):
        """Update feature bank for track"""
        if track_id not in self.track_features:
            self.track_features[track_id] = []
        
        self.track_features[track_id].append(feature)
        
        # Limit feature bank size
        if len(self.track_features[track_id]) > self.max_features_per_track:
            self.track_features[track_id] = self.track_features[track_id][-self.max_features_per_track:]
    
    def _calculate_iou(self, bbox1: Tuple, bbox2: Tuple) -> float:
        """Calculate IOU between two boxes"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        return inter / (area1 + area2 - inter + 1e-6)
    
    def _position_to_bbox(self, position: Tuple[float, float]) -> Tuple:
        """Convert position to bounding box"""
        x, y = position
        w, h = 50, 50
        return (x - w/2, y - h/2, x + w/2, y + h/2)
    
    def _remove_stale_tracks(self):
        """Remove old tracks"""
        self.tracks = [t for t in self.tracks if not t.is_stale(self.max_age)]
        
        # Clean up features for removed tracks
        active_ids = {t.track_id for t in self.tracks}
        for track_id in list(self.track_features.keys()):
            if track_id not in active_ids:
                del self.track_features[track_id]
    
    def get_active_tracks(self) -> List[TrackedObject]:
        """Get active tracks"""
        active = []
        for track in self.tracks:
            if track.is_confident(self.min_hits):
                state = track.get_state()
                active.append(TrackedObject(
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
                ))
        return active
    
    def get_track_by_id(self, track_id: int) -> Optional[TrackedObject]:
        for track in self.tracks:
            if track.track_id == track_id:
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
        for i, track in enumerate(self.tracks):
            if track.track_id == track_id:
                self.tracks.pop(i)
                if track_id in self.track_features:
                    del self.track_features[track_id]
                return True
        return False
    
    def reset(self) -> None:
        self.tracks.clear()
        self.track_features.clear()
        self.track_id_counter = 0
    
    def get_track_count(self) -> int:
        return len([t for t in self.tracks if t.is_confident(self.min_hits)])