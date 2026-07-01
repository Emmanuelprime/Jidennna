"""
Abstract base class for object trackers.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from ..interfaces.core_interfaces import Detection, TrackedObject

class ObjectTrackerInterface(ABC):
    """Abstract interface for multi-object trackers"""
    
    @abstractmethod
    def initialize(self, config: Dict) -> bool:
        """Initialize tracker with configuration
        Args:
            config: Dictionary with tracker configuration
        Returns:
            bool: True if initialization successful
        """
        pass
    
    @abstractmethod
    def update(self, detections: List[Detection], 
               timestamp: float) -> List[TrackedObject]:
        """Update tracks with new detections
        Args:
            detections: List of new detections
            timestamp: Current timestamp
        Returns:
            List of tracked objects
        """
        pass
    
    @abstractmethod
    def get_active_tracks(self) -> List[TrackedObject]:
        """Get all currently active tracks
        Returns:
            List of active tracked objects
        """
        pass
    
    @abstractmethod
    def get_track_by_id(self, track_id: int) -> TrackedObject:
        """Get specific track by ID
        Args:
            track_id: Track identifier
        Returns:
            TrackedObject if found, None otherwise
        """
        pass
    
    @abstractmethod
    def remove_track(self, track_id: int) -> bool:
        """Remove a specific track
        Args:
            track_id: Track identifier
        Returns:
            bool: True if track was removed
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset all tracks"""
        pass
    
    @abstractmethod
    def get_track_count(self) -> int:
        """Get number of active tracks
        Returns:
            int: Number of active tracks
        """
        pass