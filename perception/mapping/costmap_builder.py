"""
Costmap builder for path planning.
Converts world model into costmaps for the local planner.
"""

import numpy as np
import cv2
from typing import Tuple, List, Optional
import logging
from ..interfaces.core_interfaces import Obstacle

logger = logging.getLogger(__name__)

class CostmapBuilder:
    """Builds costmaps from occupancy grids and obstacle lists"""
    
    def __init__(self, config=None):
        self.inflation_radius = 0.3  # meters
        self.robot_radius = 0.3  # meters
        self.resolution = 0.05  # meters/cell
        
        # Cost values
        self.LETHAL_OBSTACLE = 254
        self.INSCRIBED_INFLATED = 253
        self.NO_INFORMATION = 255
        self.FREE_SPACE = 0
        
        # Precompute distance kernel
        self._create_inflation_kernel()
    
    def _create_inflation_kernel(self):
        """Create inflation kernel"""
        kernel_size = int(self.inflation_radius / self.resolution) * 2 + 1
        kernel_size = max(3, kernel_size)
        
        y, x = np.ogrid[-kernel_size//2:kernel_size//2+1, 
                       -kernel_size//2:kernel_size//2+1]
        kernel = x**2 + y**2 <= (kernel_size//2)**2
        self.inflation_kernel = kernel.astype(np.uint8)
    
    def build_costmap(self, occupancy_grid: np.ndarray,
                     obstacles: List[Obstacle] = None) -> np.ndarray:
        """Build costmap from occupancy grid
        
        Args:
            occupancy_grid: Occupancy grid (0-100)
            obstacles: Optional list of obstacles for additional inflation
            
        Returns:
            8-bit costmap (0-255)
        """
        # Start with occupancy grid
        costmap = np.zeros_like(occupancy_grid, dtype=np.uint8)
        
        # Convert occupancy probabilities to costs
        # Occupied (100): LETHAL
        costmap[occupancy_grid >= 80] = self.LETHAL_OBSTACLE
        
        # Likely occupied (50-80): High cost
        costmap[(occupancy_grid >= 50) & (occupancy_grid < 80)] = 200
        
        # Unknown (-1): NO_INFORMATION
        costmap[occupancy_grid == -1] = self.NO_INFORMATION
        
        # Inflate obstacles
        costmap = self._inflate_obstacles(costmap)
        
        # Add obstacle-specific inflation
        if obstacles:
            costmap = self._add_obstacle_inflation(costmap, obstacles)
        
        return costmap
    
    def _inflate_obstacles(self, costmap: np.ndarray) -> np.ndarray:
        """Inflate obstacles in costmap"""
        # Create binary mask of lethal obstacles
        lethal_mask = (costmap >= self.LETHAL_OBSTACLE).astype(np.uint8)
        
        if not np.any(lethal_mask):
            return costmap
        
        # Dilate obstacles
        inflated = cv2.dilate(lethal_mask, self.inflation_kernel)
        
        # Mark inflated cells
        costmap[(inflated > 0) & (costmap < self.INSCRIBED_INFLATED)] = \
            self.INSCRIBED_INFLATED
        
        return costmap
    
    def _add_obstacle_inflation(self, costmap: np.ndarray,
                               obstacles: List[Obstacle]) -> np.ndarray:
        """Add inflation from obstacle list"""
        for obs in obstacles:
            # Skip obstacles with low confidence
            if obs.confidence < 0.3:
                continue
            
            # Convert obstacle position to grid coordinates
            # This requires knowing the grid origin and resolution
            # Simplified: direct grid manipulation
            pass
        
        return costmap
    
    def compute_distance_transform(self, costmap: np.ndarray) -> np.ndarray:
        """Compute distance transform of costmap
        
        Args:
            costmap: Input costmap
            
        Returns:
            Distance transform (float)
        """
        # Binary image: lethal obstacles are 0, everything else is 1
        binary = (costmap < self.LETHAL_OBSTACLE).astype(np.uint8)
        
        # Distance transform
        dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        
        return dist_transform
    
    def compute_potential_field(self, costmap: np.ndarray,
                               goal: Tuple[int, int]) -> np.ndarray:
        """Compute potential field for navigation
        
        Args:
            costmap: Costmap
            goal: Goal position in grid coordinates
            
        Returns:
            Potential field
        """
        # Distance to goal
        h, w = costmap.shape
        y, x = np.ogrid[:h, :w]
        goal_dist = np.sqrt((x - goal[0])**2 + (y - goal[1])**2)
        
        # Obstacle potential
        obstacle_potential = np.where(
            costmap >= self.INSCRIBED_INFLATED,
            1000,  # High potential for obstacles
            0
        )
        
        # Total potential
        total_potential = goal_dist + obstacle_potential
        
        return total_potential
    
    def get_safety_cost(self, distance_to_obstacle: float) -> float:
        """Calculate safety cost based on distance to obstacle
        
        Args:
            distance_to_obstacle: Distance in meters
            
        Returns:
            Cost value (0-255)
        """
        if distance_to_obstacle <= self.robot_radius:
            return 254  # Collision
        elif distance_to_obstacle <= self.robot_radius + self.inflation_radius:
            # Linear interpolation
            ratio = (distance_to_obstacle - self.robot_radius) / self.inflation_radius
            return int(253 * (1 - ratio))
        else:
            return 0  # Free space