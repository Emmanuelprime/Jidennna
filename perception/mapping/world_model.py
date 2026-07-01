"""
Unified world model combining occupancy grid, obstacles, and semantic information.
Provides a complete environmental representation for planners.
"""

import numpy as np
import time
import logging
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict

from .occupancy_grid import OccupancyGrid
from .obstacle_map import ObstacleMap
from ..interfaces.core_interfaces import (
    Obstacle, TrackedObject, RobotPose, WorldModelInterface
)

logger = logging.getLogger(__name__)

class WorldModel(WorldModelInterface):
    """Unified world model combining multiple information sources
    
    This is the central repository for all environmental understanding.
    It combines:
    - Occupancy grid for static environment
    - Dynamic obstacle tracking
    - Semantic scene understanding
    - Risk assessment
    - Free space analysis
    """
    
    def __init__(self, mapping_config=None):
        """
        Args:
            mapping_config: Mapping configuration object
        """
        # Configuration
        self.config = mapping_config
        self.grid_width = getattr(mapping_config, 'grid_width', 200)
        self.grid_height = getattr(mapping_config, 'grid_height', 200)
        self.resolution = getattr(mapping_config, 'resolution', 0.05)
        
        # Core components
        self.occupancy_grid = OccupancyGrid(
            width=self.grid_width,
            height=self.grid_height,
            resolution=self.resolution
        )
        
        self.obstacle_map = ObstacleMap(mapping_config)
        
        # State tracking
        self.current_time = 0.0
        self.robot_pose: Optional[RobotPose] = None
        self.robot_position = (0.0, 0.0)
        self.robot_velocity = (0.0, 0.0)
        
        # World state
        self.world_state: Dict[str, Any] = {
            'timestamp': 0.0,
            'robot_pose': None,
            'dynamic_objects': [],
            'static_objects': [],
            'objects_of_interest': [],
            'risk_zones': [],
            'free_space_polygons': [],
            'semantic_info': {}
        }
        
        # Update tracking
        self.update_count = 0
        self.last_update_time = 0.0
        self.update_interval = 1.0 / getattr(mapping_config, 'update_frequency', 10.0)
        
        # History for temporal analysis
        self.pose_history: List[Tuple[float, float, float]] = []
        self.max_pose_history = 100
        
        # Semantic mapping
        self.semantic_labels: Dict[Tuple[int, int], str] = {}
        
        # Risk assessment
        self.risk_map = np.zeros((self.grid_height, self.grid_width))
        self.risk_decay_rate = 0.9  # Decay factor per update
        
        logger.info(f"World model initialized: {self.grid_width}x{self.grid_height} "
                   f"grid @ {self.resolution}m resolution")
    
    def update(self, tracked_objects: List[TrackedObject],
               obstacles: List[Obstacle],
               timestamp: float,
               robot_pose: Optional[RobotPose] = None,
               sensor_data: Dict = None):
        """Update world model with new observations
        
        Args:
            tracked_objects: List of tracked dynamic objects
            obstacles: List of obstacles in robot frame
            timestamp: Current timestamp
            robot_pose: Current robot pose (optional)
            sensor_data: Additional sensor data (optional)
        """
        update_start = time.time()
        
        # Rate limiting
        if timestamp - self.last_update_time < self.update_interval * 0.5:
            return
        
        self.current_time = timestamp
        self.update_count += 1
        
        # Update robot state
        if robot_pose is not None:
            self.robot_pose = robot_pose
            self.robot_position = (robot_pose.x, robot_pose.y)
            self.pose_history.append((robot_pose.x, robot_pose.y, robot_pose.theta))
            if len(self.pose_history) > self.max_pose_history:
                self.pose_history.pop(0)
        
        # Update dynamic objects
        if tracked_objects:
            self._update_dynamic_objects(tracked_objects)
        
        # Update obstacle map
        if obstacles:
            self._update_obstacle_map(obstacles)
        
        # Update occupancy grid
        self._update_occupancy_grid()
        
        # Update risk assessment
        self._update_risk_assessment()
        
        # Update world state
        self._update_world_state()
        
        # Handle additional sensor data
        if sensor_data:
            self._process_sensor_data(sensor_data)
        
        self.last_update_time = timestamp
        
        update_time = (time.time() - update_start) * 1000
        if update_time > 10:  # Log if update takes more than 10ms
            logger.debug(f"World model update took {update_time:.1f}ms")
    
    def _update_dynamic_objects(self, tracked_objects: List[TrackedObject]):
        """Update dynamic object tracking"""
        # Update obstacle map with tracked objects
        self.obstacle_map.update_dynamic_obstacles(tracked_objects, self.current_time)
        
        # Store in world state
        self.world_state['dynamic_objects'] = [
            {
                'id': obj.track_id,
                'type': obj.class_name,
                'position': obj.position,
                'velocity': obj.velocity,
                'confidence': obj.confidence,
                'age': obj.age
            }
            for obj in tracked_objects
        ]
    
    def _update_obstacle_map(self, obstacles: List[Obstacle]):
        """Update static obstacle information"""
        for obs in obstacles:
            if obs.obstacle_type.startswith('static'):
                self.obstacle_map.add_static_obstacle(
                    position=obs.position,
                    radius=obs.radius,
                    obstacle_type=obs.obstacle_type
                )
    
    def _update_occupancy_grid(self):
        """Update occupancy grid with current obstacles"""
        # Get all obstacles
        all_obstacles = self.obstacle_map.get_all_obstacles()
        
        # Update occupancy grid
        self.occupancy_grid.update_occupancy(all_obstacles, self.robot_position)
        
        # Inflate obstacles for safety
        self.occupancy_grid.inflate_obstacles()
        
        # Raytrace free space from robot position
        if self.robot_position is not None:
            self.occupancy_grid.raytrace_free_space(
                self.robot_position,
                max_range=5.0  # 5 meter local map
            )
    
    def _update_risk_assessment(self):
        """Update risk assessment for the environment"""
        # Decay previous risk map
        self.risk_map *= self.risk_decay_rate
        
        # Add risk from dynamic objects
        for obs in self.obstacle_map.dynamic_obstacles.values():
            if obs.velocity and np.linalg.norm(obs.velocity) > 0.1:
                # Convert position to grid
                grid_x, grid_y = self.occupancy_grid.world_to_grid(
                    obs.position[0], obs.position[1]
                )
                
                # Add risk based on speed
                speed = np.linalg.norm(obs.velocity)
                risk_level = min(1.0, speed / 5.0)  # Scale: 5 m/s = max risk
                
                # Add risk in radius around object
                radius_cells = int(1.0 / self.resolution)  # 1 meter radius
                
                for dx in range(-radius_cells, radius_cells + 1):
                    for dy in range(-radius_cells, radius_cells + 1):
                        if dx**2 + dy**2 <= radius_cells**2:
                            nx, ny = grid_x + dx, grid_y + dy
                            if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height:
                                self.risk_map[ny, nx] = max(
                                    self.risk_map[ny, nx],
                                    risk_level
                                )
    
    def _update_world_state(self):
        """Update complete world state representation"""
        # Dynamic object info
        dynamic_objects = list(self.obstacle_map.dynamic_obstacles.values())
        static_objects = list(self.obstacle_map.static_obstacles.values())
        
        # Objects of interest (delivery-relevant)
        objects_of_interest = self._identify_objects_of_interest(
            dynamic_objects + static_objects
        )
        
        # Risk zones
        risk_zones = self._calculate_risk_zones()
        
        # Free space analysis
        free_space = self._analyze_free_space()
        
        # Update world state dictionary
        self.world_state = {
            'timestamp': self.current_time,
            'robot_pose': {
                'x': self.robot_position[0],
                'y': self.robot_position[1],
                'theta': self.pose_history[-1][2] if self.pose_history else 0.0
            } if self.robot_position else None,
            'dynamic_objects_count': len(dynamic_objects),
            'static_objects_count': len(static_objects),
            'objects_of_interest': objects_of_interest,
            'risk_zones': risk_zones,
            'free_space_polygons': free_space,
            'grid_info': {
                'occupied_cells': int(np.sum(self.occupancy_grid.grid == 100)),
                'free_cells': int(np.sum(self.occupancy_grid.grid == 0)),
                'unknown_cells': int(np.sum(self.occupancy_grid.grid == -1))
            },
            'update_count': self.update_count
        }
    
    def _identify_objects_of_interest(self, obstacles: List[Obstacle]) -> List[Dict]:
        """Identify delivery-relevant objects in the environment
        
        Args:
            obstacles: List of all obstacles
            
        Returns:
            List of objects of interest with metadata
        """
        objects_of_interest = []
        
        interest_classes = ['person', 'vehicle', 'delivery_box', 'door', 'mailbox']
        
        for obs in obstacles:
            # Check if object type is of interest
            is_interesting = any(
                interest_class in obs.obstacle_type.lower()
                for interest_class in interest_classes
            )
            
            if is_interesting or obs.confidence > 0.8:
                # Calculate distance to robot
                distance = np.sqrt(
                    (obs.position[0] - self.robot_position[0])**2 +
                    (obs.position[1] - self.robot_position[1])**2
                )
                
                # Calculate bearing relative to robot
                bearing = np.arctan2(
                    obs.position[1] - self.robot_position[1],
                    obs.position[0] - self.robot_position[0]
                )
                
                # Get robot heading
                robot_theta = self.pose_history[-1][2] if self.pose_history else 0.0
                relative_bearing = bearing - robot_theta
                
                # Normalize to [-pi, pi]
                while relative_bearing > np.pi:
                    relative_bearing -= 2 * np.pi
                while relative_bearing < -np.pi:
                    relative_bearing += 2 * np.pi
                
                obj_info = {
                    'id': obs.id,
                    'type': obs.obstacle_type,
                    'position': obs.position,
                    'distance': distance,
                    'bearing': relative_bearing,
                    'velocity': obs.velocity,
                    'confidence': obs.confidence,
                    'is_moving': obs.velocity is not None and np.linalg.norm(obs.velocity) > 0.1
                }
                
                # Add predicted path for moving objects
                if obj_info['is_moving']:
                    obj_info['predicted_path'] = self._predict_object_path(obs, horizon=3.0)
                
                objects_of_interest.append(obj_info)
        
        # Sort by distance
        objects_of_interest.sort(key=lambda x: x['distance'])
        
        return objects_of_interest
    
    def _predict_object_path(self, obstacle: Obstacle, 
                            horizon: float = 3.0,
                            steps: int = 10) -> List[Tuple[float, float]]:
        """Predict future path of a moving object
        
        Args:
            obstacle: Moving obstacle
            horizon: Prediction horizon in seconds
            steps: Number of prediction steps
            
        Returns:
            List of predicted positions
        """
        if obstacle.velocity is None:
            return [obstacle.position]
        
        dt = horizon / steps
        path = []
        
        pos = np.array(obstacle.position, dtype=float)
        vel = np.array(obstacle.velocity, dtype=float)
        
        for i in range(steps):
            pos = pos + vel * dt
            path.append(tuple(pos))
        
        return path
    
    def _calculate_risk_zones(self) -> List[Dict]:
        """Calculate high-risk areas in the environment
        
        Returns:
            List of risk zones with metadata
        """
        risk_zones = []
        
        # Find high-risk areas from risk map
        high_risk_mask = self.risk_map > 0.5
        
        if np.any(high_risk_mask):
            # Find connected components (simple implementation)
            from scipy import ndimage
            
            labeled, num_features = ndimage.label(high_risk_mask)
            
            for i in range(1, num_features + 1):
                region = labeled == i
                if np.sum(region) > 5:  # Minimum size
                    # Get region center in world coordinates
                    y_indices, x_indices = np.where(region)
                    center_x = np.mean(x_indices)
                    center_y = np.mean(y_indices)
                    
                    world_x, world_y = self.occupancy_grid.grid_to_world(
                        int(center_x), int(center_y)
                    )
                    
                    risk_zones.append({
                        'position': (world_x, world_y),
                        'risk_level': float(np.mean(self.risk_map[region])),
                        'area_cells': int(np.sum(region)),
                        'area_meters': float(np.sum(region) * self.resolution**2)
                    })
        
        return risk_zones
    
    def _analyze_free_space(self) -> List[List[Tuple[float, float]]]:
        """Analyze free space for navigation
        
        Returns:
            List of polygons representing free space
        """
        # Get free space from occupancy grid
        free_mask = self.occupancy_grid.grid <= 0  # Free or unknown
        
        # Find connected components of free space
        from scipy import ndimage
        
        labeled, num_features = ndimage.label(free_mask)
        
        free_polygons = []
        
        for i in range(1, min(num_features + 1, 10)):  # Limit to top 10 regions
            region = labeled == i
            
            # Find contour of region
            import cv2
            contours, _ = cv2.findContours(
                region.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            for contour in contours:
                if len(contour) > 3:
                    # Convert to world coordinates
                    polygon = [
                        self.occupancy_grid.grid_to_world(int(pt[0][0]), int(pt[0][1]))
                        for pt in contour
                    ]
                    free_polygons.append(polygon)
        
        return free_polygons
    
    def _process_sensor_data(self, sensor_data: Dict):
        """Process additional sensor data
        
        Args:
            sensor_data: Dictionary of sensor readings
        """
        # Process LiDAR data if available
        if 'lidar' in sensor_data:
            self._process_lidar(sensor_data['lidar'])
        
        # Process ultrasonic data if available
        if 'ultrasonic' in sensor_data:
            self._process_ultrasonic(sensor_data['ultrasonic'])
    
    def _process_lidar(self, lidar_data: Dict):
        """Process LiDAR point cloud data"""
        # Placeholder for LiDAR integration
        pass
    
    def _process_ultrasonic(self, ultrasonic_data: Dict):
        """Process ultrasonic sensor data"""
        # Placeholder for ultrasonic integration
        pass
    
    def get_obstacles(self) -> List[Obstacle]:
        """Get all current obstacles in robot frame
        
        Returns:
            List of Obstacle objects
        """
        return self.obstacle_map.get_all_obstacles()
    
    def get_dynamic_obstacles(self) -> List[Obstacle]:
        """Get only dynamic obstacles
        
        Returns:
            List of dynamic Obstacle objects
        """
        return list(self.obstacle_map.dynamic_obstacles.values())
    
    def get_static_obstacles(self) -> List[Obstacle]:
        """Get only static obstacles
        
        Returns:
            List of static Obstacle objects
        """
        return list(self.obstacle_map.static_obstacles.values())
    
    def get_occupancy_grid(self) -> np.ndarray:
        """Get current occupancy grid
        
        Returns:
            2D numpy array (0-100 occupancy probability, -1 unknown)
        """
        return self.occupancy_grid.get_grid()
    
    def get_risk_map(self) -> np.ndarray:
        """Get current risk map
        
        Returns:
            2D numpy array (0-1 risk levels)
        """
        return self.risk_map.copy()
    
    def get_costmap(self, robot_radius: float = 0.3) -> np.ndarray:
        """Get costmap for path planning
        
        Args:
            robot_radius: Robot radius in meters
            
        Returns:
            2D numpy array (0-255 cost values)
        """
        from .costmap_builder import CostmapBuilder
        
        costmap_builder = CostmapBuilder()
        costmap_builder.robot_radius = robot_radius
        costmap_builder.inflation_radius = getattr(
            self.config, 'obstacle_inflation_radius', 0.3
        )
        
        return costmap_builder.build_costmap(
            self.occupancy_grid.get_grid(),
            self.obstacle_map.get_all_obstacles()
        )
    
    def get_world_state(self) -> Dict[str, Any]:
        """Get complete world state
        
        Returns:
            Dictionary with world state information
        """
        return self.world_state
    
    def get_obstacles_near_point(self, point: Tuple[float, float],
                                 radius: float = 2.0) -> List[Obstacle]:
        """Get obstacles within radius of a point
        
        Args:
            point: (x, y) position
            radius: Search radius in meters
            
        Returns:
            List of nearby obstacles
        """
        return self.obstacle_map.get_obstacles_in_radius(point, radius)
    
    def get_nearest_obstacle(self, point: Tuple[float, float] = None) -> Optional[Obstacle]:
        """Get nearest obstacle to a point
        
        Args:
            point: Reference point (uses robot position if None)
            
        Returns:
            Nearest obstacle or None
        """
        if point is None:
            point = self.robot_position
        
        return self.obstacle_map.get_nearest_obstacle(point)
    
    def is_collision_free(self, point: Tuple[float, float], 
                         radius: float = 0.3) -> bool:
        """Check if a point is collision-free
        
        Args:
            point: (x, y) position to check
            radius: Safety radius
            
        Returns:
            True if collision-free
        """
        # Convert to grid coordinates
        grid_x, grid_y = self.occupancy_grid.world_to_grid(point[0], point[1])
        
        # Check radius in cells
        radius_cells = int(radius / self.resolution)
        
        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                if dx**2 + dy**2 <= radius_cells**2:
                    nx, ny = grid_x + dx, grid_y + dy
                    if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height:
                        if self.occupancy_grid.grid[ny, nx] >= 50:  # Likely occupied
                            return False
        
        return True
    
    def get_free_path_points(self, start: Tuple[float, float],
                            end: Tuple[float, float],
                            num_points: int = 10) -> List[bool]:
        """Check if path between two points is collision-free
        
        Args:
            start: Start position (x, y)
            end: End position (x, y)
            num_points: Number of points to check along path
            
        Returns:
            List of boolean values indicating free space at each point
        """
        points = []
        
        for i in range(num_points):
            t = i / (num_points - 1)
            x = start[0] + t * (end[0] - start[0])
            y = start[1] + t * (end[1] - start[1])
            points.append((x, y))
        
        return [self.is_collision_free(p) for p in points]
    
    def predict_future_state(self, time_horizon: float = 1.0) -> Dict:
        """Predict future state of the world
        
        Args:
            time_horizon: Prediction horizon in seconds
            
        Returns:
            Dictionary with predicted state
        """
        predicted_obstacles = []
        
        for obs in self.obstacle_map.dynamic_obstacles.values():
            if obs.velocity and np.linalg.norm(obs.velocity) > 0.1:
                # Predict future position
                future_pos = (
                    obs.position[0] + obs.velocity[0] * time_horizon,
                    obs.position[1] + obs.velocity[1] * time_horizon
                )
                
                predicted_obstacles.append({
                    'id': obs.id,
                    'type': obs.obstacle_type,
                    'current_position': obs.position,
                    'predicted_position': future_pos,
                    'velocity': obs.velocity
                })
        
        return {
            'timestamp': self.current_time + time_horizon,
            'predicted_obstacles': predicted_obstacles,
            'num_dynamic_objects': len(predicted_obstacles)
        }
    
    def add_semantic_label(self, position: Tuple[float, float], label: str):
        """Add semantic label to a position
        
        Args:
            position: (x, y) world position
            label: Semantic label (e.g., 'sidewalk', 'road', 'door')
        """
        grid_x, grid_y = self.occupancy_grid.world_to_grid(
            position[0], position[1]
        )
        self.semantic_labels[(grid_x, grid_y)] = label
    
    def get_semantic_info(self, position: Tuple[float, float]) -> Optional[str]:
        """Get semantic label at a position
        
        Args:
            position: (x, y) world position
            
        Returns:
            Semantic label or None
        """
        grid_x, grid_y = self.occupancy_grid.world_to_grid(
            position[0], position[1]
        )
        return self.semantic_labels.get((grid_x, grid_y))
    
    def clear(self):
        """Clear the world model"""
        self.occupancy_grid = OccupancyGrid(
            width=self.grid_width,
            height=self.grid_height,
            resolution=self.resolution
        )
        self.obstacle_map.clear()
        self.risk_map.fill(0)
        self.semantic_labels.clear()
        self.world_state['objects_of_interest'] = []
        self.world_state['risk_zones'] = []
        self.world_state['free_space_polygons'] = []
        logger.info("World model cleared")
    
    def get_statistics(self) -> Dict:
        """Get world model statistics
        
        Returns:
            Dictionary with statistics
        """
        dynamic_count, static_count = self.obstacle_map.get_obstacle_count()
        
        return {
            'update_count': self.update_count,
            'dynamic_obstacles': dynamic_count,
            'static_obstacles': static_count,
            'objects_of_interest': len(self.world_state.get('objects_of_interest', [])),
            'risk_zones': len(self.world_state.get('risk_zones', [])),
            'grid_occupied_percent': float(
                np.sum(self.occupancy_grid.grid == 100) / 
                (self.grid_width * self.grid_height) * 100
            ),
            'semantic_labels': len(self.semantic_labels)
        }
    
    def visualize(self) -> Dict[str, np.ndarray]:
        """Create visualization images of world model
        
        Returns:
            Dictionary with visualization images
        """
        visualizations = {}
        
        # Occupancy grid visualization
        grid_vis = np.zeros((self.grid_height, self.grid_width, 3), dtype=np.uint8)
        grid_vis[self.occupancy_grid.grid == 0] = [255, 255, 255]    # White for free
        grid_vis[self.occupancy_grid.grid == 100] = [0, 0, 0]       # Black for occupied
        grid_vis[self.occupancy_grid.grid == -1] = [128, 128, 128]  # Gray for unknown
        grid_vis[(self.occupancy_grid.grid > 0) & 
                (self.occupancy_grid.grid < 100)] = [0, 0, 255]     # Red for inflated
        
        # Draw robot position
        if self.robot_position:
            grid_x, grid_y = self.occupancy_grid.world_to_grid(
                self.robot_position[0], self.robot_position[1]
            )
            cv2 = __import__('cv2')
            cv2.circle(grid_vis, (grid_x, grid_y), 5, (0, 255, 0), -1)
        
        visualizations['occupancy_grid'] = grid_vis
        
        # Risk map visualization
        risk_vis = (self.risk_map * 255).astype(np.uint8)
        risk_vis = cv2.applyColorMap(risk_vis, cv2.COLORMAP_HOT)
        visualizations['risk_map'] = risk_vis
        
        return visualizations