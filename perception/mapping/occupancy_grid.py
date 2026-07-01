# mapping/occupancy_grid.py

import numpy as np
from typing import Tuple, List, Optional
import cv2

class OccupancyGrid:
    """Occupancy grid map implementation"""
    
    def __init__(self, width: int = 200, height: int = 200, 
                 resolution: float = 0.05):
        """
        Args:
            width: Grid width in cells
            height: Grid height in cells
            resolution: Meters per cell
        """
        self.width = width
        self.height = height
        self.resolution = resolution
        
        # Initialize grid (0-100 for occupancy probability)
        # -1: unknown, 0: free, 100: occupied
        self.grid = np.full((height, width), -1, dtype=np.int8)
        
        # Log odds representation for occupancy
        self.log_odds = np.zeros((height, width), dtype=np.float32)
        
        # Grid origin (in meters, robot-centric)
        self.origin_x = -width * resolution / 2
        self.origin_y = -height * resolution / 2
        
        # Probability thresholds
        self.occupied_threshold = 0.65
        self.free_threshold = 0.2
        
        # Inflation kernel
        self.inflation_radius = 0.3  # meters
        self._create_inflation_kernel()
    
    def _create_inflation_kernel(self):
        """Create inflation kernel for obstacles"""
        kernel_size = int(self.inflation_radius / self.resolution)
        kernel_size = max(3, kernel_size * 2 + 1)  # Ensure odd
        
        y, x = np.ogrid[-kernel_size//2:kernel_size//2+1, 
                       -kernel_size//2:kernel_size//2+1]
        self.inflation_kernel = (x**2 + y**2 <= (kernel_size//2)**2).astype(np.uint8)
    
    def world_to_grid(self, world_x: float, world_y: float) -> Tuple[int, int]:
        """Convert world coordinates to grid coordinates"""
        grid_x = int((world_x - self.origin_x) / self.resolution)
        grid_y = int((world_y - self.origin_y) / self.resolution)
        
        # Clamp to grid bounds
        grid_x = np.clip(grid_x, 0, self.width - 1)
        grid_y = np.clip(grid_y, 0, self.height - 1)
        
        return grid_x, grid_y
    
    def grid_to_world(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        """Convert grid coordinates to world coordinates"""
        world_x = grid_x * self.resolution + self.origin_x
        world_y = grid_y * self.resolution + self.origin_y
        return world_x, world_y
    
    def update_occupancy(self, obstacles: List[Obstacle], 
                        robot_position: Tuple[float, float]):
        """Update occupancy grid with new obstacle observations"""
        
        # Clear dynamic obstacles from previous frame
        # (Keep static obstacles)
        
        for obstacle in obstacles:
            # Convert to grid coordinates
            obs_x, obs_y = self.world_to_grid(
                obstacle.position[0], 
                obstacle.position[1]
            )
            
            # Calculate obstacle radius in cells
            radius_cells = int(obstacle.radius / self.resolution)
            
            # Update occupancy in circular region
            for dx in range(-radius_cells, radius_cells + 1):
                for dy in range(-radius_cells, radius_cells + 1):
                    if dx**2 + dy**2 <= radius_cells**2:
                        grid_x = obs_x + dx
                        grid_y = obs_y + dy
                        
                        if 0 <= grid_x < self.width and \
                           0 <= grid_y < self.height:
                            # Update log odds (occupied)
                            self.log_odds[grid_y, grid_x] += 1.0
                            
                            # Convert to probability
                            prob = 1.0 - 1.0 / (1.0 + np.exp(
                                self.log_odds[grid_y, grid_x]
                            ))
                            
                            if prob > self.occupied_threshold:
                                self.grid[grid_y, grid_x] = 100
                            elif prob < self.free_threshold:
                                self.grid[grid_y, grid_x] = 0
                            else:
                                self.grid[grid_y, grid_x] = 50
    
    def raytrace_free_space(self, robot_position: Tuple[float, float],
                           max_range: float = 5.0):
        """Mark free space using ray tracing"""
        # Convert robot position to grid
        robot_grid_x, robot_grid_y = self.world_to_grid(
            robot_position[0], robot_position[1]
        )
        
        # Cast rays in multiple directions
        num_rays = 180  # One ray per 2 degrees
        for angle in np.linspace(0, 2*np.pi, num_rays):
            for r in np.arange(0, max_range, self.resolution):
                check_x = robot_position[0] + r * np.cos(angle)
                check_y = robot_position[1] + r * np.sin(angle)
                
                grid_x, grid_y = self.world_to_grid(check_x, check_y)
                
                # Check if we hit an obstacle
                if self.grid[grid_y, grid_x] == 100:
                    break
                
                # Mark as free
                self.log_odds[grid_y, grid_x] -= 0.5
                
                prob = 1.0 - 1.0 / (1.0 + np.exp(
                    self.log_odds[grid_y, grid_x]
                ))
                
                if prob < self.free_threshold:
                    self.grid[grid_y, grid_x] = 0
    
    def inflate_obstacles(self):
        """Inflate obstacles for safety margin"""
        # Create binary obstacle map
        obstacle_mask = (self.grid >= 50).astype(np.uint8)
        
        # Apply dilation
        inflated = cv2.dilate(obstacle_mask, self.inflation_kernel)
        
        # Update grid
        self.grid[inflated > 0] = np.maximum(self.grid[inflated > 0], 80)
    
    def get_grid(self) -> np.ndarray:
        """Get current occupancy grid"""
        return self.grid.copy()
    
    def get_costmap(self, robot_radius: float = 0.3) -> np.ndarray:
        """Generate costmap for path planning"""
        costmap = np.zeros_like(self.grid, dtype=np.float32)
        
        # Occupied cells: high cost
        costmap[self.grid == 100] = 255
        
        # Unknown cells: medium cost
        costmap[self.grid == -1] = 128
        
        # Apply distance transform for gradient
        if np.any(self.grid == 100):
            dist_transform = cv2.distanceTransform(
                (self.grid != 100).astype(np.uint8),
                cv2.DIST_L2, 5
            )
            
            # Costs based on distance to obstacles
            safe_distance = robot_radius / self.resolution
            costmap[dist_transform < safe_distance] = 255
            costmap[dist_transform < safe_distance * 2] = 200
            costmap[dist_transform < safe_distance * 3] = 150
        
        return costmap

# mapping/world_model.py

class WorldModel(WorldModelInterface):
    """Unified world model combining multiple information sources"""
    
    def __init__(self, mapping_config):
        self.config = mapping_config
        
        # Occupancy grid
        self.occupancy_grid = OccupancyGrid(
            width=mapping_config.grid_width,
            height=mapping_config.grid_height,
            resolution=mapping_config.resolution
        )
        
        # Dynamic objects
        self.dynamic_objects: Dict[int, Obstacle] = {}
        self.static_objects: Dict[int, Obstacle] = {}
        
        # World state
        self.world_state = {
            'timestamp': 0.0,
            'robot_pose': None,
            'objects_of_interest': [],  # Delivery-relevant objects
            'risk_zones': [],  # High-risk areas
            'free_space_polygons': []  # Navigable areas
        }
        
        # Update counter
        self.update_count = 0
    
    def update(self, tracked_objects: List[TrackedObject],
               obstacles: List[Obstacle],
               timestamp: float):
        """Update world model with new observations"""
        
        self.update_count += 1
        
        # Update dynamic objects
        self._update_dynamic_objects(tracked_objects, timestamp)
        
        # Update static objects
        self._update_static_objects(obstacles)
        
        # Update occupancy grid
        self._update_occupancy_grid()
        
        # Update world state
        self._update_world_state(timestamp)
    
    def _update_dynamic_objects(self, tracked_objects: List[TrackedObject],
                               timestamp: float):
        """Update tracked dynamic objects"""
        for obj in tracked_objects:
            obstacle = Obstacle(
                id=obj.track_id,
                position=obj.position,
                radius=self._get_class_radius(obj.class_name),
                obstacle_type=obj.class_name,
                velocity=obj.velocity,
                confidence=obj.confidence,
                timestamp=timestamp
            )
            
            self.dynamic_objects[obj.track_id] = obstacle
    
    def _get_class_radius(self, class_name: str) -> float:
        """Get obstacle radius based on class"""
        radius_map = {
            'person': 0.3,
            'bicycle': 0.5,
            'car': 1.5,
            'truck': 2.0,
            'bus': 2.5,
            'dog': 0.2,
            'cat': 0.15
        }
        return radius_map.get(class_name, 0.5)
    
    def _update_occupancy_grid(self):
        """Update occupancy grid with all obstacles"""
        all_obstacles = list(self.dynamic_objects.values()) + \
                       list(self.static_objects.values())
        
        # Clear previous dynamic obstacles
        # (Implementation depends on grid update frequency)
        
        # Update grid
        robot_position = self.world_state.get('robot_pose', (0, 0))
        self.occupancy_grid.update_occupancy(all_obstacles, robot_position)
        
        # Inflate obstacles
        self.occupancy_grid.inflate_obstacles()
    
    def _update_world_state(self, timestamp: float):
        """Update world state metadata"""
        self.world_state['timestamp'] = timestamp
        self.world_state['dynamic_object_count'] = len(self.dynamic_objects)
        self.world_state['static_object_count'] = len(self.static_objects)
        
        # Identify objects of interest for delivery
        self.world_state['objects_of_interest'] = [
            obj for obj in self.dynamic_objects.values()
            if obj.obstacle_type in ['delivery_box', 'person', 'vehicle']
        ]
        
        # Calculate risk zones (areas with high dynamic object density)
        risk_zones = self._calculate_risk_zones()
        self.world_state['risk_zones'] = risk_zones
    
    def _calculate_risk_zones(self) -> List[Dict]:
        """Calculate high-risk areas based on dynamic object predictions"""
        risk_zones = []
        
        for obj_id, obj in self.dynamic_objects.items():
            # Predict future positions
            if obj.velocity and np.linalg.norm(obj.velocity) > 0.5:
                # Object is moving significantly
                future_positions = self._predict_trajectory(obj, 
                    time_horizon=3.0, steps=10)
                
                risk_zones.append({
                    'object_id': obj_id,
                    'type': obj.obstacle_type,
                    'current_position': obj.position,
                    'predicted_path': future_positions,
                    'risk_level': self._calculate_risk_level(obj)
                })
        
        return risk_zones
    
    def _predict_trajectory(self, obstacle: Obstacle, 
                           time_horizon: float, 
                           steps: int) -> List[Tuple[float, float]]:
        """Predict future trajectory of dynamic object"""
        trajectory = []
        
        dt = time_horizon / steps
        pos = np.array(obstacle.position)
        vel = np.array(obstacle.velocity) if obstacle.velocity else np.zeros(2)
        
        for i in range(steps):
            pos = pos + vel * dt
            trajectory.append(tuple(pos))
        
        return trajectory
    
    def _calculate_risk_level(self, obstacle: Obstacle) -> float:
        """Calculate risk level (0-1) for an obstacle"""
        risk = 0.0
        
        # Speed-based risk
        if obstacle.velocity:
            speed = np.linalg.norm(obstacle.velocity)
            risk += min(speed / 5.0, 0.5)  # Max 0.5 from speed
        
        # Proximity-based risk (to be calculated with robot position)
        # Type-based risk
        if obstacle.obstacle_type in ['person', 'bicycle']:
            risk += 0.3  # Vulnerable road users
        elif obstacle.obstacle_type in ['car', 'truck', 'bus']:
            risk += 0.5  # Vehicles
        
        return min(risk, 1.0)
    
    def get_obstacles(self) -> List[Obstacle]:
        """Get all current obstacles"""
        return list(self.dynamic_objects.values()) + \
               list(self.static_objects.values())
    
    def get_occupancy_grid(self) -> np.ndarray:
        """Get current occupancy grid"""
        return self.occupancy_grid.get_grid()
    
    def get_world_state(self) -> Dict:
        """Get complete world state"""
        return self.world_state
    
    def get_costmap(self) -> np.ndarray:
        """Get costmap for planning"""
        return self.occupancy_grid.get_costmap()