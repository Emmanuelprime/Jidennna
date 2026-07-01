# simulation/simulated_camera.py

import numpy as np
import cv2
from typing import Tuple, List, Dict
import time

from perception.camera.camera_interface import CameraInterface

class SimulatedCamera(CameraInterface):
    """Simulated camera that generates realistic images from virtual world"""
    
    def __init__(self, config: Dict):
        self.width = config.get('width', 640)
        self.height = config.get('height', 480)
        self.fps = config.get('fps', 30)
        
        # Camera parameters
        self.fx = config.get('fx', 615.0)
        self.fy = config.get('fy', 615.0)
        self.cx = config.get('cx', 320.0)
        self.cy = config.get('cy', 240.0)
        
        # Simulation state
        self.frame_count = 0
        self.last_frame_time = time.time()
        
        # Background
        self.background = np.ones((self.height, self.width, 3), 
                                 dtype=np.uint8) * 200  # Gray background
        
        # Ground plane
        self.ground = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.ground[:] = [100, 150, 100]  # Green ground
        
    def initialize(self) -> bool:
        return True
    
    def get_frame(self) -> Tuple[np.ndarray, float]:
        """Generate simulated camera frame"""
        # Control frame rate
        current_time = time.time()
        time_diff = current_time - self.last_frame_time
        if time_diff < 1.0 / self.fps:
            time.sleep(1.0 / self.fps - time_diff)
        
        # Create base frame
        frame = self.ground.copy()
        
        # Add horizon
        horizon_y = int(self.height * 0.6)
        frame[:horizon_y, :] = [135, 206, 235]  # Sky blue
        
        # Get simulated objects from world
        objects = self._get_simulated_objects()
        
        # Render objects
        for obj in objects:
            frame = self._render_object(frame, obj)
        
        self.frame_count += 1
        self.last_frame_time = time.time()
        
        return frame, self.last_frame_time
    
    def _render_object(self, frame: np.ndarray, obj: Dict) -> np.ndarray:
        """Render a 3D object onto 2D image"""
        # Get object position in camera frame
        pos_3d = obj['position']  # (x, y, z) in camera frame
        
        if pos_3d[2] <= 0:  # Behind camera
            return frame
        
        # Project to image
        u = int(self.fx * pos_3d[0] / pos_3d[2] + self.cx)
        v = int(self.fy * pos_3d[1] / pos_3d[2] + self.cy)
        
        # Check if in image
        if not (0 <= u < self.width and 0 <= v < self.height):
            return frame
        
        # Draw based on object type
        if obj['type'] == 'person':
            color = (0, 0, 255)  # Red
            size = max(5, int(30 * 1.7 / pos_3d[2]))  # Scale with distance
            cv2.circle(frame, (u, v), size, color, -1)
            
            # Draw bounding box
            bbox_size = size * 2
            cv2.rectangle(frame, 
                         (u - bbox_size, v - bbox_size * 2),
                         (u + bbox_size, v),
                         color, 2)
            
        elif obj['type'] in ['car', 'truck', 'bus']:
            color = (255, 0, 0)  # Blue
            size = max(10, int(50 * 4.5 / pos_3d[2]))
            cv2.rectangle(frame,
                         (u - size, v - size//2),
                         (u + size, v + size//2),
                         color, -1)
            
        elif obj['type'] == 'delivery_box':
            color = (0, 255, 0)  # Green
            size = max(5, int(20 * 0.5 / pos_3d[2]))
            cv2.rectangle(frame,
                         (u - size, v - size),
                         (u + size, v + size),
                         color, -1)
        
        return frame
    
    def _get_simulated_objects(self) -> List[Dict]:
        """Get objects from simulated world"""
        # In production, this would query the SimulatedWorld
        # For demo, return some sample objects
        return []
    
    def get_intrinsics(self) -> Dict:
        return {
            'width': self.width,
            'height': self.height,
            'fx': self.fx,
            'fy': self.fy,
            'cx': self.cx,
            'cy': self.cy
        }
    
    def release(self) -> None:
        pass
    
    def is_healthy(self) -> bool:
        return True

# simulation/simulated_world.py

import numpy as np
from typing import List, Dict, Tuple
import yaml

class SimulatedWorld:
    """Simulated environment with moving objects"""
    
    def __init__(self, scenario_file: str = None):
        self.objects: List[Dict] = []
        self.robot_position = np.array([0.0, 0.0, 0.0])
        self.time = 0.0
        self.dt = 0.033  # 30 FPS
        
        # Noise parameters
        self.position_noise_std = 0.02
        self.velocity_noise_std = 0.1
        
        # Load scenario if provided
        if scenario_file:
            self._load_scenario(scenario_file)
    
    def _load_scenario(self, scenario_file: str):
        """Load scenario from YAML file"""
        with open(scenario_file, 'r') as f:
            scenario = yaml.safe_load(f)
        
        # Create pedestrians
        for ped_config in scenario.get('pedestrians', []):
            self.add_pedestrian(
                position=ped_config['start_position'],
                velocity=ped_config['velocity'],
                path=ped_config.get('path', [])
            )
        
        # Create vehicles
        for veh_config in scenario.get('vehicles', []):
            self.add_vehicle(
                position=veh_config['start_position'],
                velocity=veh_config['velocity'],
                path=veh_config.get('path', [])
            )
    
    def add_pedestrian(self, position: Tuple[float, float],
                      velocity: Tuple[float, float] = (0, 0),
                      path: List[Tuple[float, float]] = None):
        """Add a pedestrian to the simulation"""
        self.objects.append({
            'id': len(self.objects),
            'type': 'person',
            'position': np.array([position[0], position[1], 0.0]),
            'velocity': np.array([velocity[0], velocity[1], 0.0]),
            'path': path or [],
            'path_index': 0,
            'speed': np.linalg.norm(velocity),
            'direction': np.arctan2(velocity[1], velocity[0]) if np.linalg.norm(velocity) > 0 else 0
        })
    
    def add_vehicle(self, position: Tuple[float, float],
                   velocity: Tuple[float, float] = (0, 0),
                   path: List[Tuple[float, float]] = None):
        """Add a vehicle to the simulation"""
        self.objects.append({
            'id': len(self.objects),
            'type': 'car',
            'position': np.array([position[0], position[1], 0.0]),
            'velocity': np.array([velocity[0], velocity[1], 0.0]),
            'path': path or [],
            'path_index': 0,
            'speed': np.linalg.norm(velocity),
            'direction': np.arctan2(velocity[1], velocity[0]) if np.linalg.norm(velocity) > 0 else 0
        })
    
    def update(self, dt: float = None):
        """Update simulation step"""
        if dt is None:
            dt = self.dt
        
        self.time += dt
        
        for obj in self.objects:
            if obj['path'] and len(obj['path']) > 0:
                # Follow path
                self._follow_path(obj, dt)
            else:
                # Constant velocity movement
                obj['position'] += obj['velocity'] * dt
                
                # Add noise
                obj['position'] += np.random.normal(0, self.position_noise_std, 3)
                obj['velocity'] += np.random.normal(0, self.velocity_noise_std, 3)
    
    def _follow_path(self, obj: Dict, dt: float):
        """Make object follow a predefined path"""
        if obj['path_index'] >= len(obj['path']):
            # Loop back to start
            obj['path_index'] = 0
        
        # Get target waypoint
        target = np.array(obj['path'][obj['path_index']])
        current = obj['position'][:2]
        
        # Calculate direction to target
        direction = target - current
        distance = np.linalg.norm(direction)
        
        if distance < 0.5:  # Reached waypoint
            obj['path_index'] += 1
            if obj['path_index'] < len(obj['path']):
                target = np.array(obj['path'][obj['path_index']])
                direction = target - current
                distance = np.linalg.norm(direction)
        
        if distance > 0:
            # Move towards target
            speed = obj['speed']
            velocity = (direction / distance) * speed
            obj['velocity'] = np.array([velocity[0], velocity[1], 0.0])
            obj['position'] += obj['velocity'] * dt
    
    def get_objects_in_view(self, camera_position: np.ndarray,
                           camera_fov: float = np.pi/2,
                           max_distance: float = 20.0) -> List[Dict]:
        """Get objects visible from camera position"""
        visible_objects = []
        
        for obj in self.objects:
            # Calculate relative position
            relative = obj['position'] - camera_position
            distance = np.linalg.norm(relative)
            
            if distance < max_distance:
                # Check if in field of view
                angle = np.arctan2(relative[1], relative[0])
                if abs(angle) < camera_fov / 2:
                    visible_objects.append(obj)
        
        return visible_objects
    
    def get_ground_truth(self) -> List[Dict]:
        """Get ground truth for all objects"""
        return [{
            'id': obj['id'],
            'type': obj['type'],
            'position': obj['position'].tolist(),
            'velocity': obj['velocity'].tolist()
        } for obj in self.objects]

# simulation/pedestrian_simulator.py

class PedestrianSimulator:
    """Simulates realistic pedestrian behavior"""
    
    def __init__(self):
        self.pedestrians = []
        self.social_force_params = {
            'desired_speed': 1.4,  # m/s
            'relaxation_time': 0.5,  # seconds
            'repulsion_strength': 2.0,
            'repulsion_range': 2.0
        }
    
    def add_pedestrian(self, start_pos, goal_pos, speed=None):
        """Add pedestrian with goal"""
        if speed is None:
            speed = self.social_force_params['desired_speed']
        
        pedestrian = {
            'position': np.array(start_pos, dtype=float),
            'goal': np.array(goal_pos, dtype=float),
            'velocity': np.zeros(2),
            'speed': speed,
            'path_history': []
        }
        
        self.pedestrians.append(pedestrian)
        return len(self.pedestrians) - 1
    
    def update(self, dt: float, obstacles: List[np.ndarray] = None):
        """Update all pedestrians using social force model"""
        for ped in self.pedestrians:
            # Calculate desired force (towards goal)
            to_goal = ped['goal'] - ped['position']
            distance_to_goal = np.linalg.norm(to_goal)
            
            if distance_to_goal > 0.1:
                desired_direction = to_goal / distance_to_goal
                desired_velocity = desired_direction * ped['speed']
                desired_force = (desired_velocity - ped['velocity']) / \
                               self.social_force_params['relaxation_time']
                
                # Social forces (repulsion from other pedestrians)
                social_force = np.zeros(2)
                for other in self.pedestrians:
                    if other != ped:
                        to_other = ped['position'] - other['position']
                        distance = np.linalg.norm(to_other)
                        
                        if distance < self.social_force_params['repulsion_range']:
                            direction = to_other / (distance + 1e-6)
                            strength = self.social_force_params['repulsion_strength'] * \
                                      np.exp(-distance / self.social_force_params['repulsion_range'])
                            social_force += strength * direction
                
                # Update velocity and position
                total_force = desired_force + social_force
                ped['velocity'] += total_force * dt
                
                # Limit speed
                current_speed = np.linalg.norm(ped['velocity'])
                if current_speed > ped['speed'] * 1.2:
                    ped['velocity'] = ped['velocity'] / current_speed * ped['speed'] * 1.2
                
                ped['position'] += ped['velocity'] * dt
                
                # Store history
                ped['path_history'].append(ped['position'].copy())
                
                # Check if reached goal
                if distance_to_goal < 0.5:
                    # Assign new random goal
                    ped['goal'] = ped['position'] + \
                                 np.random.uniform(-10, 10, 2)