"""
Visualization tools for perception debugging and monitoring.
"""

import cv2
import numpy as np
from typing import Optional, List, Tuple
import time
from .core_interfaces import (
    Detection, TrackedObject, Obstacle, 
    PerceptionOutput, RobotPose
)

class VisualizationManager:
    """Manages visualization of perception data"""
    
    def __init__(self, config=None):
        self.config = config
        self.window_name = "Perception View"
        self.fps_history = []
        self.last_fps_update = time.time()
        self.current_fps = 0
        self.frame_count = 0
        
        # Colors (BGR)
        self.colors = {
            'person': (0, 255, 0),      # Green
            'bicycle': (255, 255, 0),    # Cyan
            'car': (255, 0, 0),          # Blue
            'truck': (255, 0, 255),      # Magenta
            'bus': (0, 255, 255),        # Yellow
            'dog': (128, 128, 0),        # Teal
            'cat': (128, 0, 128),        # Purple
            'default': (0, 255, 0)       # Green
        }
        
        # Initialize windows
        if self.config and self.config.get('enabled', True):
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.namedWindow("Occupancy Grid", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Tracks View", cv2.WINDOW_NORMAL)
    
    def update_fps(self):
        """Update FPS counter"""
        self.frame_count += 1
        current_time = time.time()
        elapsed = current_time - self.last_fps_update
        
        if elapsed >= 1.0:
            self.current_fps = self.frame_count / elapsed
            self.frame_count = 0
            self.last_fps_update = current_time
    
    def draw_detections(self, image: np.ndarray, 
                        detections: List[Detection]) -> np.ndarray:
        """Draw detection boxes on image"""
        vis_image = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = self.colors.get(det.class_name, self.colors['default'])
            
            # Draw bounding box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{det.class_name}: {det.confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(vis_image, 
                         (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0], y1),
                         color, -1)
            cv2.putText(vis_image, label, 
                       (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return vis_image
    
    def draw_tracks(self, image: np.ndarray, 
                    tracked_objects: List[TrackedObject]) -> np.ndarray:
        """Draw tracked objects on image"""
        vis_image = image.copy()
        
        for obj in tracked_objects:
            if obj.bbox is not None:
                x1, y1, x2, y2 = obj.bbox
                color = self.colors.get(obj.class_name, self.colors['default'])
                
                # Draw thicker box for tracked objects
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 3)
                
                # Draw track ID and velocity
                label = f"ID:{obj.track_id} {obj.class_name}"
                cv2.putText(vis_image, label,
                           (x1, y1 - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Draw velocity arrow
                if obj.velocity is not None:
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    arrow_end = (
                        int(center_x + obj.velocity[0] * 50),
                        int(center_y + obj.velocity[1] * 50)
                    )
                    cv2.arrowedLine(vis_image, 
                                   (center_x, center_y), 
                                   arrow_end, 
                                   (0, 0, 255), 2)
        
        return vis_image
    
    def draw_occupancy_grid(self, grid: np.ndarray) -> np.ndarray:
        """Create visualization of occupancy grid"""
        # Normalize grid values to 0-255
        vis_grid = np.zeros((*grid.shape, 3), dtype=np.uint8)
        
        # Free space: white
        vis_grid[grid == 0] = [255, 255, 255]
        # Occupied: black
        vis_grid[grid == 100] = [0, 0, 0]
        # Unknown: gray
        vis_grid[grid == -1] = [128, 128, 128]
        # Inflated obstacles: red
        vis_grid[(grid > 0) & (grid < 100)] = [0, 0, 255]
        
        # Resize for display
        display_size = (400, 400)
        vis_grid = cv2.resize(vis_grid, display_size, 
                             interpolation=cv2.INTER_NEAREST)
        
        return vis_grid
    
    def draw_robot_on_grid(self, grid_vis: np.ndarray, 
                          robot_pose: RobotPose,
                          grid_resolution: float = 0.05) -> np.ndarray:
        """Draw robot position on occupancy grid"""
        center_x = grid_vis.shape[1] // 2
        center_y = grid_vis.shape[0] // 2
        
        # Draw robot as a circle
        cv2.circle(grid_vis, (center_x, center_y), 8, (0, 255, 0), -1)
        
        # Draw orientation
        arrow_length = 20
        arrow_end = (
            int(center_x + arrow_length * np.cos(robot_pose.theta)),
            int(center_y - arrow_length * np.sin(robot_pose.theta))
        )
        cv2.arrowedLine(grid_vis, (center_x, center_y), arrow_end, 
                       (0, 255, 0), 2)
        
        return grid_vis
    
    def render(self, perception_output: PerceptionOutput, 
               raw_image: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """Render all visualizations"""
        self.update_fps()
        
        visualizations = {}
        
        # 1. Detection/Tracking view
        if raw_image is not None:
            det_vis = self.draw_detections(
                raw_image, 
                []  # Would need detections, using tracked objects instead
            )
            track_vis = self.draw_tracks(
                det_vis, 
                perception_output.tracked_objects
            )
            
            # Add FPS
            cv2.putText(track_vis, f"FPS: {self.current_fps:.1f}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0, 255, 0), 2)
            
            visualizations['main_view'] = track_vis
        
        # 2. Occupancy grid
        grid_vis = self.draw_occupancy_grid(perception_output.occupancy_grid)
        grid_vis = self.draw_robot_on_grid(
            grid_vis, 
            perception_output.robot_pose
        )
        visualizations['occupancy_grid'] = grid_vis
        
        # 3. Bird's eye view
        bird_view = self._create_bird_view(perception_output)
        visualizations['bird_view'] = bird_view
        
        return visualizations
    
    def _create_bird_view(self, output: PerceptionOutput) -> np.ndarray:
        """Create bird's eye view of obstacles"""
        size = 500
        scale = 100  # pixels per meter
        bird_view = np.ones((size, size, 3), dtype=np.uint8) * 255
        
        center = size // 2
        
        # Draw grid
        for i in range(0, size, scale):
            cv2.line(bird_view, (i, 0), (i, size), (200, 200, 200), 1)
            cv2.line(bird_view, (0, i), (size, i), (200, 200, 200), 1)
        
        # Draw obstacles
        for obs in output.obstacle_list:
            x = int(center + obs.position[0] * scale)
            y = int(center - obs.position[1] * scale)
            radius = int(obs.radius * scale)
            
            if 0 <= x < size and 0 <= y < size:
                color = (0, 0, 255) if obs.obstacle_type == 'person' else (255, 0, 0)
                cv2.circle(bird_view, (x, y), max(radius, 5), color, -1)
        
        # Draw robot
        cv2.circle(bird_view, (center, center), 10, (0, 255, 0), -1)
        
        return bird_view
    
    def show(self, visualizations: Dict[str, np.ndarray]) -> None:
        """Display all visualizations"""
        if 'main_view' in visualizations:
            cv2.imshow(self.window_name, visualizations['main_view'])
        
        if 'occupancy_grid' in visualizations:
            cv2.imshow("Occupancy Grid", visualizations['occupancy_grid'])
        
        if 'bird_view' in visualizations:
            cv2.imshow("Tracks View", visualizations['bird_view'])
        
        cv2.waitKey(1)
    
    def cleanup(self):
        """Clean up visualization windows"""
        cv2.destroyAllWindows()