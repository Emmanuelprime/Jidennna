"""
Stereo vision processing for depth estimation.
Uses OpenCV's stereo matching algorithms.
"""

import numpy as np
import cv2
from typing import Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)

class StereoProcessor:
    """Processes stereo image pairs for depth estimation"""
    
    def __init__(self):
        # Stereo matcher
        self.stereo = None
        self.matcher_type = 'SGBM'  # or 'BM'
        
        # Camera parameters
        self.baseline = 0.12  # meters between cameras
        self.focal_length = 615.0  # pixels
        
        # Disparity parameters
        self.min_disparity = 0
        self.num_disparities = 128
        self.block_size = 11
        
        # Post-processing
        self.use_wls_filter = True
        self.wls_lambda = 8000
        self.wls_sigma = 1.5
        
        self.is_initialized = False
    
    def initialize(self, config: Dict = None) -> bool:
        """Initialize stereo processor
        
        Args:
            config: Configuration dictionary
            
        Returns:
            bool: True if initialization successful
        """
        try:
            if config:
                self.baseline = config.get('baseline', self.baseline)
                self.focal_length = config.get('focal_length', self.focal_length)
                self.matcher_type = config.get('matcher_type', self.matcher_type)
            
            # Create stereo matcher
            if self.matcher_type == 'SGBM':
                self.stereo = cv2.StereoSGBM_create(
                    minDisparity=self.min_disparity,
                    numDisparities=self.num_disparities,
                    blockSize=self.block_size,
                    P1=8 * 3 * self.block_size ** 2,
                    P2=32 * 3 * self.block_size ** 2,
                    disp12MaxDiff=1,
                    uniquenessRatio=10,
                    speckleWindowSize=100,
                    speckleRange=32
                )
            else:  # BM
                self.stereo = cv2.StereoBM_create(
                    numDisparities=self.num_disparities,
                    blockSize=self.block_size
                )
            
            # Create WLS filter for post-processing
            if self.use_wls_filter:
                self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(self.stereo)
                self.right_matcher = cv2.ximgproc.createRightMatcher(self.stereo)
                
                self.wls_filter.setLambda(self.wls_lambda)
                self.wls_filter.setSigmaColor(self.wls_sigma)
            
            self.is_initialized = True
            logger.info(f"Stereo processor initialized ({self.matcher_type})")
            return True
            
        except Exception as e:
            logger.error(f"Stereo initialization failed: {e}")
            return False
    
    def compute_depth(self, left_image: np.ndarray,
                     right_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute depth map from stereo pair
        
        Args:
            left_image: Left camera image
            right_image: Right camera image
            
        Returns:
            (disparity_map, depth_map)
        """
        if not self.is_initialized:
            raise RuntimeError("Stereo processor not initialized")
        
        # Convert to grayscale
        if len(left_image.shape) == 3:
            left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
        else:
            left_gray = left_image
            right_gray = right_image
        
        # Compute disparity
        if self.use_wls_filter:
            # Compute left and right disparities
            left_disp = self.stereo.compute(left_gray, right_gray)
            right_disp = self.right_matcher.compute(right_gray, left_gray)
            
            # Apply WLS filter
            disparity = self.wls_filter.filter(
                left_disp, left_gray,
                disparity_map_right=right_disp
            )
        else:
            disparity = self.stereo.compute(left_gray, right_gray)
        
        # Convert disparity to float32
        disparity = disparity.astype(np.float32) / 16.0
        
        # Compute depth map
        depth_map = np.zeros_like(disparity)
        mask = disparity > 0
        
        # Depth = (focal_length * baseline) / disparity
        depth_map[mask] = (self.focal_length * self.baseline) / disparity[mask]
        
        # Clip to reasonable range
        depth_map = np.clip(depth_map, 0, 50)
        
        return disparity, depth_map
    
    def get_point_cloud(self, disparity: np.ndarray,
                       left_image: np.ndarray = None) -> np.ndarray:
        """Generate 3D point cloud from disparity map
        
        Args:
            disparity: Disparity map
            left_image: Optional left image for coloring points
            
        Returns:
            Point cloud (N, 3) or (N, 6) with colors
        """
        h, w = disparity.shape
        
        # Create meshgrid of pixel coordinates
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        
        # Filter valid disparities
        valid = disparity > 0
        u_valid = u[valid]
        v_valid = v[valid]
        d_valid = disparity[valid]
        
        # Reproject to 3D
        Z = (self.focal_length * self.baseline) / d_valid
        X = (u_valid - w / 2) * Z / self.focal_length
        Y = (v_valid - h / 2) * Z / self.focal_length
        
        points_3d = np.stack([X, Y, Z], axis=1)
        
        if left_image is not None:
            colors = left_image[valid]
            points_3d = np.concatenate([points_3d, colors], axis=1)
        
        return points_3d
    
    def get_depth_at_point(self, depth_map: np.ndarray,
                          point: Tuple[int, int]) -> Optional[float]:
        """Get depth at a specific image point
        
        Args:
            depth_map: Depth map
            point: (u, v) image coordinates
            
        Returns:
            Depth value or None if invalid
        """
        u, v = point
        h, w = depth_map.shape
        
        if 0 <= u < w and 0 <= v < h:
            depth = depth_map[v, u]
            if depth > 0:
                return float(depth)
        
        return None