"""
Camera calibration utilities for intrinsic and extrinsic parameters.
Supports chessboard pattern calibration and parameter management.
"""

import cv2
import numpy as np
import json
import os
import logging
from typing import Tuple, List, Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

class CameraCalibration:
    """Camera calibration using chessboard or circle grid patterns
    
    This class handles:
    - Intrinsic calibration (camera matrix, distortion coefficients)
    - Extrinsic calibration (camera mounting pose)
    - Calibration image collection and validation
    - Parameter saving and loading
    - Image undistortion
    """
    
    def __init__(self, pattern_size: Tuple[int, int] = (9, 6), 
                 square_size: float = 0.025,
                 pattern_type: str = 'chessboard'):
        """
        Args:
            pattern_size: Number of inner corners (columns, rows)
            square_size: Size of chessboard square in meters
            pattern_type: 'chessboard', 'circles', or 'asymmetric_circles'
        """
        self.pattern_size = pattern_size
        self.square_size = square_size
        self.pattern_type = pattern_type
        
        # Prepare object points (3D points in real world space)
        self.objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:pattern_size[0], 
                                    0:pattern_size[1]].T.reshape(-1, 2)
        self.objp *= square_size
        
        # Arrays to store object points and image points from all images
        self.objpoints = []  # 3D points in real world space
        self.imgpoints = []  # 2D points in image plane
        
        # Calibration results
        self.camera_matrix = None
        self.dist_coeffs = None
        self.rvecs = None
        self.tvecs = None
        self.calibration_error = None
        self.image_size = None
        
        # Calibration quality metrics
        self.per_view_errors = []
        self.is_calibrated = False
        
        # Extrinsic parameters (camera mounting)
        self.rotation_matrix = None
        self.translation_vector = None
        
    def find_corners(self, image: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        """Find calibration pattern corners in image
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            Tuple of (success, corners) where corners is Nx1x2 array
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        if self.pattern_type == 'chessboard':
            return self._find_chessboard_corners(gray)
        elif self.pattern_type == 'circles':
            return self._find_circles_grid(gray)
        elif self.pattern_type == 'asymmetric_circles':
            return self._find_asymmetric_circles(gray)
        else:
            raise ValueError(f"Unknown pattern type: {self.pattern_type}")
    
    def _find_chessboard_corners(self, gray: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        """Find chessboard corners using OpenCV"""
        # Try different methods for better detection
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        
        ret, corners = cv2.findChessboardCorners(
            gray, self.pattern_size, flags
        )
        
        if ret:
            # Refine corner locations to sub-pixel accuracy
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            return True, corners
        
        return False, None
    
    def _find_circles_grid(self, gray: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        """Find symmetric circles grid pattern"""
        flags = cv2.CALIB_CB_SYMMETRIC_GRID
        
        ret, centers = cv2.findCirclesGrid(
            gray, self.pattern_size, flags=flags
        )
        
        return ret, centers if ret else None
    
    def _find_asymmetric_circles(self, gray: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        """Find asymmetric circles grid pattern"""
        flags = cv2.CALIB_CB_ASYMMETRIC_GRID
        
        ret, centers = cv2.findCirclesGrid(
            gray, self.pattern_size, flags=flags
        )
        
        return ret, centers if ret else None
    
    def add_calibration_image(self, image: np.ndarray) -> bool:
        """Add an image for calibration
        
        Args:
            image: Calibration pattern image
            
        Returns:
            bool: True if pattern was found and added
        """
        if self.image_size is None:
            self.image_size = (image.shape[1], image.shape[0])
        
        ret, corners = self.find_corners(image)
        
        if ret:
            self.objpoints.append(self.objp)
            self.imgpoints.append(corners)
            logger.info(f"Added calibration image. Total: {len(self.objpoints)}")
            return True
        
        logger.warning("Calibration pattern not found in image")
        return False
    
    def calibrate(self, image_size: Optional[Tuple[int, int]] = None) -> bool:
        """Perform camera calibration
        
        Args:
            image_size: Image size (width, height). Uses stored size if None
            
        Returns:
            bool: True if calibration successful
        """
        if image_size is not None:
            self.image_size = image_size
        
        if self.image_size is None:
            logger.error("Image size not set")
            return False
        
        min_images = 10 if self.pattern_type == 'chessboard' else 5
        
        if len(self.objpoints) < min_images:
            logger.error(f"Need at least {min_images} images, got {len(self.objpoints)}")
            return False
        
        logger.info(f"Calibrating camera with {len(self.objpoints)} images...")
        
        try:
            # Perform calibration
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                self.objpoints, 
                self.imgpoints, 
                self.image_size, 
                None, 
                None,
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
            )
            
            if ret:
                self.camera_matrix = mtx
                self.dist_coeffs = dist
                self.rvecs = rvecs
                self.tvecs = tvecs
                
                # Calculate reprojection error
                self.calibration_error = self._calculate_reprojection_error()
                self.per_view_errors = self._calculate_per_view_errors()
                
                self.is_calibrated = True
                
                logger.info(f"Calibration successful!")
                logger.info(f"  RMS Error: {self.calibration_error:.4f} pixels")
                logger.info(f"  Camera Matrix:\n{self.camera_matrix}")
                logger.info(f"  Distortion Coeffs: {self.dist_coeffs.ravel()}")
                
                return True
            else:
                logger.error("Calibration optimization failed")
                return False
                
        except Exception as e:
            logger.error(f"Calibration failed: {e}")
            return False
    
    def _calculate_reprojection_error(self) -> float:
        """Calculate overall RMS reprojection error"""
        total_error = 0
        total_points = 0
        
        for i in range(len(self.objpoints)):
            imgpoints2, _ = cv2.projectPoints(
                self.objpoints[i], 
                self.rvecs[i], 
                self.tvecs[i], 
                self.camera_matrix, 
                self.dist_coeffs
            )
            error = cv2.norm(self.imgpoints[i], imgpoints2, cv2.NORM_L2)
            total_error += error ** 2
            total_points += len(self.objpoints[i])
        
        return np.sqrt(total_error / total_points)
    
    def _calculate_per_view_errors(self) -> List[float]:
        """Calculate reprojection error for each view"""
        errors = []
        
        for i in range(len(self.objpoints)):
            imgpoints2, _ = cv2.projectPoints(
                self.objpoints[i], 
                self.rvecs[i], 
                self.tvecs[i], 
                self.camera_matrix, 
                self.dist_coeffs
            )
            error = cv2.norm(self.imgpoints[i], imgpoints2, cv2.NORM_L2)
            errors.append(error / len(self.objpoints[i]))
        
        return errors
    
    def undistort_image(self, image: np.ndarray, 
                       alpha: float = 0.0,
                       crop: bool = True) -> np.ndarray:
        """Undistort image using calibration parameters
        
        Args:
            image: Input distorted image
            alpha: Free scaling parameter (0=crop to valid, 1=keep all pixels)
            crop: Whether to crop the image to valid region
            
        Returns:
            Undistorted image
        """
        if not self.is_calibrated:
            logger.warning("Camera not calibrated, returning original image")
            return image
        
        h, w = image.shape[:2]
        
        # Get optimal camera matrix
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, (w, h), alpha, (w, h)
        )
        
        # Undistort
        if alpha == 0:
            # Use cv2.undistort for cropped result
            dst = cv2.undistort(image, self.camera_matrix, self.dist_coeffs, None, newcameramtx)
        else:
            # Use remap for more control
            mapx, mapy = cv2.initUndistortRectifyMap(
                self.camera_matrix, self.dist_coeffs, None,
                newcameramtx, (w, h), 5
            )
            dst = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
        
        # Crop the image if requested
        if crop and alpha == 0:
            x, y, w, h = roi
            if all([x, y, w, h]):
                dst = dst[y:y+h, x:x+w]
        
        return dst
    
    def calibrate_extrinsic(self, object_points: np.ndarray,
                           image_points: np.ndarray) -> bool:
        """Calibrate extrinsic parameters (camera pose)
        
        Args:
            object_points: 3D points in world frame (N, 3)
            image_points: 2D points in image (N, 2)
            
        Returns:
            bool: True if successful
        """
        if not self.is_calibrated:
            logger.error("Must calibrate intrinsic parameters first")
            return False
        
        try:
            ret, rvec, tvec = cv2.solvePnP(
                object_points, image_points,
                self.camera_matrix, self.dist_coeffs
            )
            
            if ret:
                self.rotation_matrix, _ = cv2.Rodrigues(rvec)
                self.translation_vector = tvec
                
                logger.info("Extrinsic calibration successful")
                logger.info(f"  Translation: {tvec.ravel()}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Extrinsic calibration failed: {e}")
            return False
    
    def get_camera_pose(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Get camera pose in world frame
        
        Returns:
            Tuple of (rotation_matrix, translation_vector) or None
        """
        if self.rotation_matrix is None or self.translation_vector is None:
            return None
        
        return self.rotation_matrix.copy(), self.translation_vector.copy()
    
    def project_points(self, object_points: np.ndarray) -> Optional[np.ndarray]:
        """Project 3D points to image plane
        
        Args:
            object_points: 3D points (N, 3)
            
        Returns:
            2D image points (N, 2) or None
        """
        if not self.is_calibrated:
            return None
        
        # Need both intrinsic and extrinsic
        if self.rotation_matrix is None or self.translation_vector is None:
            # Assume points are in camera frame
            rvec = np.zeros(3)
            tvec = np.zeros(3)
        else:
            rvec, _ = cv2.Rodrigues(self.rotation_matrix)
            tvec = self.translation_vector
        
        image_points, _ = cv2.projectPoints(
            object_points, rvec, tvec,
            self.camera_matrix, self.dist_coeffs
        )
        
        return image_points.reshape(-1, 2)
    
    def save_calibration(self, filepath: str) -> bool:
        """Save calibration parameters to JSON file
        
        Args:
            filepath: Path to save calibration file
            
        Returns:
            bool: True if saved successfully
        """
        if not self.is_calibrated:
            logger.error("No calibration data to save")
            return False
        
        calibration_data = {
            'camera_matrix': self.camera_matrix.tolist(),
            'dist_coeffs': self.dist_coeffs.tolist(),
            'calibration_error': float(self.calibration_error),
            'image_size': list(self.image_size),
            'pattern_size': list(self.pattern_size),
            'square_size': self.square_size,
            'pattern_type': self.pattern_type,
            'per_view_errors': [float(e) for e in self.per_view_errors],
            'num_images': len(self.objpoints)
        }
        
        # Add extrinsic parameters if available
        if self.rotation_matrix is not None:
            calibration_data['rotation_matrix'] = self.rotation_matrix.tolist()
        if self.translation_vector is not None:
            calibration_data['translation_vector'] = self.translation_vector.tolist()
        
        # Create directory if needed
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(calibration_data, f, indent=2)
            
            logger.info(f"Calibration saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save calibration: {e}")
            return False
    
    def load_calibration(self, filepath: str) -> bool:
        """Load calibration parameters from JSON file
        
        Args:
            filepath: Path to calibration file
            
        Returns:
            bool: True if loaded successfully
        """
        if not os.path.exists(filepath):
            logger.error(f"Calibration file not found: {filepath}")
            return False
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.camera_matrix = np.array(data['camera_matrix'])
            self.dist_coeffs = np.array(data['dist_coeffs'])
            self.calibration_error = data.get('calibration_error', None)
            self.image_size = tuple(data['image_size'])
            self.pattern_size = tuple(data['pattern_size'])
            self.square_size = data['square_size']
            self.pattern_type = data.get('pattern_type', 'chessboard')
            self.per_view_errors = data.get('per_view_errors', [])
            
            # Load extrinsic parameters if available
            if 'rotation_matrix' in data:
                self.rotation_matrix = np.array(data['rotation_matrix'])
            if 'translation_vector' in data:
                self.translation_vector = np.array(data['translation_vector'])
            
            self.is_calibrated = True
            
            logger.info(f"Calibration loaded from {filepath}")
            logger.info(f"  Camera Matrix:\n{self.camera_matrix}")
            logger.info(f"  RMS Error: {self.calibration_error:.4f} pixels")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load calibration: {e}")
            return False
    
    def get_calibration_params(self) -> Dict[str, Any]:
        """Get all calibration parameters
        
        Returns:
            Dictionary with calibration parameters
        """
        if not self.is_calibrated:
            return {}
        
        params = {
            'camera_matrix': self.camera_matrix,
            'dist_coeffs': self.dist_coeffs,
            'error': self.calibration_error,
            'image_size': self.image_size,
            'is_calibrated': self.is_calibrated
        }
        
        if self.rotation_matrix is not None:
            params['rotation_matrix'] = self.rotation_matrix
        if self.translation_vector is not None:
            params['translation_vector'] = self.translation_vector
        
        return params
    
    def create_undistort_maps(self, image_size: Tuple[int, int],
                             alpha: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """Create undistortion maps for faster processing
        
        Args:
            image_size: Image size (width, height)
            alpha: Free scaling parameter
            
        Returns:
            Tuple of (mapx, mapy) for cv2.remap
        """
        if not self.is_calibrated:
            raise RuntimeError("Camera not calibrated")
        
        newcameramtx, _ = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, image_size, alpha, image_size
        )
        
        mapx, mapy = cv2.initUndistortRectifyMap(
            self.camera_matrix, self.dist_coeffs, None,
            newcameramtx, image_size, 5
        )
        
        return mapx, mapy
    
    def draw_corners(self, image: np.ndarray) -> Tuple[bool, np.ndarray]:
        """Find and draw calibration pattern corners
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (success, annotated_image)
        """
        vis_image = image.copy()
        ret, corners = self.find_corners(image)
        
        if ret:
            cv2.drawChessboardCorners(
                vis_image, self.pattern_size, corners, ret
            )
        
        return ret, vis_image
    
    def get_calibration_quality(self) -> Dict[str, Any]:
        """Assess calibration quality
        
        Returns:
            Dictionary with quality metrics
        """
        if not self.is_calibrated:
            return {'status': 'not_calibrated'}
        
        quality = {
            'status': 'calibrated',
            'rms_error': self.calibration_error,
            'num_images': len(self.objpoints),
            'image_size': self.image_size
        }
        
        # Quality assessment based on RMS error
        if self.calibration_error < 0.2:
            quality['grade'] = 'excellent'
        elif self.calibration_error < 0.5:
            quality['grade'] = 'good'
        elif self.calibration_error < 1.0:
            quality['grade'] = 'acceptable'
        else:
            quality['grade'] = 'poor'
        
        # Check for problematic images
        if self.per_view_errors:
            mean_error = np.mean(self.per_view_errors)
            std_error = np.std(self.per_view_errors)
            problematic = [
                i for i, e in enumerate(self.per_view_errors)
                if e > mean_error + 2 * std_error
            ]
            quality['problematic_images'] = problematic
            quality['max_error'] = max(self.per_view_errors)
            quality['min_error'] = min(self.per_view_errors)
        
        return quality
    
    def reset(self):
        """Reset calibration data"""
        self.objpoints = []
        self.imgpoints = []
        self.camera_matrix = None
        self.dist_coeffs = None
        self.rvecs = None
        self.tvecs = None
        self.calibration_error = None
        self.image_size = None
        self.per_view_errors = []
        self.is_calibrated = False
        self.rotation_matrix = None
        self.translation_vector = None
        logger.info("Calibration data reset")

# Example usage and utility functions

def calibrate_from_images(image_paths: List[str], 
                         pattern_size: Tuple[int, int] = (9, 6),
                         square_size: float = 0.025) -> Optional[CameraCalibration]:
    """Convenience function to calibrate from a list of image paths
    
    Args:
        image_paths: List of paths to calibration images
        pattern_size: Chessboard pattern size
        square_size: Square size in meters
        
    Returns:
        Calibrated CameraCalibration object or None
    """
    calibration = CameraCalibration(pattern_size, square_size)
    
    for path in image_paths:
        image = cv2.imread(path)
        if image is None:
            logger.warning(f"Could not read image: {path}")
            continue
        
        success = calibration.add_calibration_image(image)
        if success:
            logger.info(f"Found pattern in {path}")
        else:
            logger.warning(f"No pattern found in {path}")
    
    if calibration.calibrate():
        return calibration
    
    return None


def calibrate_from_camera(camera, num_images: int = 20,
                         pattern_size: Tuple[int, int] = (9, 6),
                         square_size: float = 0.025,
                         delay: float = 1.0) -> Optional[CameraCalibration]:
    """Convenience function to calibrate from live camera feed
    
    Args:
        camera: Camera interface object
        num_images: Number of images to collect
        pattern_size: Chessboard pattern size
        square_size: Square size in meters
        delay: Delay between captures in seconds
        
    Returns:
        Calibrated CameraCalibration object or None
    """
    import time
    
    calibration = CameraCalibration(pattern_size, square_size)
    images_collected = 0
    
    logger.info(f"Starting live calibration. Need {num_images} images.")
    logger.info("Show calibration pattern to camera...")
    
    while images_collected < num_images:
        frame, _ = camera.get_frame()
        if frame is None:
            continue
        
        # Draw corners for visualization
        ret, vis = calibration.draw_corners(frame)
        
        # Display
        cv2.imshow('Calibration', vis)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):  # Space to capture
            if ret:
                calibration.add_calibration_image(frame)
                images_collected += 1
                logger.info(f"Captured image {images_collected}/{num_images}")
            else:
                logger.warning("Pattern not found. Adjust position.")
        
        elif key == ord('q'):  # Q to quit
            break
        
        time.sleep(0.01)
    
    cv2.destroyWindow('Calibration')
    
    if images_collected >= 10:
        calibration.calibrate()
        return calibration
    
    logger.error(f"Not enough images collected: {images_collected}")
    return None


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create calibration object
    calib = CameraCalibration(pattern_size=(9, 6), square_size=0.025)
    
    # Load test images (replace with actual paths)
    test_images = ["calib_img1.jpg", "calib_img2.jpg"]
    
    calibration = calibrate_from_images(test_images)
    
    if calibration and calibration.is_calibrated:
        print(f"Calibration successful! Error: {calibration.calibration_error:.4f} px")
        calibration.save_calibration("camera_calibration.json")
    else:
        print("Calibration failed")