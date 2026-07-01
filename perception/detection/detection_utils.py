"""
Utility functions for object detection processing.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class ImagePreprocessor:
    """Preprocess images for neural network input"""
    
    @staticmethod
    def resize_and_pad(image: np.ndarray, 
                       target_size: Tuple[int, int] = (640, 640),
                       pad_color: Tuple[int, int, int] = (114, 114, 114)) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """Resize image with letterbox padding
        Args:
            image: Input image
            target_size: Desired output size (width, height)
            pad_color: Color for padding
        Returns:
            (resized_image, scale_factor, padding_offset)
        """
        h, w = image.shape[:2]
        target_w, target_h = target_size
        
        # Calculate scale
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Pad
        pad_w = (target_w - new_w) // 2
        pad_h = (target_h - new_h) // 2
        
        padded = np.full((target_h, target_w, 3), pad_color, dtype=np.uint8)
        padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
        
        return padded, scale, (pad_w, pad_h)
    
    @staticmethod
    def normalize(image: np.ndarray, 
                  mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
                  std: Tuple[float, float, float] = (0.229, 0.224, 0.225)) -> np.ndarray:
        """Normalize image with mean and std
        Args:
            image: Input image (H, W, 3) in range [0, 255]
            mean: Mean values for each channel
            std: Standard deviation for each channel
        Returns:
            Normalized image (H, W, 3) as float32
        """
        image = image.astype(np.float32) / 255.0
        image = (image - np.array(mean)) / np.array(std)
        return image
    
    @staticmethod
    def to_chw(image: np.ndarray) -> np.ndarray:
        """Convert HWC to CHW format
        Args:
            image: Input image (H, W, C)
        Returns:
            Image in CHW format
        """
        return np.transpose(image, (2, 0, 1))
    
    @staticmethod
    def add_batch_dim(image: np.ndarray) -> np.ndarray:
        """Add batch dimension
        Args:
            image: Input image
        Returns:
            Image with batch dimension
        """
        return np.expand_dims(image, axis=0)

class NMSProcessor:
    """Non-Maximum Suppression processor"""
    
    @staticmethod
    def nms(boxes: np.ndarray, scores: np.ndarray, 
            iou_threshold: float = 0.45) -> np.ndarray:
        """Apply Non-Maximum Suppression
        Args:
            boxes: Array of boxes (N, 4) in format [x1, y1, x2, y2]
            scores: Array of scores (N,)
            iou_threshold: IOU threshold for suppression
        Returns:
            Indices of kept boxes
        """
        # Convert to [x, y, w, h] format
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            # Compute IOU
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds = np.where(ovr <= iou_threshold)[0]
            order = order[inds + 1]
        
        return np.array(keep)
    
    @staticmethod
    def soft_nms(boxes: np.ndarray, scores: np.ndarray,
                 sigma: float = 0.5, score_threshold: float = 0.001,
                 method: str = 'gaussian') -> Tuple[np.ndarray, np.ndarray]:
        """Apply Soft-NMS
        Args:
            boxes: Array of boxes (N, 4)
            scores: Array of scores (N,)
            sigma: Gaussian sigma
            score_threshold: Minimum score threshold
            method: 'linear' or 'gaussian'
        Returns:
            (kept_boxes, kept_scores)
        """
        N = boxes.shape[0]
        indices = np.arange(N)
        
        for i in range(N):
            max_score = scores[i]
            max_pos = i
            
            # Find maximum score
            pos = i + 1
            while pos < N:
                if max_score < scores[pos]:
                    max_score = scores[pos]
                    max_pos = pos
                pos += 1
            
            # Swap
            boxes[i], boxes[max_pos] = boxes[max_pos].copy(), boxes[i].copy()
            scores[i], scores[max_pos] = scores[max_pos], scores[i]
            indices[i], indices[max_pos] = indices[max_pos], indices[i]
            
            # Calculate IOU with remaining boxes
            x1 = boxes[i, 0]
            y1 = boxes[i, 1]
            x2 = boxes[i, 2]
            y2 = boxes[i, 3]
            area = (x2 - x1 + 1) * (y2 - y1 + 1)
            
            pos = i + 1
            while pos < N:
                xx1 = max(x1, boxes[pos, 0])
                yy1 = max(y1, boxes[pos, 1])
                xx2 = min(x2, boxes[pos, 2])
                yy2 = min(y2, boxes[pos, 3])
                
                w = max(0, xx2 - xx1 + 1)
                h = max(0, yy2 - yy1 + 1)
                inter = w * h
                
                iou = inter / (area + (boxes[pos, 2] - boxes[pos, 0] + 1) * 
                              (boxes[pos, 3] - boxes[pos, 1] + 1) - inter)
                
                if method == 'linear':
                    weight = 1 - iou if iou > score_threshold else 1
                else:  # gaussian
                    weight = np.exp(-(iou * iou) / sigma)
                
                scores[pos] *= weight
                
                if scores[pos] < score_threshold:
                    boxes[pos] = boxes[N-1]
                    scores[pos] = scores[N-1]
                    indices[pos] = indices[N-1]
                    N -= 1
                    pos -= 1
                
                pos += 1
        
        keep = N
        return boxes[:keep], scores[:keep]

class DetectionUtils:
    """General detection utilities"""
    
    @staticmethod
    def scale_boxes(boxes: np.ndarray, 
                   original_shape: Tuple[int, int],
                   target_shape: Tuple[int, int],
                   pad_offset: Tuple[int, int] = (0, 0)) -> np.ndarray:
        """Scale bounding boxes from model input size to original image size
        Args:
            boxes: Array of boxes (N, 4)
            original_shape: Original image shape (height, width)
            target_shape: Model input shape (height, width)
            pad_offset: Padding offset (pad_w, pad_h)
        Returns:
            Scaled boxes
        """
        oh, ow = original_shape
        th, tw = target_shape
        pad_w, pad_h = pad_offset
        
        # Calculate scale
        scale = min(tw / ow, th / oh)
        
        boxes_scaled = boxes.copy()
        boxes_scaled[:, [0, 2]] = (boxes[:, [0, 2]] - pad_w) / scale
        boxes_scaled[:, [1, 3]] = (boxes[:, [1, 3]] - pad_h) / scale
        
        # Clip to image boundaries
        boxes_scaled[:, [0, 2]] = np.clip(boxes_scaled[:, [0, 2]], 0, ow)
        boxes_scaled[:, [1, 3]] = np.clip(boxes_scaled[:, [1, 3]], 0, oh)
        
        return boxes_scaled
    
    @staticmethod
    def calculate_iou(box1: Tuple[int, int, int, int],
                     box2: Tuple[int, int, int, int]) -> float:
        """Calculate IOU between two boxes
        Args:
            box1: First box (x1, y1, x2, y2)
            box2: Second box (x1, y1, x2, y2)
        Returns:
            IOU value
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union_area = box1_area + box2_area - inter_area
        
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area
    
    @staticmethod
    def filter_by_roi(detections: List, 
                      roi: Tuple[int, int, int, int]) -> List:
        """Filter detections by region of interest
        Args:
            detections: List of Detection objects
            roi: Region of interest (x1, y1, x2, y2)
        Returns:
            Filtered detections
        """
        from ..interfaces.core_interfaces import Detection
        
        x1, y1, x2, y2 = roi
        filtered = []
        
        for det in detections:
            bx1, by1, bx2, by2 = det.bbox
            
            # Check if box center is in ROI
            cx = (bx1 + bx2) / 2
            cy = (by1 + by2) / 2
            
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                filtered.append(det)
        
        return filtered