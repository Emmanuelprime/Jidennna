"""
YOLOv8/v11 detector implementation.
Supports both PyTorch and ONNX runtime.
"""

import time
import logging
import numpy as np
import cv2
from typing import List, Dict, Any, Optional
from .object_detector import ObjectDetectorInterface
from .detection_utils import ImagePreprocessor, NMSProcessor
from ..interfaces.core_interfaces import Detection

logger = logging.getLogger(__name__)

class YOLODetector(ObjectDetectorInterface):
    """YOLO-based object detector with multiple backend support"""
    
    def __init__(self):
        self.model = None
        self.backend = None  # 'pytorch', 'onnx', 'tensorrt'
        self.device = None
        self.input_size = (640, 640)
        self.confidence_threshold = 0.5
        self.nms_threshold = 0.45
        self.classes_of_interest = None
        self.class_names = []
        self.is_initialized = False
        
        # Performance tracking
        self.inference_times = []
        self.preprocess_times = []
        self.postprocess_times = []
        
    def initialize(self, config: Dict) -> bool:
        """Initialize YOLO detector"""
        try:
            self.confidence_threshold = config.get('confidence_threshold', 0.5)
            self.nms_threshold = config.get('nms_threshold', 0.45)
            self.input_size = config.get('input_size', (640, 640))
            self.classes_of_interest = config.get('classes_of_interest', None)
            
            # Determine backend
            use_tensorrt = config.get('use_tensorrt', False)
            
            if use_tensorrt:
                return self._initialize_tensorrt(config)
            else:
                return self._initialize_pytorch(config)
                
        except Exception as e:
            logger.error(f"YOLO initialization failed: {e}")
            return False
    
    def _initialize_pytorch(self, config: Dict) -> bool:
        """Initialize PyTorch backend"""
        try:
            import torch
            from ultralytics import YOLO
            
            # Determine device
            if config.get('device', 'auto') == 'auto':
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            else:
                self.device = config['device']
            
            # Load model
            model_path = config['model_path']
            logger.info(f"Loading YOLO model from {model_path}")
            self.model = YOLO(model_path)
            
            if self.device == 'cuda':
                self.model.to('cuda')
                logger.info("Model moved to CUDA")
            
            self.backend = 'pytorch'
            self.class_names = list(self.model.names.values())
            self.is_initialized = True
            
            # Warm up
            self.warmup(3)
            
            return True
            
        except ImportError:
            logger.error("PyTorch or Ultralytics not installed")
            return False
        except Exception as e:
            logger.error(f"PyTorch initialization failed: {e}")
            return False
    
    def _initialize_tensorrt(self, config: Dict) -> bool:
        """Initialize TensorRT backend"""
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit
            
            engine_path = config.get('engine_path')
            if not engine_path:
                logger.error("TensorRT engine path not provided")
                return False
            
            logger.info(f"Loading TensorRT engine from {engine_path}")
            
            # Load engine
            with open(engine_path, 'rb') as f:
                runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
                self.model = runtime.deserialize_cuda_engine(f.read())
            
            if self.model is None:
                raise RuntimeError("Failed to load TensorRT engine")
            
            self.context = self.model.create_execution_context()
            
            # Allocate buffers
            self._allocate_buffers()
            
            self.backend = 'tensorrt'
            self.is_initialized = True
            
            # Warm up
            self.warmup(3)
            
            return True
            
        except ImportError as e:
            logger.error(f"TensorRT dependencies not available: {e}")
            return False
        except Exception as e:
            logger.error(f"TensorRT initialization failed: {e}")
            return False
    
    def _allocate_buffers(self):
        """Allocate input/output buffers for TensorRT"""
        import pycuda.driver as cuda
        
        self.bindings = []
        self.input_binding = None
        self.output_bindings = []
        
        for binding in self.model:
            size = trt.volume(self.model.get_binding_shape(binding))
            dtype = trt.nptype(self.model.get_binding_dtype(binding))
            
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            self.bindings.append(int(device_mem))
            
            if self.model.binding_is_input(binding):
                self.input_binding = {
                    'name': binding,
                    'host': host_mem,
                    'device': device_mem,
                    'size': size,
                    'dtype': dtype
                }
            else:
                self.output_bindings.append({
                    'name': binding,
                    'host': host_mem,
                    'device': device_mem,
                    'size': size,
                    'dtype': dtype
                })
    
    def detect(self, image: np.ndarray) -> List[Detection]:
        """Detect objects in image"""
        if not self.is_initialized:
            raise RuntimeError("Detector not initialized")
        
        try:
            if self.backend == 'pytorch':
                return self._detect_pytorch(image)
            elif self.backend == 'tensorrt':
                return self._detect_tensorrt(image)
            else:
                raise RuntimeError(f"Unknown backend: {self.backend}")
                
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return []
    
    def _detect_pytorch(self, image: np.ndarray) -> List[Detection]:
        """PyTorch inference"""
        import torch
        
        # Run inference
        start_time = time.time()
        results = self.model(
            image,
            conf=self.confidence_threshold,
            iou=self.nms_threshold,
            verbose=False
        )
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time * 1000)
        
        # Parse results
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Get class info
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = self.class_names[class_id]
                    
                    # Filter classes
                    if self.classes_of_interest and \
                       class_name not in self.classes_of_interest:
                        continue
                    
                    detection = Detection(
                        class_name=class_name,
                        confidence=confidence,
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                        detection_time=time.time()
                    )
                    detections.append(detection)
        
        return detections
    
    def _detect_tensorrt(self, image: np.ndarray) -> List[Detection]:
        """TensorRT inference"""
        import pycuda.driver as cuda
        
        # Preprocess
        preprocess_start = time.time()
        input_tensor = self._preprocess_image(image)
        preprocess_time = time.time() - preprocess_start
        self.preprocess_times.append(preprocess_time * 1000)
        
        # Copy to device
        np.copyto(self.input_binding['host'], input_tensor.ravel())
        cuda.memcpy_htod(self.input_binding['device'], 
                        self.input_binding['host'])
        
        # Execute
        start_time = time.time()
        self.context.execute_v2(self.bindings)
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time * 1000)
        
        # Copy outputs back
        outputs = []
        for binding in self.output_bindings:
            cuda.memcpy_dtoh(binding['host'], binding['device'])
            outputs.append(binding['host'].copy())
        
        # Postprocess
        postprocess_start = time.time()
        detections = self._postprocess_outputs(outputs, image.shape)
        postprocess_time = time.time() - postprocess_start
        self.postprocess_times.append(postprocess_time * 1000)
        
        return detections
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for TensorRT"""
        # Resize and pad
        input_img, _, _ = ImagePreprocessor.resize_and_pad(
            image, self.input_size
        )
        
        # Normalize and convert to CHW
        input_img = input_img.astype(np.float32) / 255.0
        input_img = ImagePreprocessor.to_chw(input_img)
        input_img = ImagePreprocessor.add_batch_dim(input_img)
        
        return input_img
    
    def _postprocess_outputs(self, outputs: List[np.ndarray], 
                            original_shape: Tuple[int, int]) -> List[Detection]:
        """Postprocess TensorRT outputs"""
        # This needs to be adapted based on your TensorRT model output format
        # Typically: [num_detections, detection_classes, detection_scores, detection_boxes]
        
        detections = []
        
        if len(outputs) >= 4:
            num_detections = int(outputs[0][0])
            
            for i in range(num_detections):
                class_id = int(outputs[1][i])
                confidence = float(outputs[2][i])
                
                if confidence < self.confidence_threshold:
                    continue
                
                # Get box (normalized coordinates)
                y1, x1, y2, x2 = outputs[3][i]
                
                # Scale to original image
                h, w = original_shape[:2]
                x1 = int(x1 * w)
                y1 = int(y1 * h)
                x2 = int(x2 * w)
                y2 = int(y2 * h)
                
                class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                
                if self.classes_of_interest and \
                   class_name not in self.classes_of_interest:
                    continue
                
                detections.append(Detection(
                    class_name=class_name,
                    confidence=confidence,
                    bbox=(x1, y1, x2, y2),
                    detection_time=time.time()
                ))
        
        return detections
    
    def get_supported_classes(self) -> List[str]:
        """Get supported class names"""
        return self.class_names
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'backend': self.backend,
            'device': self.device,
            'input_size': self.input_size,
            'num_classes': len(self.class_names),
            'initialized': self.is_initialized,
            'avg_inference_time_ms': np.mean(self.inference_times[-100:]) if self.inference_times else 0
        }
    
    def warmup(self, num_iterations: int = 3) -> None:
        """Warm up detector"""
        logger.info(f"Warming up detector ({num_iterations} iterations)...")
        dummy_input = np.zeros((*self.input_size, 3), dtype=np.uint8)
        
        for i in range(num_iterations):
            _ = self.detect(dummy_input)
        
        # Clear statistics from warmup
        self.inference_times.clear()
        self.preprocess_times.clear()
        self.postprocess_times.clear()
        
        logger.info("Warmup complete")
    
    def shutdown(self) -> None:
        """Clean up resources"""
        if self.model:
            del self.model
            self.model = None
        self.is_initialized = False
        logger.info("Detector shutdown")