"""
TensorRT optimized detector for edge devices.
Specifically optimized for Jetson Nano.
"""

import time
import logging
import numpy as np
import cv2
from typing import List, Dict, Any, Optional, Tuple
from .object_detector import ObjectDetectorInterface
from .detection_utils import ImagePreprocessor, NMSProcessor, DetectionUtils
from ..interfaces.core_interfaces import Detection

logger = logging.getLogger(__name__)

class TensorRTDetector(ObjectDetectorInterface):
    """TensorRT-optimized detector for maximum performance"""
    
    def __init__(self):
        self.engine = None
        self.context = None
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = None
        
        # Configuration
        self.input_shape = (3, 640, 640)
        self.confidence_threshold = 0.5
        self.nms_threshold = 0.45
        self.class_names = []
        self.classes_of_interest = None
        
        # Performance
        self.inference_times = []
        self.total_frames = 0
        self.is_initialized = False
        
    def initialize(self, config: Dict) -> bool:
        """Initialize TensorRT detector"""
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit
            
            self.confidence_threshold = config.get('confidence_threshold', 0.5)
            self.nms_threshold = config.get('nms_threshold', 0.45)
            self.classes_of_interest = config.get('classes_of_interest', None)
            
            # Load engine
            engine_path = config.get('engine_path', 'models/yolov8.trt')
            logger.info(f"Loading TensorRT engine: {engine_path}")
            
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            
            with open(engine_path, 'rb') as f:
                runtime = trt.Runtime(TRT_LOGGER)
                self.engine = runtime.deserialize_cuda_engine(f.read())
            
            if self.engine is None:
                raise RuntimeError("Failed to deserialize engine")
            
            # Create execution context
            self.context = self.engine.create_execution_context()
            
            # Setup I/O bindings
            self._setup_bindings()
            
            # Load class names
            class_names_path = config.get('class_names_path', 'models/coco.names')
            self._load_class_names(class_names_path)
            
            self.is_initialized = True
            
            # Warmup
            self.warmup()
            
            logger.info("TensorRT detector initialized successfully")
            return True
            
        except ImportError as e:
            logger.error(f"TensorRT/PyCUDA not installed: {e}")
            return False
        except Exception as e:
            logger.error(f"TensorRT initialization failed: {e}")
            return False
    
    def _setup_bindings(self):
        """Setup input/output bindings"""
        import pycuda.driver as cuda
        
        self.inputs = []
        self.outputs = []
        self.bindings = []
        
        for binding in self.engine:
            binding_shape = self.engine.get_binding_shape(binding)
            size = trt.volume(binding_shape)
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            # Allocate memory
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            self.bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(binding):
                self.inputs.append({
                    'name': binding,
                    'host': host_mem,
                    'device': device_mem,
                    'shape': binding_shape,
                    'dtype': dtype
                })
            else:
                self.outputs.append({
                    'name': binding,
                    'host': host_mem,
                    'device': device_mem,
                    'shape': binding_shape,
                    'dtype': dtype
                })
        
        # Create CUDA stream
        self.stream = cuda.Stream()
    
    def _load_class_names(self, path: str):
        """Load class names from file"""
        try:
            with open(path, 'r') as f:
                self.class_names = [line.strip() for line in f.readlines()]
            logger.info(f"Loaded {len(self.class_names)} classes")
        except FileNotFoundError:
            logger.warning(f"Class names file not found: {path}")
            # COCO classes as default
            self.class_names = [
                'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                'scissors', 'teddy bear', 'hair drier', 'toothbrush'
            ]
    
    def detect(self, image: np.ndarray) -> List[Detection]:
        """Run detection on image"""
        if not self.is_initialized:
            raise RuntimeError("Detector not initialized")
        
        try:
            import pycuda.driver as cuda
            
            # Preprocess image
            input_data = self._preprocess(image)
            
            # Copy input to device
            np.copyto(self.inputs[0]['host'], input_data.ravel())
            cuda.memcpy_htod_async(
                self.inputs[0]['device'],
                self.inputs[0]['host'],
                self.stream
            )
            
            # Run inference
            start_time = time.time()
            self.context.execute_async_v2(self.bindings, self.stream.handle)
            
            # Copy outputs back
            for output in self.outputs:
                cuda.memcpy_dtoh_async(
                    output['host'],
                    output['device'],
                    self.stream
                )
            
            self.stream.synchronize()
            inference_time = time.time() - start_time
            
            # Track performance
            self.inference_times.append(inference_time * 1000)
            self.total_frames += 1
            
            # Postprocess
            detections = self._postprocess(image.shape[:2])
            
            return detections
            
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return []
    
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for inference"""
        # Resize with letterbox
        input_img, scale, pad = ImagePreprocessor.resize_and_pad(
            image,
            (self.input_shape[2], self.input_shape[1])
        )
        
        # BGR to RGB
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        
        # Normalize
        input_img = input_img.astype(np.float32) / 255.0
        
        # HWC to CHW
        input_img = np.transpose(input_img, (2, 0, 1))
        
        # Add batch dimension
        input_img = np.expand_dims(input_img, axis=0)
        
        # Ensure contiguous
        input_img = np.ascontiguousarray(input_img)
        
        return input_img
    
    def _postprocess(self, original_shape: Tuple[int, int]) -> List[Detection]:
        """Postprocess detection outputs"""
        detections = []
        
        # Get output data
        output_data = self.outputs[0]['host'].reshape(-1)
        
        # Parse based on model output format
        # This is model-specific - adjust for your model
        num_detections = int(output_data[0])
        
        for i in range(num_detections):
            idx = 1 + i * 7  # 7 values per detection
            
            if idx + 6 >= len(output_data):
                break
            
            # Parse detection
            class_id = int(output_data[idx + 1])
            confidence = float(output_data[idx + 2])
            
            if confidence < self.confidence_threshold:
                continue
            
            # Get box coordinates (normalized)
            x = float(output_data[idx + 3])
            y = float(output_data[idx + 4])
            w = float(output_data[idx + 5])
            h = float(output_data[idx + 6])
            
            # Convert to pixel coordinates
            img_h, img_w = original_shape
            x1 = int((x - w/2) * img_w)
            y1 = int((y - h/2) * img_h)
            x2 = int((x + w/2) * img_w)
            y2 = int((y + h/2) * img_h)
            
            # Clip to image bounds
            x1 = max(0, min(x1, img_w))
            y1 = max(0, min(y1, img_h))
            x2 = max(0, min(x2, img_w))
            y2 = max(0, min(y2, img_h))
            
            # Get class name
            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
            
            # Filter by class
            if self.classes_of_interest and class_name not in self.classes_of_interest:
                continue
            
            detections.append(Detection(
                class_name=class_name,
                confidence=confidence,
                bbox=(x1, y1, x2, y2),
                detection_time=time.time()
            ))
        
        return detections
    
    def get_supported_classes(self) -> List[str]:
        return self.class_names
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            'backend': 'tensorrt',
            'input_shape': self.input_shape,
            'num_classes': len(self.class_names),
            'avg_inference_time_ms': np.mean(self.inference_times[-100:]) if self.inference_times else 0,
            'total_frames': self.total_frames,
            'fps': 1000.0 / np.mean(self.inference_times[-100:]) if self.inference_times else 0
        }
    
    def warmup(self, num_iterations: int = 5) -> None:
        """Warm up with dummy inputs"""
        logger.info("Warming up TensorRT detector...")
        dummy = np.zeros((self.input_shape[2], self.input_shape[1], 3), dtype=np.uint8)
        for _ in range(num_iterations):
            self.detect(dummy)
        self.inference_times.clear()
    
    def shutdown(self) -> None:
        """Clean up"""
        if self.engine:
            del self.engine
            del self.context
        self.is_initialized = False
        logger.info("TensorRT detector shutdown")