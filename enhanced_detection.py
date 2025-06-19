import cv2
import numpy as np
import time
from typing import List, Tuple, Optional, Dict, Any
from collections import deque
import threading
from dataclasses import dataclass
from advanced_detection import Detection, TemporalFilter, MotionDetector, EnsembleDetector, AdvancedNMS, ConfidenceCalibrator

class MultiScaleDetector:
    def __init__(self, scales: List[float] = None):
        self.scales = scales or [0.8, 1.0, 1.2]
        self.detection_cache = {}
    
    def detect_multi_scale(self, net: cv2.dnn.Net, frame: np.ndarray, 
                          output_layers: List[str], classes: List[str]) -> List[Detection]:
        all_detections = []
        height, width = frame.shape[:2]
        
        for scale in self.scales:
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            resized = cv2.resize(frame, (new_width, new_height))
            detections = self._detect_at_scale(net, resized, output_layers, classes, scale, width, height)
            all_detections.extend(detections)
        
        return self._merge_multi_scale_detections(all_detections)
    
    def _detect_at_scale(self, net: cv2.dnn.Net, frame: np.ndarray, 
                        output_layers: List[str], classes: List[str], 
                        scale: float, original_width: int, original_height: int) -> List[Detection]:
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        outputs = net.forward(output_layers)
        
        detections = []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > 0.3:
                    center_x = int(detection[0] * original_width)
                    center_y = int(detection[1] * original_height)
                    w = int(detection[2] * original_width)
                    h = int(detection[3] * original_height)
                    
                    x = int(center_x - w/2)
                    y = int(center_y - h/2)
                    
                    detections.append(Detection(
                        bbox=[x, y, w, h],
                        confidence=float(confidence),
                        class_id=int(class_id),
                        class_name=classes[class_id] if class_id < len(classes) else f"Class_{class_id}",
                        timestamp=time.time()
                    ))
        
        return detections
    
    def _merge_multi_scale_detections(self, detections: List[Detection]) -> List[Detection]:
        if not detections:
            return []
        
        merged = []
        used = set()
        
        for i, det1 in enumerate(detections):
            if i in used:
                continue
            
            similar_detections = [det1]
            used.add(i)
            
            for j, det2 in enumerate(detections[i+1:], i+1):
                if j in used:
                    continue
                
                if self._are_similar(det1, det2):
                    similar_detections.append(det2)
                    used.add(j)
            
            if similar_detections:
                merged.append(self._merge_similar_detections(similar_detections))
        
        return merged
    
    def _are_similar(self, det1: Detection, det2: Detection) -> bool:
        if det1.class_id != det2.class_id:
            return False
        
        iou = self._calculate_iou(det1.bbox, det2.bbox)
        return iou > 0.4
    
    def _merge_similar_detections(self, detections: List[Detection]) -> Detection:
        if len(detections) == 1:
            return detections[0]
        
        avg_bbox = np.mean([det.bbox for det in detections], axis=0).astype(int)
        avg_confidence = np.mean([det.confidence for det in detections])
        
        return Detection(
            bbox=avg_bbox.tolist(),
            confidence=avg_confidence,
            class_id=detections[0].class_id,
            class_name=detections[0].class_name,
            timestamp=time.time()
        )
    
    def _calculate_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        union = w1 * h1 + w2 * h2 - intersection
        
        return intersection / union if union > 0 else 0.0

class AdaptiveThresholdManager:
    def __init__(self, base_threshold: float = 0.5, adaptation_rate: float = 0.1):
        self.base_threshold = base_threshold
        self.adaptation_rate = adaptation_rate
        self.current_threshold = base_threshold
        self.detection_history = deque(maxlen=50)
        self.false_positive_history = deque(maxlen=50)
    
    def update_threshold(self, detections: List[Detection], frame_quality: float):
        self.detection_history.append(len(detections))
        
        if len(self.detection_history) < 10:
            return
        
        recent_detections = list(self.detection_history)[-10:]
        avg_detections = np.mean(recent_detections)
        
        if avg_detections > 10:
            self.current_threshold = min(0.8, self.current_threshold + self.adaptation_rate)
        elif avg_detections < 2:
            self.current_threshold = max(0.3, self.current_threshold - self.adaptation_rate)
        
        self.current_threshold = self.current_threshold * frame_quality + self.base_threshold * (1 - frame_quality)
    
    def get_current_threshold(self) -> float:
        return self.current_threshold

class FrameQualityAnalyzer:
    def __init__(self):
        self.previous_frame = None
    
    def analyze_quality(self, frame: np.ndarray) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.previous_frame is None:
            self.previous_frame = gray
            return 1.0
        
        blur_score = self._calculate_blur_score(gray)
        noise_score = self._calculate_noise_score(gray)
        motion_score = self._calculate_motion_score(gray)
        
        self.previous_frame = gray
        
        quality = (blur_score + noise_score + motion_score) / 3.0
        return max(0.1, min(1.0, quality))
    
    def _calculate_blur_score(self, gray: np.ndarray) -> float:
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return min(1.0, laplacian_var / 500.0)
    
    def _calculate_noise_score(self, gray: np.ndarray) -> float:
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        noise = cv2.filter2D(gray, -1, kernel)
        noise_var = noise.var()
        return max(0.1, 1.0 - (noise_var / 1000.0))
    
    def _calculate_motion_score(self, gray: np.ndarray) -> float:
        if self.previous_frame is None:
            return 1.0
        
        diff = cv2.absdiff(gray, self.previous_frame)
        motion_score = 1.0 - (diff.mean() / 255.0)
        return max(0.1, motion_score)

class EnhancedDetectionPipeline:
    def __init__(self, weights_path: str = "yolov4.weights",
                 config_path: str = "yolov4.cfg",
                 classes_path: str = "coco.names",
                 confidence_threshold: float = 0.6,
                 nms_threshold: float = 0.3):
        
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        
        try:
            self.net = cv2.dnn.readNet(weights_path, config_path)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            self.output_layers = self.net.getUnconnectedOutLayersNames()
            
            with open(classes_path, 'r') as f:
                self.classes = [line.strip() for line in f.readlines()]
            
            print(f"Loaded YOLO model with {len(self.classes)} classes")
            
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            self.net = None
            self.classes = ["person", "car", "bicycle", "dog", "cat"]
        
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
        
        self.temporal_filter = TemporalFilter(max_history=20, stability_threshold=0.5)
        self.motion_detector = MotionDetector(motion_threshold=0.003)
        self.ensemble_detector = EnsembleDetector(confidence_thresholds=[0.2, 0.4, 0.6, 0.8])
        self.multi_scale_detector = MultiScaleDetector(scales=[0.7, 1.0, 1.3])
        self.advanced_nms = AdvancedNMS(nms_threshold=nms_threshold, class_agnostic=False)
        self.confidence_calibrator = ConfidenceCalibrator(calibration_window=300)
        self.adaptive_threshold = AdaptiveThresholdManager(base_threshold=0.6, adaptation_rate=0.05)
        self.quality_analyzer = FrameQualityAnalyzer()
        
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        self.video_writer = None
        self.recording = False
    
    def detect_objects_enhanced(self, frame: np.ndarray) -> List[Detection]:
        if self.net is None:
            return []
        
        frame_quality = self.quality_analyzer.analyze_quality(frame)
        self.adaptive_threshold.update_threshold([], frame_quality)
        
        motion_mask = self.motion_detector.detect_motion(frame)
        
        ensemble_detections = self.ensemble_detector.ensemble_detect(
            self.net, frame, self.output_layers, self.classes
        )
        
        multi_scale_detections = self.multi_scale_detector.detect_multi_scale(
            self.net, frame, self.output_layers, self.classes
        )
        
        all_detections = ensemble_detections + multi_scale_detections
        
        motion_filtered = []
        for det in all_detections:
            if self.motion_detector.is_motion_in_region(det.bbox):
                motion_filtered.append(det)
        
        calibrated = self.confidence_calibrator.calibrate_confidence(motion_filtered)
        
        current_threshold = self.adaptive_threshold.get_current_threshold()
        threshold_filtered = [det for det in calibrated if det.confidence > current_threshold]
        
        nms_filtered = self.advanced_nms.apply_advanced_nms(threshold_filtered)
        
        temporal_filtered = self.temporal_filter.update(nms_filtered)
        
        return temporal_filtered
    
    def draw_enhanced_detections(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        for det in detections:
            x, y, w, h = det.bbox
            class_id = det.class_id
            confidence = det.confidence
            
            if class_id < len(self.classes):
                label = det.class_name
                color = self.colors[class_id]
            else:
                label = f"Class {class_id}"
                color = (0, 255, 0)
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            label_text = f"{label}: {confidence:.3f}"
            (label_w, label_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x, y - label_h - 10), (x + label_w, y), color, -1)
            cv2.putText(frame, label_text, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return frame
    
    def update_fps(self):
        self.fps_counter += 1
        if self.fps_counter >= 30:
            end_time = time.time()
            self.current_fps = self.fps_counter / (end_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = end_time
    
    def draw_enhanced_info(self, frame: np.ndarray, detection_count: int) -> np.ndarray:
        current_threshold = self.adaptive_threshold.get_current_threshold()
        
        info_text = f"Enhanced FPS: {self.current_fps:.1f} | Detections: {detection_count}"
        cv2.putText(frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        threshold_text = f"Adaptive Threshold: {current_threshold:.3f}"
        cv2.putText(frame, threshold_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        recording_text = "RECORDING" if self.recording else "NOT RECORDING"
        recording_color = (0, 0, 255) if self.recording else (255, 255, 255)
        cv2.putText(frame, recording_text, (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, recording_color, 2)
        
        cv2.putText(frame, "Enhanced Detection Active - Press 'q' to quit", (10, frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def start_recording(self, width: int, height: int, fps: int = 30):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"enhanced_detection_{timestamp}.mp4"
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
        self.recording = True
        print(f"Started recording: {filename}")
    
    def stop_recording(self):
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            self.recording = False
            print("Recording stopped")
    
    def run_enhanced(self, camera_index: int = 0, window_name: str = "Enhanced Object Detection"):
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_index}")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        print("Starting enhanced detection pipeline...")
        print("Press 'q' to quit, 'r' to toggle recording")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame")
                    break
                
                detections = self.detect_objects_enhanced(frame)
                
                frame = self.draw_enhanced_detections(frame, detections)
                
                self.update_fps()
                frame = self.draw_enhanced_info(frame, len(detections))
                
                if self.recording and self.video_writer:
                    self.video_writer.write(frame)
                
                cv2.imshow(window_name, frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    if self.recording:
                        self.stop_recording()
                    else:
                        self.start_recording(width, height, fps)
                
        except KeyboardInterrupt:
            print("\nStopping enhanced detection...")
        
        finally:
            self.stop_recording()
            cap.release()
            cv2.destroyAllWindows()
            print("Enhanced detection pipeline stopped") 