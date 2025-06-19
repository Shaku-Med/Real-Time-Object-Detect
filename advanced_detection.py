import cv2
import numpy as np
import time
from typing import List, Tuple, Optional, Dict, Any
from collections import deque
import threading
from dataclasses import dataclass

@dataclass
class Detection:
    bbox: List[int]
    confidence: float
    class_id: int
    class_name: str
    timestamp: float

class TemporalFilter:
    def __init__(self, max_history: int = 10, stability_threshold: float = 0.3):
        self.max_history = max_history
        self.stability_threshold = stability_threshold
        self.detection_history = deque(maxlen=max_history)
        self.stable_detections = {}
    
    def update(self, detections: List[Detection]) -> List[Detection]:
        current_time = time.time()
        self.detection_history.append((current_time, detections))
        
        if len(self.detection_history) < 3:
            return detections
        
        stable_detections = []
        for detection in detections:
            stability_score = self._calculate_stability(detection)
            if stability_score >= self.stability_threshold:
                stable_detections.append(detection)
        
        return stable_detections
    
    def _calculate_stability(self, detection: Detection) -> float:
        recent_detections = []
        for timestamp, dets in self.detection_history:
            for det in dets:
                if self._is_same_detection(detection, det):
                    recent_detections.append(det.confidence)
        
        if len(recent_detections) < 2:
            return 0.0
        
        return np.mean(recent_detections)

    def _is_same_detection(self, det1: Detection, det2: Detection) -> bool:
        if det1.class_id != det2.class_id:
            return False
        
        bbox1, bbox2 = det1.bbox, det2.bbox
        iou = self._calculate_iou(bbox1, bbox2)
        return iou > 0.5
    
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

class MotionDetector:
    def __init__(self, motion_threshold: float = 0.01):
        self.motion_threshold = motion_threshold
        self.previous_frame = None
        self.motion_mask = None
    
    def detect_motion(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if self.previous_frame is None:
            self.previous_frame = gray
            return np.ones_like(gray)
        
        frame_delta = cv2.absdiff(self.previous_frame, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        
        thresh = cv2.dilate(thresh, None, iterations=2)
        self.motion_mask = thresh
        self.previous_frame = gray
        
        return thresh
    
    def is_motion_in_region(self, bbox: List[int]) -> bool:
        if self.motion_mask is None:
            return True
        
        x, y, w, h = bbox
        roi = self.motion_mask[y:y+h, x:x+w]
        
        if roi.size == 0:
            return True
        
        motion_ratio = np.sum(roi > 0) / roi.size
        return motion_ratio > self.motion_threshold

class EnsembleDetector:
    def __init__(self, confidence_thresholds: List[float] = None):
        self.confidence_thresholds = confidence_thresholds or [0.3, 0.5, 0.7]
        self.detection_cache = {}
    
    def ensemble_detect(self, net: cv2.dnn.Net, frame: np.ndarray, 
                       output_layers: List[str], classes: List[str]) -> List[Detection]:
        all_detections = []
        
        for threshold in self.confidence_thresholds:
            detections = self._detect_with_threshold(net, frame, output_layers, classes, threshold)
            all_detections.extend(detections)
        
        return self._merge_ensemble_detections(all_detections)
    
    def _detect_with_threshold(self, net: cv2.dnn.Net, frame: np.ndarray, 
                              output_layers: List[str], classes: List[str], 
                              threshold: float) -> List[Detection]:
        height, width = frame.shape[:2]
        
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        outputs = net.forward(output_layers)
        
        detections = []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > threshold:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
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
    
    def _merge_ensemble_detections(self, detections: List[Detection]) -> List[Detection]:
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
        return iou > 0.3
    
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

class AdvancedNMS:
    def __init__(self, nms_threshold: float = 0.4, class_agnostic: bool = False):
        self.nms_threshold = nms_threshold
        self.class_agnostic = class_agnostic
    
    def apply_advanced_nms(self, detections: List[Detection]) -> List[Detection]:
        if not detections:
            return []
        
        if self.class_agnostic:
            return self._class_agnostic_nms(detections)
        else:
            return self._class_aware_nms(detections)
    
    def _class_aware_nms(self, detections: List[Detection]) -> List[Detection]:
        class_groups = {}
        for det in detections:
            if det.class_id not in class_groups:
                class_groups[det.class_id] = []
            class_groups[det.class_id].append(det)
        
        final_detections = []
        for class_id, class_dets in class_groups.items():
            filtered = self._apply_nms_to_class(class_dets)
            final_detections.extend(filtered)
        
        return final_detections
    
    def _class_agnostic_nms(self, detections: List[Detection]) -> List[Detection]:
        return self._apply_nms_to_class(detections)
    
    def _apply_nms_to_class(self, detections: List[Detection]) -> List[Detection]:
        if len(detections) <= 1:
            return detections
        
        boxes = [det.bbox for det in detections]
        confidences = [det.confidence for det in detections]
        
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.0, self.nms_threshold)
        
        if indices is None:
            return []
        
        return [detections[i] for i in indices.flatten()]

class ConfidenceCalibrator:
    def __init__(self, calibration_window: int = 100):
        self.calibration_window = calibration_window
        self.confidence_history = deque(maxlen=calibration_window)
        self.class_confidence_stats = {}
    
    def calibrate_confidence(self, detections: List[Detection]) -> List[Detection]:
        for det in detections:
            self.confidence_history.append(det.confidence)
            
            if det.class_id not in self.class_confidence_stats:
                self.class_confidence_stats[det.class_id] = {
                    'mean': det.confidence,
                    'std': 0.0,
                    'count': 1
                }
            else:
                stats = self.class_confidence_stats[det.class_id]
                stats['count'] += 1
                old_mean = stats['mean']
                stats['mean'] = (stats['mean'] * (stats['count'] - 1) + det.confidence) / stats['count']
                stats['std'] = np.sqrt(((stats['count'] - 1) * stats['std']**2 + 
                                      (det.confidence - old_mean) * (det.confidence - stats['mean'])) / stats['count'])
        
        calibrated_detections = []
        for det in detections:
            if det.class_id in self.class_confidence_stats:
                stats = self.class_confidence_stats[det.class_id]
                if stats['count'] > 10:
                    z_score = abs(det.confidence - stats['mean']) / (stats['std'] + 1e-8)
                    if z_score < 2.0:
                        calibrated_detections.append(det)
                else:
                    calibrated_detections.append(det)
            else:
                calibrated_detections.append(det)
        
        return calibrated_detections

class AdvancedDetectionPipeline:
    def __init__(self, weights_path: str = "yolov4.weights",
                 config_path: str = "yolov4.cfg",
                 classes_path: str = "coco.names",
                 confidence_threshold: float = 0.5,
                 nms_threshold: float = 0.4):
        
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
        
        self.temporal_filter = TemporalFilter(max_history=15, stability_threshold=0.4)
        self.motion_detector = MotionDetector(motion_threshold=0.005)
        self.ensemble_detector = EnsembleDetector(confidence_thresholds=[0.3, 0.5, 0.7])
        self.advanced_nms = AdvancedNMS(nms_threshold=nms_threshold, class_agnostic=False)
        self.confidence_calibrator = ConfidenceCalibrator(calibration_window=200)
        
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        self.video_writer = None
        self.recording = False
    
    def detect_objects_advanced(self, frame: np.ndarray) -> List[Detection]:
        if self.net is None:
            return []
        
        motion_mask = self.motion_detector.detect_motion(frame)
        
        detections = self.ensemble_detector.ensemble_detect(
            self.net, frame, self.output_layers, self.classes
        )
        
        motion_filtered = []
        for det in detections:
            if self.motion_detector.is_motion_in_region(det.bbox):
                motion_filtered.append(det)
        
        calibrated = self.confidence_calibrator.calibrate_confidence(motion_filtered)
        
        nms_filtered = self.advanced_nms.apply_advanced_nms(calibrated)
        
        temporal_filtered = self.temporal_filter.update(nms_filtered)
        
        return [det for det in temporal_filtered if det.confidence > self.confidence_threshold]
    
    def draw_advanced_detections(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
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
    
    def draw_advanced_info(self, frame: np.ndarray, detection_count: int) -> np.ndarray:
        info_text = f"Advanced FPS: {self.current_fps:.1f} | Detections: {detection_count}"
        cv2.putText(frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        recording_text = "RECORDING" if self.recording else "NOT RECORDING"
        recording_color = (0, 0, 255) if self.recording else (255, 255, 255)
        cv2.putText(frame, recording_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, recording_color, 2)
        
        cv2.putText(frame, "Advanced Detection Active - Press 'q' to quit", (10, frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def start_recording(self, width: int, height: int, fps: int = 30):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"advanced_detection_{timestamp}.mp4"
        
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
    
    def run_advanced(self, camera_index: int = 0, window_name: str = "Advanced Object Detection"):
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
        
        print("Starting advanced detection pipeline...")
        print("Press 'q' to quit, 'r' to toggle recording")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame")
                    break
                
                detections = self.detect_objects_advanced(frame)
                
                frame = self.draw_advanced_detections(frame, detections)
                
                self.update_fps()
                frame = self.draw_advanced_info(frame, len(detections))
                
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
            print("\nStopping advanced detection...")
        
        finally:
            self.stop_recording()
            cap.release()
            cv2.destroyAllWindows()
            print("Advanced detection pipeline stopped") 