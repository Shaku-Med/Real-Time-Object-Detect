import cv2
import numpy as np
import time
from typing import List, Tuple, Optional

class LiveDetectionPipeline:
    def __init__(self, 
                 weights_path: str = "yolov4.weights",
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
            print("Using basic template - you'll need to download YOLO files")
            self.net = None
            self.classes = ["person", "car", "bicycle", "dog", "cat"]
        
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
        
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        self.video_writer = None
        self.recording = False
    
    def detect_objects(self, frame: np.ndarray) -> Tuple[List, List, List]:
        if self.net is None:
            return [], [], []
        
        height, width = frame.shape[:2]
        
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        
        outputs = self.net.forward(self.output_layers)
        
        boxes = []
        confidences = []
        class_ids = []
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > self.confidence_threshold:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    x = int(center_x - w/2)
                    y = int(center_y - h/2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        return boxes, confidences, class_ids
    
    def apply_nms(self, boxes: List, confidences: List, class_ids: List) -> List[int]:
        if not boxes:
            return []
        
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 
                                   self.confidence_threshold, 
                                   self.nms_threshold)
        
        return indices.flatten() if indices is not None else []
    
    def draw_detections(self, frame: np.ndarray, boxes: List, 
                       confidences: List, class_ids: List, indices: List[int]) -> np.ndarray:
        for i in indices:
            x, y, w, h = boxes[i]
            class_id = class_ids[i]
            confidence = confidences[i]
            
            if class_id < len(self.classes):
                label = self.classes[class_id]
                color = self.colors[class_id]
            else:
                label = f"Class {class_id}"
                color = (0, 255, 0)
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            label_text = f"{label}: {confidence:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x, y - label_h - 10), (x + label_w, y), color, -1)
            
            cv2.putText(frame, label_text, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return frame
    
    def update_fps(self):
        self.fps_counter += 1
        if self.fps_counter >= 30:
            end_time = time.time()
            self.current_fps = self.fps_counter / (end_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = end_time
    
    def draw_info(self, frame: np.ndarray, detection_count: int) -> np.ndarray:
        info_text = f"FPS: {self.current_fps:.1f} | Detections: {detection_count}"
        cv2.putText(frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        recording_text = "RECORDING" if self.recording else "NOT RECORDING"
        recording_color = (0, 0, 255) if self.recording else (255, 255, 255)
        cv2.putText(frame, recording_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, recording_color, 2)
        
        cv2.putText(frame, "Press 'q' to quit, 's' to save frame, 'r' to toggle recording", (10, frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def start_recording(self, width: int, height: int, fps: int = 30):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"detection_recording_{timestamp}.mp4"
        
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
    
    def run(self, camera_index: int = 0, window_name: str = "Live Object Detection"):
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
        
        print("Starting live detection...")
        print("Press 'q' to quit, 's' to save frame, 'r' to toggle recording")
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame")
                    break
                
                boxes, confidences, class_ids = self.detect_objects(frame)
                
                indices = self.apply_nms(boxes, confidences, class_ids)
                
                frame = self.draw_detections(frame, boxes, confidences, class_ids, indices)
                
                self.update_fps()
                frame = self.draw_info(frame, len(indices))
                
                if self.recording and self.video_writer:
                    self.video_writer.write(frame)
                
                cv2.imshow(window_name, frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"detection_frame_{timestamp}_{frame_count:04d}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"Saved frame as {filename}")
                elif key == ord('r'):
                    if self.recording:
                        self.stop_recording()
                    else:
                        self.start_recording(width, height, fps)
                
                frame_count += 1
                
        except KeyboardInterrupt:
            print("\nStopping detection...")
        
        finally:
            self.stop_recording()
            cap.release()
            cv2.destroyAllWindows()
            print("Pipeline stopped") 