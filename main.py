from enhanced_detection import EnhancedDetectionPipeline
from utils import download_yolo_files, check_yolo_files
from config import Config

def main():
    print("Initializing Enhanced Detection Pipeline...")
    
    if not check_yolo_files():
        print("Downloading required YOLO files...")
        download_yolo_files()
    
    pipeline = EnhancedDetectionPipeline(
        weights_path=Config.YOLO_WEIGHTS_PATH,
        config_path=Config.YOLO_CONFIG_PATH,
        classes_path=Config.YOLO_CLASSES_PATH,
        confidence_threshold=Config.CONFIDENCE_THRESHOLD,
        nms_threshold=Config.NMS_THRESHOLD
    )
    
    pipeline.run_enhanced(
        camera_index=Config.CAMERA_INDEX,
        window_name="Enhanced Object Detection"
    )

if __name__ == "__main__":
    main()