from detection_pipeline import LiveDetectionPipeline
from advanced_detection import AdvancedDetectionPipeline
from enhanced_detection import EnhancedDetectionPipeline
from utils import download_yolo_files, check_yolo_files
from config import Config

def main():
    print("YOLO Object Detection System")
    print("=" * 40)
    print("1. Basic Detection (Original)")
    print("2. Advanced Detection (Temporal + Motion + Ensemble)")
    print("3. Enhanced Detection (Multi-scale + Adaptive + Quality)")
    print("=" * 40)
    
    while True:
        try:
            choice = input("Select detection mode (1-3): ").strip()
            
            if choice == "1":
                print("Starting Basic Detection...")
                if not check_yolo_files():
                    print("Downloading required YOLO files...")
                    download_yolo_files()
                
                pipeline = LiveDetectionPipeline(
                    weights_path=Config.YOLO_WEIGHTS_PATH,
                    config_path=Config.YOLO_CONFIG_PATH,
                    classes_path=Config.YOLO_CLASSES_PATH,
                    confidence_threshold=Config.CONFIDENCE_THRESHOLD,
                    nms_threshold=Config.NMS_THRESHOLD
                )
                pipeline.run(camera_index=Config.CAMERA_INDEX, window_name="Basic Detection")
                break
                
            elif choice == "2":
                print("Starting Advanced Detection...")
                if not check_yolo_files():
                    print("Downloading required YOLO files...")
                    download_yolo_files()
                
                pipeline = AdvancedDetectionPipeline(
                    weights_path=Config.YOLO_WEIGHTS_PATH,
                    config_path=Config.YOLO_CONFIG_PATH,
                    classes_path=Config.YOLO_CLASSES_PATH,
                    confidence_threshold=Config.CONFIDENCE_THRESHOLD,
                    nms_threshold=Config.NMS_THRESHOLD
                )
                pipeline.run_advanced(camera_index=Config.CAMERA_INDEX, window_name="Advanced Detection")
                break
                
            elif choice == "3":
                print("Starting Enhanced Detection...")
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
                pipeline.run_enhanced(camera_index=Config.CAMERA_INDEX, window_name="Enhanced Detection")
                break
                
            else:
                print("Invalid choice. Please select 1, 2, or 3.")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
            break

if __name__ == "__main__":
    main() 