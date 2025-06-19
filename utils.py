import os
import urllib.request

def download_yolo_files():
    files_to_download = {
        "yolov4.weights": "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights",
        "yolov4.cfg": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg",
        "coco.names": "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"
    }
    
    print("Downloading YOLO model files...")
    
    for filename, url in files_to_download.items():
        if not os.path.exists(filename):
            print(f"Downloading {filename}...")
            try:
                urllib.request.urlretrieve(url, filename)
                print(f"✓ Downloaded {filename}")
            except Exception as e:
                print(f"✗ Error downloading {filename}: {e}")
        else:
            print(f"✓ {filename} already exists")
    
    print("\nDownload complete! You can now run the object detection.")

def check_yolo_files():
    required_files = ["yolov4.weights", "yolov4.cfg", "coco.names"]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"Missing YOLO files: {missing_files}")
        return False
    return True 