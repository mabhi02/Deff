from flask import Flask, Response, jsonify
from flask_cors import CORS
import cv2
import torch
import numpy as np
import threading
import time

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:5173"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"],
        "expose_headers": ["Content-Type"]
    }
})

# Global variables
detection_running = False
frame_buffer = None
detection_thread = None

class DetectionThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.running = True
        self.model = None
        self.cap = None

    def run(self):
        global frame_buffer

        try:
            # Load model
            print("Loading YOLOv5 model...")
            model_path = "./unified_military_dataset/best.pt"
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
            self.model.conf = 0.25  # Confidence threshold
            self.model.iou = 0.45   # NMS threshold
            if torch.cuda.is_available():
                self.model.cuda()

            # Initialize webcam
            print("Initializing webcam...")
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FPS, 30)  # Set to 30 FPS
            if not self.cap.isOpened():
                raise Exception("Failed to open webcam")

            # Define colors for different classes
            colors = {
                'military_vehicle': (0, 0, 255),    # Red
                'aircraft': (255, 0, 0),            # Blue
                'soldier': (0, 255, 0),             # Green
                'civilian': (255, 255, 0),          # Cyan
                'ordnance': (128, 0, 128)           # Purple
            }

            print("Starting detection loop...")
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    continue

                # Run detection
                results = self.model(frame)

                # Process detections
                for det in results.xyxy[0]:
                    if len(det) >= 6:
                        x1, y1, x2, y2, conf, cls = det[:6]
                        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                        class_name = self.model.names[int(cls)]
                        color = colors.get(class_name, (255, 255, 255))

                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                        # Add label with confidence score
                        label = f"{class_name} {conf:.2f}"
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.6
                        thickness = 2
                        
                        # Add label background
                        (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
                        cv2.rectangle(frame, (x1, y1-text_height-5), (x1+text_width, y1), color, -1)
                        
                        # Add text
                        cv2.putText(frame, label, (x1, y1-5), font, font_scale, (255, 255, 255), thickness)

                # Add FPS counter
                fps = int(self.cap.get(cv2.CAP_PROP_FPS))
                cv2.putText(frame, f'FPS: {fps}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Update frame buffer
                _, buffer = cv2.imencode('.jpg', frame)
                frame_buffer = buffer.tobytes()

                # Small delay to maintain frame rate
                time.sleep(1/30)

        except Exception as e:
            print(f"Error in detection thread: {e}")
        finally:
            if self.cap is not None:
                self.cap.release()
            print("Detection thread stopped")

    def stop(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()

def gen_frames():
    while True:
        if frame_buffer is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_buffer + b'\r\n')
        time.sleep(1/30)

@app.route('/start_detection', methods=['POST'])
def start_detection():
    global detection_running, detection_thread
    
    if not detection_running:
        print("Starting detection thread...")
        detection_running = True
        detection_thread = DetectionThread()
        detection_thread.start()
        return jsonify({"status": "started"})
    return jsonify({"status": "already running"})

@app.route('/stop_detection', methods=['POST'])
def stop_detection():
    global detection_running, detection_thread
    
    if detection_running:
        print("Stopping detection thread...")
        detection_running = False
        if detection_thread:
            detection_thread.stop()
            detection_thread.join()
            detection_thread = None
        return jsonify({"status": "stopped"})
    return jsonify({"status": "not running"})

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    try:
        print("Starting Flask server...")
        app.run(host='0.0.0.0', port=5000, threaded=True)
    except Exception as e:
        print(f"Server error: {e}")