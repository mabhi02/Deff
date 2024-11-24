import torch
import cv2
import numpy as np
from typing import List

class MilitaryDetector:
    def __init__(self, model_path: str, conf_threshold: float = 0.25):
        """Initialize the military object detector."""
        # Load model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model using torch hub
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        self.model.conf = conf_threshold
        self.model.to(self.device)
        
        # Class names - make sure these match your trained model's classes
        self.class_names = ['military_vehicle', 'aircraft', 'soldier', 'civilian', 'ordnance']
        
        # Colors (BGR)
        self.colors = {
            'military_vehicle': (0, 0, 255),
            'aircraft': (255, 0, 0),
            'soldier': (0, 255, 0),
            'civilian': (0, 255, 255),
            'ordnance': (255, 0, 255)
        }

    def detect(self, image_path: str, save_path: str) -> List:
        """Detect objects in an image and save the annotated result."""
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
            
            # Run inference
            results = self.model(image)
            
            # Process results
            detections = []
            annotated_image = image.copy()
            
            # Get pandas DataFrame of results
            df = results.pandas().xyxy[0]
            
            if len(df) == 0:
                print("No detections found in the image.")
                cv2.imwrite(save_path, annotated_image)  # Save original image
                return []
            
            for _, det in df.iterrows():
                try:
                    x1, y1, x2, y2 = map(int, [det['xmin'], det['ymin'], det['xmax'], det['ymax']])
                    conf = float(det['confidence'])
                    class_idx = int(det['class'])
                    
                    # Verify class index is valid
                    if class_idx < 0 or class_idx >= len(self.class_names):
                        print(f"Warning: Invalid class index {class_idx}")
                        continue
                        
                    class_name = self.class_names[class_idx]
                    color = self.colors[class_name]
                    
                    # Draw box
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                    
                    # Add label
                    label = f"{class_name} {conf:.2f}"
                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cv2.rectangle(annotated_image, (x1, y1 - h - 5), (x1 + w + 5, y1), color, -1)
                    cv2.putText(annotated_image, label, (x1 + 3, y1 - 3), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    
                    # Store detection
                    detections.append({
                        'class': class_name,
                        'confidence': conf,
                        'bbox': [x1, y1, x2, y2]
                    })
                except Exception as e:
                    print(f"Warning: Error processing detection: {e}")
                    continue
            
            # Save the annotated image
            cv2.imwrite(save_path, annotated_image)
            print(f"Saved annotated image to: {save_path}")
            
            # Print detections
            if detections:
                print("\nDetections found:")
                for det in detections:
                    print(f"Class: {det['class']} | Confidence: {det['confidence']:.2f} | "
                          f"Box: {det['bbox']}")
            else:
                print("No valid detections found in the image.")
            
            return detections
            
        except Exception as e:
            print(f"Error during detection: {str(e)}")
            return []

if __name__ == "__main__":
    try:
        # Initialize detector
        detector = MilitaryDetector(
            model_path='unified_military_dataset/best.pt',
            conf_threshold=0.25  # You might want to lower this if no detections are found
        )
        
        # Run detection
        detections = detector.detect(
            image_path="army-tankBoxes.jpg",
            save_path="boxesCheck.jpg"
        )
        
        # Print summary
        print(f"\nProcessing complete. Found {len(detections)} detections.")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print(f"Full error details: ", e.__class__.__name__)