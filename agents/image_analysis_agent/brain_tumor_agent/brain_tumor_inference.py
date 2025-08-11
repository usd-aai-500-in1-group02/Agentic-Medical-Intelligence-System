import os
import logging
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

class BrainTumorDetection:
    """Handles brain tumor detection using a trained YOLO model."""
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.device = DEVICE
        self.model = self._load_model()
        
    def _load_model(self):
        """Load the trained YOLO model."""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            # Load YOLO model
            model = YOLO(self.model_path)
            logger.info(f"Brain tumor detection model loaded successfully from {self.model_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading brain tumor model: {e}")
            raise e
    
    def predict(self, image_path, conf_threshold=0.5):
        """Detect brain tumors in MRI image and return classification result."""
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # Run inference
            results = self.model(image_path, conf=conf_threshold, verbose=False)
            
            # Parse results
            detections = []
            has_tumor = False
            
            for result in results:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    has_tumor = True
                    for box in boxes:
                        detection = {
                            "class": int(box.cls[0]),
                            "class_name": self.model.names[int(box.cls[0])],
                            "confidence": float(box.conf[0]),
                            "bbox": box.xyxy[0].tolist(),  # [x1, y1, x2, y2]
                        }
                        detections.append(detection)
            
            # Return simple classification result similar to other agents
            if has_tumor:
                result_class = "tumor_detected"
                logger.info(f"Brain tumor detected in {image_path} with {len(detections)} detection(s)")
            else:
                result_class = "no_tumor"
                logger.info(f"No brain tumor detected in {image_path}")
            
            return result_class
            
        except Exception as e:
            logger.error(f"Error during brain tumor detection: {e}")
            return "error"
    
    def visualize_detections(self, image_path, output_path, conf_threshold=0.5):
        """Generate visualization with bounding boxes around detected tumors for UI display."""
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Run inference
            results = self.model(image_path, conf=conf_threshold, verbose=False)
            
            # Check if any detections were made
            has_detections = False
            for result in results:
                if result.boxes is not None and len(result.boxes) > 0:
                    has_detections = True
                    break
            
            if has_detections:
                # Save annotated image with bounding boxes using YOLO's built-in visualization
                annotated_result = results[0].plot()
                cv2.imwrite(output_path, annotated_result)
                logger.info(f"Brain tumor detection visualization saved to {output_path}")
                return True
            else:
                # If no detections, save original image with "No Tumor Detected" text overlay
                import matplotlib.pyplot as plt
                import matplotlib.patches as patches
                
                # Read and display original image
                img = cv2.imread(image_path)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                fig, ax = plt.subplots(figsize=(10, 8))
                ax.imshow(img_rgb)
                ax.axis('off')
                
                # Add "No Tumor Detected" text overlay
                ax.text(0.5, 0.95, 'NO TUMOR DETECTED', 
                       transform=ax.transAxes, fontsize=16, fontweight='bold',
                       ha='center', va='top', 
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='green', alpha=0.7, edgecolor='darkgreen'),
                       color='white')
                
                plt.savefig(output_path, bbox_inches='tight', dpi=150, facecolor='white')
                plt.close()
                logger.info(f"Brain tumor 'no detection' visualization saved to {output_path}")
                return True
                
        except Exception as e:
            logger.error(f"Error generating brain tumor visualization: {e}")
            return False
    
    def get_detailed_results(self, image_path, conf_threshold=0.5):
        """Get detailed detection results with bounding boxes and confidence scores."""
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # Run inference
            results = self.model(image_path, conf=conf_threshold, verbose=False)
            
            # Parse detailed results
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        detection = {
                            "class": int(box.cls[0]),
                            "class_name": self.model.names[int(box.cls[0])],
                            "confidence": float(box.conf[0]),
                            "bbox": box.xyxy[0].tolist(),  # [x1, y1, x2, y2]
                            "center": [(box.xyxy[0][0] + box.xyxy[0][2])/2,
                                     (box.xyxy[0][1] + box.xyxy[0][3])/2]
                        }
                        detections.append(detection)
            
            return {
                "detections": detections,
                "num_tumors": len(detections),
                "image_path": image_path,
                "model": "YOLO-Brain-Tumor"
            }
            
        except Exception as e:
            logger.error(f"Error getting detailed brain tumor results: {e}")
            return {"error": str(e)}


# # Example Usage
# if __name__ == "__main__":
#     detector = BrainTumorDetection(model_path="./models/brain_tumour_od.pt")
#     result = detector.predict("./images/brain_mri_sample.jpg")
#     logger.info(f"Detection result: {result}")