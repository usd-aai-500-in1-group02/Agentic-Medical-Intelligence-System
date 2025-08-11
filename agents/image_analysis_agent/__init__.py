from .image_classifier import ImageClassifier
from .chest_xray_agent.covid_chest_xray_inference import ChestXRayClassification
from .brain_tumor_agent.brain_tumor_inference import BrainTumorDetection
from .skin_lesion_agent.skin_lesion_inference import SkinLesionSegmentation

class ImageAnalysisAgent:
    """
    Agent responsible for processing image uploads and classifying them as medical or non-medical, and determining their type.
    """
    
    def __init__(self, config):
        self.image_classifier = ImageClassifier(vision_model=config.medical_cv.llm)
        self.chest_xray_agent = ChestXRayClassification(model_path=config.medical_cv.chest_xray_model_path)
        self.brain_tumor_agent = BrainTumorDetection(model_path=config.medical_cv.brain_tumor_model_path)
        self.skin_lesion_agent = SkinLesionSegmentation(model_path=config.medical_cv.skin_lesion_model_path)
        self.skin_lesion_segmentation_output_path = config.medical_cv.skin_lesion_segmentation_output_path
        self.brain_tumor_detection_output_path = config.medical_cv.brain_tumor_detection_output_path
    
    # classify image
    def analyze_image(self, image_path: str) -> str:
        """Classifies images as medical or non-medical and determines their type."""
        return self.image_classifier.classify_image(image_path)
    
    # chest x-ray agent
    def classify_chest_xray(self, image_path: str) -> str:
        return self.chest_xray_agent.predict(image_path)
    
    # brain tumor agent
    def classify_brain_tumor(self, image_path: str) -> str:
        return self.brain_tumor_agent.predict(image_path)
    
    # brain tumor visualization
    def visualize_brain_tumor_detection(self, image_path: str) -> bool:
        return self.brain_tumor_agent.visualize_detections(image_path, self.brain_tumor_detection_output_path)
    
    # skin lesion agent
    def segment_skin_lesion(self, image_path: str) -> str:
        return self.skin_lesion_agent.predict(image_path, self.skin_lesion_segmentation_output_path)
