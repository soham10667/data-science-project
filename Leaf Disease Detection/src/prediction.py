import os
import json
import numpy as np
from src.preprocessing import load_and_preprocess_image, validate_image
from src.disease_info import get_disease_details, DISCLAIMER_TEXT

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
PYTORCH_MODEL_PATH = os.path.join(MODEL_DIR, "leaf_disease_model.pt")
KERAS_MODEL_PATH = os.path.join(MODEL_DIR, "leaf_disease_model.keras")
DEFAULT_CLASS_PATH = os.path.join(MODEL_DIR, "class_names.json")

class LeafDiseasePredictor:
    """
    Inference engine for Leaf Disease Classification Model.
    Supports PyTorch (.pt) and TensorFlow/Keras (.keras) models.
    """
    def __init__(self, class_path=DEFAULT_CLASS_PATH):
        self.class_path = class_path
        self.backend = None
        self.model = None
        self.class_names = []
        self._load_resources()

    def _load_resources(self):
        """Loads model weights and categorical class labels."""
        # Load Class Mapping
        if os.path.exists(self.class_path):
            with open(self.class_path, "r") as f:
                self.class_names = json.load(f)
        else:
            self.class_names = [
                "Apple_Black_Rot", "Apple_Healthy", "Apple_Scab",
                "Corn_Common_Rust", "Corn_Healthy",
                "Potato_Early_Blight", "Potato_Healthy", "Potato_Late_Blight",
                "Tomato_Bacterial_Spot", "Tomato_Early_Blight", "Tomato_Healthy",
                "Tomato_Late_Blight", "Tomato_Leaf_Spot", "Tomato_Powdery_Mildew"
            ]

        # 1. Check PyTorch Model
        if os.path.exists(PYTORCH_MODEL_PATH):
            try:
                import torch
                import torch.nn as nn
                from torchvision import models

                num_classes = len(self.class_names)
                
                # Build MobileNetV2 architecture matching train.py
                try:
                    try:
                        weights = models.MobileNet_V2_Weights.DEFAULT
                        model = models.mobilenet_v2(weights=weights)
                    except AttributeError:
                        model = models.mobilenet_v2(pretrained=False)
                    
                    in_features = model.classifier[1].in_features
                    model.classifier = nn.Sequential(
                        nn.Linear(in_features, 256),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(256, num_classes)
                    )
                except Exception:
                    # Custom CNN fallback
                    class CustomCNN(nn.Module):
                        def __init__(self, num_classes):
                            super().__init__()
                            self.features = nn.Sequential(
                                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                                nn.BatchNorm2d(32),
                                nn.ReLU(),
                                nn.MaxPool2d(2, 2),
                                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                                nn.BatchNorm2d(64),
                                nn.ReLU(),
                                nn.MaxPool2d(2, 2),
                                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                                nn.BatchNorm2d(128),
                                nn.ReLU(),
                                nn.AdaptiveAvgPool2d((1, 1))
                            )
                            self.classifier = nn.Sequential(
                                nn.Linear(128, 128),
                                nn.ReLU(),
                                nn.Dropout(0.3),
                                nn.Linear(128, num_classes)
                            )
                        def forward(self, x):
                            x = self.features(x)
                            x = x.view(x.size(0), -1)
                            return self.classifier(x)
                            
                    model = CustomCNN(num_classes)

                # Load saved state dict weights
                model.load_state_dict(torch.load(PYTORCH_MODEL_PATH, map_location=torch.device('cpu')))
                model.eval()
                self.model = model
                self.backend = "pytorch"
                return
            except Exception as e:
                print(f"Error loading PyTorch model: {e}")

        # 2. Check Keras Model
        if os.path.exists(KERAS_MODEL_PATH):
            try:
                import tensorflow as tf
                self.model = tf.keras.models.load_model(KERAS_MODEL_PATH)
                self.backend = "keras"
                return
            except Exception as e:
                print(f"Error loading Keras model: {e}")

        self.model = None
        self.backend = None

    def is_model_loaded(self):
        """Returns True if a model is loaded and ready for inference."""
        return self.model is not None

    def predict(self, image_input):
        """
        Executes prediction on input image.
        
        Args:
            image_input: File path, BytesIO, PIL Image, or UploadedFile.
            
        Returns:
            dict: Comprehensive prediction output.
        """
        is_valid, validation_res = validate_image(image_input)
        if not is_valid:
            return {"success": False, "error": validation_res}

        if self.model is None:
            return {
                "success": False,
                "error": "Model file not found. Please run model training ('python train.py') first."
            }

        try:
            img_tensor = load_and_preprocess_image(validation_res)

            if self.backend == "pytorch":
                import torch
                import torch.nn.functional as F

                with torch.no_grad():
                    outputs = self.model(img_tensor)
                    probs = F.softmax(outputs, dim=1).squeeze(0).numpy()

            elif self.backend == "keras":
                predictions = self.model.predict(img_tensor, verbose=0)[0]
                probs = predictions
            else:
                return {"success": False, "error": "Unknown model backend."}

            top_index = int(np.argmax(probs))
            confidence_score = float(probs[top_index])
            confidence_pct = round(confidence_score * 100, 2)

            if top_index < len(self.class_names):
                predicted_class = self.class_names[top_index]
            else:
                predicted_class = f"Class_{top_index}"

            disease_info = get_disease_details(predicted_class)

            top_3_indices = np.argsort(probs)[::-1][:3]
            top_3_list = []
            for idx in top_3_indices:
                label = self.class_names[idx] if idx < len(self.class_names) else f"Class_{idx}"
                prob = float(probs[idx]) * 100
                top_3_list.append({
                    "class": label.replace("_", " "),
                    "confidence": round(prob, 2)
                })

            return {
                "success": True,
                "predicted_class": predicted_class.replace("_", " "),
                "raw_class_key": predicted_class,
                "status": disease_info["status"],
                "confidence": confidence_pct,
                "disease_info": disease_info,
                "top_3_predictions": top_3_list,
                "disclaimer": DISCLAIMER_TEXT
            }

        except Exception as e:
            return {"success": False, "error": f"An error occurred during inference: {str(e)}"}

_predictor_instance = None

def get_predictor():
    """Returns singleton instance of LeafDiseasePredictor."""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = LeafDiseasePredictor()
    return _predictor_instance
