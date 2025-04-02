import os
import torch
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import torchvision.transforms as transforms
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image as keras_image

# Path to model files
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
BREAST_CANCER_MODEL = os.path.join(
    MODEL_DIR, 'Breast_Cancer.h5')
BRAIN_TUMOR_MODEL = os.path.join(
    MODEL_DIR, 'Brain_Tumor.h5')
BONE_FRACTURE_MODEL = os.path.join(
    MODEL_DIR, 'Bone_Fracture.keras')

# Dictionary to store loaded models
loaded_models = {}


def create_dummy_model(model_type):
    """Create a simple dummy model when the actual model file is not found."""
    print(
        f"Creating a dummy model for {model_type} since the actual model file is not found.")

    if model_type == "breast_cancer":
        # Create a simple model that outputs 3 classes
        base_model = MobileNetV2(
            weights=None, include_top=False, input_shape=(224, 224, 3))
        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            # 3 classes: benign, malignant, normal
            Dense(3, activation='softmax')
        ])
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy', metrics=['accuracy'])

    elif model_type == "brain_tumor":
        # Create a simple model that outputs binary classification
        base_model = MobileNetV2(
            weights=None, include_top=False, input_shape=(224, 224, 1))
        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(1, activation='sigmoid')  # Binary: tumor or no tumor
        ])
        model.compile(optimizer='adam',
                      loss='binary_crossentropy', metrics=['accuracy'])

    elif model_type == "bone_fracture":
        # Create a simple model that outputs 2 classes
        base_model = MobileNetV2(
            weights=None, include_top=False, input_shape=(224, 224, 3))
        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            # 2 classes: fracture or no fracture
            Dense(2, activation='softmax')
        ])
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def load_specific_model(model_id):
    """Load a specific model if it's not already loaded."""
    if model_id not in loaded_models:
        try:
            if model_id == "breast_cancer":
                if os.path.exists(BREAST_CANCER_MODEL):
                    model = load_model(BREAST_CANCER_MODEL)
                else:
                    print(
                        f"Warning: Breast cancer model file not found at {BREAST_CANCER_MODEL}")
                    print("Using a dummy model instead. Results will be random.")
                    model = create_dummy_model("breast_cancer")

            elif model_id == "brain_tumor":
                if os.path.exists(BRAIN_TUMOR_MODEL):
                    model = load_model(BRAIN_TUMOR_MODEL)
                else:
                    print(
                        f"Warning: Brain tumor model file not found at {BRAIN_TUMOR_MODEL}")
                    print("Using a dummy model instead. Results will be random.")
                    model = create_dummy_model("brain_tumor")

            elif model_id == "bone_fracture":
                if os.path.exists(BONE_FRACTURE_MODEL):
                    model = load_model(BONE_FRACTURE_MODEL)
                else:
                    print(
                        f"Warning: Bone fracture model file not found at {BONE_FRACTURE_MODEL}")
                    print("Using a dummy model instead. Results will be random.")
                    model = create_dummy_model("bone_fracture")

            elif model_id == "resnet50":
                model = torch.hub.load(
                    'pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
                model.eval()

            elif model_id == "mobilenet":
                model = torch.hub.load(
                    'pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
                model.eval()

            elif model_id == "efficientnet":
                model = torch.hub.load(
                    'pytorch/vision:v0.10.0', 'efficientnet_b0', pretrained=True)
                model.eval()

            else:
                raise ValueError(f"Unknown model: {model_id}")

            loaded_models[model_id] = model
        except Exception as e:
            print(f"Error loading model {model_id}: {str(e)}")
            raise

    return loaded_models[model_id]


def preprocess_image(image_tensor, model_id):
    """Preprocess image for a specific model."""
    if model_id in ["breast_cancer", "brain_tumor", "bone_fracture"]:
        # Convert PyTorch tensor to numpy array for TensorFlow models
        img_array = image_tensor.numpy().squeeze(0).transpose(1, 2, 0)
        img_array = (img_array * 255).astype(np.uint8)

        if model_id == "breast_cancer":
            # Breast cancer model preprocessing
            img_array = cv2.resize(img_array, (224, 224))
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)

        elif model_id == "brain_tumor":
            # Brain tumor model preprocessing
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            img_array = cv2.resize(img_array, (224, 224))
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            img_array = np.expand_dims(
                img_array, axis=-1)  # Add channel dimension

        elif model_id == "bone_fracture":
            # Bone fracture model preprocessing
            img_array = cv2.resize(img_array, (224, 224))
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)

        return img_array

    # Return original tensor for PyTorch models
    return image_tensor


def get_model_predictions(model_id, image_tensor):
    """Get predictions from the specified model."""
    try:
        model = load_specific_model(model_id)

        # Process the image based on model type
        if model_id in ["breast_cancer", "brain_tumor", "bone_fracture"]:
            # Preprocess for TensorFlow models
            processed_image = preprocess_image(image_tensor, model_id)
            predictions = model.predict(processed_image)

            if model_id == "breast_cancer":
                class_names = ["benign", "malignant", "normal"]
                results = []
                for i, prob in enumerate(predictions[0]):
                    results.append({
                        "class_name": class_names[i],
                        "probability": float(prob)
                    })
                return results

            elif model_id == "brain_tumor":
                return [{
                    "class_name": "No Tumor" if predictions[0][0] < 0.5 else "Tumor",
                    "probability": float(1 - predictions[0][0]) if predictions[0][0] < 0.5 else float(predictions[0][0])
                }]

            elif model_id == "bone_fracture":
                class_names = ["No Fracture", "Fracture"]
                results = []
                for i, prob in enumerate(predictions[0]):
                    results.append({
                        "class_name": class_names[i],
                        "probability": float(prob)
                    })
                return results

        else:
            # Processing for PyTorch models
            with torch.no_grad():
                outputs = model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)

                # Get top 5 predictions
                top5_prob, top5_catid = torch.topk(probabilities, 5)

                # Convert to list of dictionaries
                predictions = []
                for i in range(5):
                    predictions.append({
                        "class_id": int(top5_catid[0][i]),
                        "probability": float(top5_prob[0][i])
                    })
            return predictions

    except Exception as e:
        print(f"Error during prediction with model {model_id}: {str(e)}")
        raise


def get_available_models():
    """Return a list of available models for the frontend."""
    models = [
        {
            "id": "breast_cancer",
            "name": "Breast Cancer Classification",
            "description": "Classifies breast ultrasound images as benign, malignant, or normal"
        },
        {
            "id": "brain_tumor",
            "name": "Brain Tumor Detection",
            "description": "Detects the presence of tumors in brain MRI images"
        },
        {
            "id": "bone_fracture",
            "name": "Bone Fracture Detection",
            "description": "Detects fractures in bone X-ray images"
        },
        {
            "id": "resnet50",
            "name": "ResNet50",
            "description": "General image classification model (1000 classes)"
        },
        {
            "id": "mobilenet",
            "name": "MobileNet",
            "description": "Lightweight image classification model (1000 classes)"
        },
        {
            "id": "efficientnet",
            "name": "EfficientNet",
            "description": "Efficient image classification model (1000 classes)"
        }
    ]
    return models
