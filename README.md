# Medical Image Analysis Application

A full-stack application for medical image analysis using advanced deep learning models. Built with Flask backend and React frontend.

## Features

- Multiple specialized medical models:
  - Breast Cancer Classification (CNN + ViT + LSTM)
  - Brain Tumor Detection (ViT + CNN hybrid)
  - Bone Fracture Detection (CNN with SE and CBAM blocks)
- General image classification models:
  - ResNet50
  - MobileNet
  - EfficientNet
- Beautiful and responsive UI with Material-UI
- Real-time image preview
- Drag and drop image upload
- Model-specific information and descriptions
- Detailed prediction results with confidence scores

## Prerequisites

- Python 3.7+
- Node.js 14+
- npm or yarn

## Backend Setup

1. Navigate to the Backend directory:

```bash
cd Backend
```

2. Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. About the models:

   - The application will work with or without the actual trained models
   - If the model files are not found, the application will use dummy models that return random predictions
   - This is useful for demonstration and testing the UI

5. If you want to use your own trained models, place them in the models directory:

```
Backend/models/
  ├── advanced_breast_cancer_classification_model.h5
  ├── brats_tumor_classification_model.h5
  └── custom_cnn_bone_fracture_classifier_v2.keras
```

6. Run the Flask server:

```bash
python app.py
```

The backend server will start at `http://localhost:5000`

### Frontend Setup

1. Navigate to the Frontend directory:

```bash
cd Frontend
```

2. Install dependencies:

```bash
npm install
```

3. Start the development server:

```bash
npm run dev
```

The frontend application will start at `http://localhost:3000`

## Usage

1. Open your browser and navigate to `http://localhost:3000`
2. Select a model from the dropdown menu (specialized medical models or general classification models)
3. Drag and drop an image or click to select one
4. Click the "Analyze Image" button to get predictions
5. View the results with confidence scores

> **Note:** If you're using the dummy models, the predictions will be randomly generated. For actual medical analysis, you should train and use real models with proper medical datasets.

## Medical Model Information

### Breast Cancer Classification Model

- **Description**: This model classifies breast ultrasound images into three categories: benign, malignant, or normal.
- **Architecture**: Combined CNN, Vision Transformer (ViT), and LSTM
- **Training Dataset**: BUSI dataset with benign, malignant, and normal images
- **Model File**: advanced_breast_cancer_classification_model.h5

### Brain Tumor Detection Model

- **Description**: This model detects the presence of tumors in brain MRI images.
- **Architecture**: Hybrid CNN and Vision Transformer (ViT)
- **Training Dataset**: BraTS 2020 dataset
- **Model File**: brats_tumor_classification_model.h5

### Bone Fracture Detection Model

- **Description**: This model identifies fractures in bone X-ray images.
- **Architecture**: Custom CNN with attention mechanisms (SE and CBAM blocks)
- **Training Dataset**: Bone fracture X-ray images
- **Model File**: custom_cnn_bone_fracture_classifier_v2.keras

## Training Your Own Models

If you want to use real predictions instead of the dummy models, you'll need to:

1. Train your own models using the provided code in the training directory
2. Save the models with the expected filenames in the Backend/models directory
3. The application will automatically use your trained models instead of the dummy ones

## Supported Image Formats

- PNG
- JPG/JPEG
- BMP
- TIF/TIFF

## Technologies Used

### Backend

- Flask
- TensorFlow
- PyTorch
- OpenCV
- NumPy

### Frontend

- React
- Vite
- Material-UI
- React Dropzone
- Axios
# Divakar-cap-project
