import React, { useState, useEffect } from 'react';
import {
  Container,
  Box,
  Typography,
  Paper,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  CircularProgress,
  Grid,
  Card,
  CardContent,
  CardMedia,
  LinearProgress,
  Button,
  Chip,
  Divider,
  Alert,
  Tooltip,
  IconButton,
  useTheme
} from '@mui/material';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';
import BiotechIcon from '@mui/icons-material/Biotech';
import UploadFileIcon from '@mui/icons-material/UploadFile';
import InfoIcon from '@mui/icons-material/Info';
import ScienceIcon from '@mui/icons-material/Science';
import MedicalServicesIcon from '@mui/icons-material/MedicalServices';
import HelpOutlineIcon from '@mui/icons-material/HelpOutline';

const API_URL = 'http://localhost:5000/api';

// Function to get model-specific information
const getModelInfo = (modelId) => {
  const modelInfo = {
    breast_cancer: {
      title: "Breast Cancer Classification",
      description: "This model classifies breast ultrasound images into three categories: benign, malignant, or normal. It was trained on the BUSI dataset using a combination of CNN, ViT, and LSTM architectures.",
      acceptedFormats: "JPEG, PNG",
      icon: <BiotechIcon />,
      color: "#e91e63"
    },
    brain_tumor: {
      title: "Brain Tumor Detection",
      description: "This model detects the presence of tumors in brain MRI images. It was trained on the BraTS 2020 dataset using a hybrid Vision Transformer (ViT) and CNN architecture.",
      acceptedFormats: "JPEG, PNG, TIFF",
      icon: <ScienceIcon />,
      color: "#2196f3"
    },
    bone_fracture: {
      title: "Bone Fracture Detection",
      description: "This model identifies fractures in bone X-ray images. It was trained using a custom CNN with attention mechanisms including Squeeze-and-Excitation (SE) and Convolutional Block Attention Module (CBAM).",
      acceptedFormats: "JPEG, PNG, BMP",
      icon: <MedicalServicesIcon />,
      color: "#ff9800"
    },
    resnet50: {
      title: "ResNet50",
      description: "A general-purpose image classification model that can recognize 1000 different object categories.",
      acceptedFormats: "JPEG, PNG",
      icon: <InfoIcon />,
      color: "#4caf50"
    },
    mobilenet: {
      title: "MobileNet",
      description: "A lightweight image classification model designed for mobile and embedded applications.",
      acceptedFormats: "JPEG, PNG",
      icon: <InfoIcon />,
      color: "#9c27b0"
    },
    efficientnet: {
      title: "EfficientNet",
      description: "An efficient image classification model that balances accuracy and computational efficiency.",
      acceptedFormats: "JPEG, PNG",
      icon: <InfoIcon />,
      color: "#607d8b"
    }
  };
  
  return modelInfo[modelId] || {
    title: "Unknown Model",
    description: "Information not available",
    acceptedFormats: "JPEG, PNG",
    icon: <InfoIcon />,
    color: "#9e9e9e"
  };
};

function App() {
  const theme = useTheme();
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);

  useEffect(() => {
    // Fetch available models
    const fetchModels = async () => {
      try {
        setLoading(true);
        const response = await axios.get(`${API_URL}/models`);
        setModels(response.data);
        
        if (response.data.length > 0) {
          const defaultModel = response.data[0].id;
          setSelectedModel(defaultModel);
          setModelInfo(getModelInfo(defaultModel));
        }
        setError(null);
      } catch (err) {
        setError('Failed to load models. Please ensure the backend server is running.');
        console.error("Error fetching models:", err);
      } finally {
        setLoading(false);
      }
    };

    fetchModels();
  }, []);

  useEffect(() => {
    // Update model info when selected model changes
    if (selectedModel) {
      setModelInfo(getModelInfo(selectedModel));
    }
  }, [selectedModel]);

  const onDrop = (acceptedFiles) => {
    const file = acceptedFiles[0];
    if (file) {
      setImage(file);
      setPreview(URL.createObjectURL(file));
      setPredictions(null);
      setError(null);
    }
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
    },
    multiple: false
  });

  const handleModelChange = (event) => {
    const newModelId = event.target.value;
    setSelectedModel(newModelId);
    setModelInfo(getModelInfo(newModelId));
    setPredictions(null); // Clear previous predictions
  };

  const handlePredict = async () => {
    if (!image || !selectedModel) return;

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('image', image);
    formData.append('model_id', selectedModel);

    try {
      const response = await axios.post(`${API_URL}/predict`, formData);
      setPredictions(response.data.predictions);
    } catch (err) {
      console.error("Error in prediction:", err);
      setError(err.response?.data?.error || 'Failed to get predictions. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const renderPredictionResults = () => {
    if (!predictions) return null;

    // Handle medical model predictions (with class_name)
    if (predictions[0]?.class_name) {
      return (
        <>
          <Typography variant="h6" gutterBottom>Results</Typography>
          {predictions.map((pred, index) => (
            <Card key={index} sx={{ mb: 1, borderLeft: `4px solid ${modelInfo?.color || theme.palette.primary.main}` }}>
              <CardContent>
                <Typography variant="body1" fontWeight="bold">
                  {pred.class_name}
                </Typography>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 1 }}>
                  <Box sx={{ flexGrow: 1 }}>
                    <LinearProgress
                      variant="determinate"
                      value={pred.probability * 100}
                      sx={{ 
                        height: 10, 
                        borderRadius: 5,
                        backgroundColor: theme.palette.grey[300],
                        '& .MuiLinearProgress-bar': {
                          backgroundColor: modelInfo?.color || theme.palette.primary.main
                        }
                      }}
                    />
                  </Box>
                  <Typography variant="body2" fontWeight="bold">
                    {(pred.probability * 100).toFixed(2)}%
                  </Typography>
                </Box>
              </CardContent>
            </Card>
          ))}
        </>
      );
    }

    // Handle general image classification models (with class_id)
    return (
      <>
        <Typography variant="h6" gutterBottom>Top Predictions</Typography>
        {predictions.map((pred, index) => (
          <Card key={index} sx={{ mb: 1, borderLeft: `4px solid ${modelInfo?.color || theme.palette.primary.main}` }}>
            <CardContent>
              <Typography variant="body1" fontWeight="bold">
                Class ID: {pred.class_id}
              </Typography>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 1 }}>
                <Box sx={{ flexGrow: 1 }}>
                  <LinearProgress
                    variant="determinate"
                    value={pred.probability * 100}
                    sx={{ 
                      height: 10, 
                      borderRadius: 5,
                      backgroundColor: theme.palette.grey[300],
                      '& .MuiLinearProgress-bar': {
                        backgroundColor: modelInfo?.color || theme.palette.primary.main
                      }
                    }}
                  />
                </Box>
                <Typography variant="body2" fontWeight="bold">
                  {(pred.probability * 100).toFixed(2)}%
                </Typography>
              </Box>
            </CardContent>
          </Card>
        ))}
      </>
    );
  };

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Box 
        sx={{ 
          textAlign: 'center', 
          mb: 4, 
          p: 3, 
          background: `linear-gradient(135deg, ${theme.palette.primary.main} 0%, ${theme.palette.secondary.main} 100%)`,
          borderRadius: 2,
          color: 'white'
        }}
      >
        <Typography variant="h3" component="h1" gutterBottom>
          Medical Image Analysis
        </Typography>
        <Typography variant="subtitle1">
          Upload an image and select a model to analyze it using advanced deep learning models
        </Typography>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      <Grid container spacing={3}>
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 3, height: '100%' }}>
            <Typography variant="h5" gutterBottom fontWeight="bold" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              {modelInfo?.icon}
              Select Model
            </Typography>
            <Divider sx={{ my: 2 }} />
            
            <FormControl fullWidth sx={{ mb: 3 }}>
              <InputLabel id="model-select-label">Model</InputLabel>
              <Select
                labelId="model-select-label"
                value={selectedModel}
                onChange={handleModelChange}
                label="Model"
                disabled={loading}
              >
                {models.map((model) => (
                  <MenuItem key={model.id} value={model.id}>
                    {model.name}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            {modelInfo && (
              <Box sx={{ mb: 3 }}>
                <Typography variant="h6" sx={{ color: modelInfo.color }}>
                  {modelInfo.title}
                </Typography>
                <Typography variant="body2" sx={{ mt: 1, color: theme.palette.text.secondary }}>
                  {modelInfo.description}
                </Typography>
                <Box sx={{ mt: 2 }}>
                  <Chip 
                    icon={<InfoIcon />} 
                    label={`Formats: ${modelInfo.acceptedFormats}`} 
                    size="small" 
                    variant="outlined"
                    sx={{ borderColor: modelInfo.color, color: modelInfo.color }}
                  />
                </Box>
              </Box>
            )}

            <Typography variant="h5" gutterBottom fontWeight="bold" sx={{ mt: 4, display: 'flex', alignItems: 'center', gap: 1 }}>
              <UploadFileIcon />
              Upload Image
            </Typography>
            <Divider sx={{ my: 2 }} />
            
            <Paper
              {...getRootProps()}
              sx={{
                p: 3,
                textAlign: 'center',
                cursor: 'pointer',
                bgcolor: isDragActive ? 'action.hover' : 'background.paper',
                border: '2px dashed',
                borderColor: isDragActive ? modelInfo?.color || 'primary.main' : 'grey.300',
                borderRadius: 2,
                transition: 'all 0.3s ease',
                '&:hover': {
                  borderColor: modelInfo?.color || 'primary.main',
                  boxShadow: '0 4px 8px rgba(0,0,0,0.1)'
                }
              }}
            >
              <input {...getInputProps()} />
              {!preview && (
                <Box sx={{ p: 2 }}>
                  <Typography variant="subtitle1" fontWeight="medium" sx={{ mb: 1 }}>
                    {isDragActive
                      ? "Drop the image here"
                      : "Drag and drop an image here, or click to select"}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Supported formats: JPG, PNG, JPEG, BMP, TIF
                  </Typography>
                </Box>
              )}
            </Paper>

            <Button
              onClick={handlePredict}
              disabled={!image || loading}
              fullWidth
              size="large"
              variant="contained"
              sx={{ 
                mt: 3, 
                py: 1.5,
                bgcolor: modelInfo?.color || 'primary.main',
                '&:hover': {
                  bgcolor: modelInfo?.color ? `${modelInfo.color}dd` : 'primary.dark',
                }
              }}
              startIcon={loading ? <CircularProgress size={20} color="inherit" /> : modelInfo?.icon}
            >
              {loading ? "Processing..." : "Analyze Image"}
            </Button>
          </Paper>
        </Grid>
        
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 3, height: '100%' }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
              <Typography variant="h5" fontWeight="bold">
                {preview ? "Image Preview" : "No Image Selected"}
              </Typography>
              {modelInfo && (
                <Tooltip title={`Using ${modelInfo.title}`}>
                  <Chip 
                    label={modelInfo.title} 
                    size="small" 
                    color="primary"
                    icon={modelInfo.icon}
                    sx={{ 
                      bgcolor: modelInfo.color,
                      '& .MuiChip-label': { fontWeight: 500 }
                    }}
                  />
                </Tooltip>
              )}
            </Box>
            <Divider sx={{ mb: 3 }} />

            {preview ? (
              <Box sx={{ position: 'relative' }}>
                <CardMedia
                  component="img"
                  image={preview}
                  alt="Uploaded Image Preview"
                  sx={{ 
                    maxHeight: 400, 
                    objectFit: 'contain',
                    borderRadius: 1,
                    boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
                    mb: 3
                  }}
                />
                
                {loading && (
                  <Box sx={{ 
                    position: 'absolute', 
                    top: 0, 
                    left: 0, 
                    right: 0, 
                    bottom: 0, 
                    display: 'flex', 
                    alignItems: 'center', 
                    justifyContent: 'center',
                    backgroundColor: 'rgba(255,255,255,0.7)'
                  }}>
                    <CircularProgress size={60} sx={{ color: modelInfo?.color || 'primary.main' }} />
                  </Box>
                )}
              </Box>
            ) : (
              <Box 
                sx={{ 
                  height: 300, 
                  display: 'flex', 
                  alignItems: 'center', 
                  justifyContent: 'center',
                  border: '1px dashed',
                  borderColor: 'grey.300',
                  borderRadius: 2,
                  p: 2,
                  mb: 3
                }}
              >
                <Box sx={{ textAlign: 'center' }}>
                  <UploadFileIcon sx={{ fontSize: 80, color: 'grey.400', mb: 2 }} />
                  <Typography color="text.secondary">
                    Please upload an image to see preview and analysis results
                  </Typography>
                </Box>
              </Box>
            )}

            {renderPredictionResults()}
          </Paper>
        </Grid>
      </Grid>

      <Paper sx={{ mt: 4, p: 3, bgcolor: theme.palette.grey[50] }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <HelpOutlineIcon color="primary" />
          <Typography variant="h6">How It Works</Typography>
        </Box>
        <Divider sx={{ my: 2 }} />
        <Grid container spacing={2}>
          <Grid item xs={12} md={4}>
            <Box>
              <Typography variant="subtitle1" fontWeight="bold">1. Select a Model</Typography>
              <Typography variant="body2" color="text.secondary">
                Choose from specialized medical models for breast cancer, brain tumors, and bone fractures, or general image classification models.
              </Typography>
            </Box>
          </Grid>
          <Grid item xs={12} md={4}>
            <Box>
              <Typography variant="subtitle1" fontWeight="bold">2. Upload an Image</Typography>
              <Typography variant="body2" color="text.secondary">
                Upload a relevant medical image in supported formats (JPG, PNG, JPEG, etc.) to be analyzed by the selected model.
              </Typography>
            </Box>
          </Grid>
          <Grid item xs={12} md={4}>
            <Box>
              <Typography variant="subtitle1" fontWeight="bold">3. Get Results</Typography>
              <Typography variant="body2" color="text.secondary">
                Our AI models will analyze the image and provide predictions with confidence scores to assist in diagnosis.
              </Typography>
            </Box>
          </Grid>
        </Grid>
      </Paper>
    </Container>
  );
}

export default App; 