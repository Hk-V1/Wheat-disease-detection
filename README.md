# Wheat Disease Recognition & Treatment Advisor

AI-powered web application for detecting wheat leaf diseases and providing treatment recommendations using deep learning.

## Overview

This application enables farmers and agricultural professionals to identify wheat diseases from leaf images. The system analyzes uploaded photos and provides disease classification along with evidence-based treatment recommendations.

## Features

- Detection of 6 disease categories: Healthy, Wheat Rust, Leaf Blight, Powdery Mildew, Septoria Leaf Spot, and Fusarium Head Blight
- Image upload interface with preprocessing pipeline
- Confidence score visualization for model predictions
- Expert-guided treatment recommendations based on disease type
- Analysis history tracking
- Mobile-responsive design for field deployment

## Requirements

- Python 3.13 or higher
- Streamlit 1.39.0+
- Pillow 10.4.0+
- NumPy 1.26.0+
- PyTorch 2.5.0+ or TensorFlow 2.18.0+ (for model integration)

```

## Model Integration

The application includes placeholder functions for model integration. Replace the mock prediction logic with your trained model.

### PyTorch Integration

```python
# In load_model() function
import torch
model = torch.load('wheat_disease_model.pth', map_location=torch.device('cpu'))
model.eval()
return model
```

### TensorFlow Integration

```python
# In load_model() function
import tensorflow as tf
model = tf.keras.models.load_model('wheat_disease_model.h5')
return model
```

## Disease Coverage and Treatment

| Disease | Primary Treatment |
|---------|-------------------|
| Healthy | Continued monitoring and preventive practices |
| Wheat Rust | Triazole or strobilurin fungicides with crop rotation |
| Leaf Blight | Mancozeb 75 WP at 2.5 g/L, reduced overhead irrigation |
| Powdery Mildew | Sulfur dust or systemic fungicides (Hexaconazole) |
| Septoria Leaf Spot | Propiconazole spray with improved field drainage |
| Fusarium Head Blight | Tebuconazole application, seed quality control |

## Technical Architecture

**Frontend:** Streamlit web framework  
**Backend:** Python 3.13  
**ML Framework:** PyTorch or TensorFlow  
**Image Processing:** Pillow, NumPy  
**Model Format:** Standard checkpoint or TensorFlow Lite (mobile deployment)  
**Annotation Tool:** LabelImg (training data preparation)

## Future Development

- Real-time camera capture integration
- Batch processing for multiple images
- PDF report generation with recommendations
- Multi-language support for international deployment
- Mobile application using TensorFlow Lite
- RESTful API using FastAPI for production deployment
- Integration with agricultural databases

## Limitations

- Current version uses simulated predictions for demonstration
- Requires trained model for production use
- Treatment recommendations are general guidelines
- Not a substitute for professional agricultural consultation

## Deployment Considerations

For production deployment:
- Train model on validated dataset with minimum 10,000 labeled images
- Implement model versioning and A/B testing
- Add authentication for commercial use
- Set up logging and monitoring
- Consider edge deployment for offline field use
- Optimize model with TensorFlow Lite or ONNX for mobile devices

