# Medical AI Diagnosis Demo

A local medical AI diagnosis web application that demonstrates disease prediction based on patient symptoms using a hybrid retrieval-augmented diagnosis system.

## Project Overview

This is a local web demo that allows users to input symptoms and receive potential disease diagnoses. The application uses a sophisticated AI model that combines symptom extraction with disease ranking to provide accurate medical predictions.
demo:
https://superficial-purposely-cherrie.ngrok-free.dev/

## Features

- **Local Execution**: Runs entirely on a gpu server with no internet required
- **Simple Interface**: Clean, user-friendly web interface for symptom input
- **Real-time Diagnosis**: Instant results with confidence scores
- **Top 3 Results**: Shows the most likely diagnoses with percentages
- **CORS-Free**: Backend and frontend run from the same origin

## Technical Architecture

### Backend
- **Framework**: Python Flask
- **Model**: InferenceEngine loaded once at startup
- **Endpoint**: `/diagnose` (POST)
- **Input**: JSON with "text" field containing patient symptoms
- **Output**: JSON with "results" array containing disease names and scores

### Frontend
- **HTML/CSS/JavaScript**: Pure client-side implementation
- **Responsive Design**: Works on desktop and mobile devices
- **Loading States**: Shows loading indicator during processing
- **Error Handling**: Graceful handling of empty inputs

## Requirements

- Python 3.6+
- Flask
- Flask-CORS
- PyTorch (with CUDA support if available)
- All model files in the `medical_ai` directory

## Setup Instructions

1. **Install Dependencies**:
   ```bash
   pip install flask flask-cors torch
   ```

2. **Run the Application**:
   ```bash
   python app.py
   ```

3. **Access the Demo**:
   Open your browser and go to `http://localhost:5000`

## Model Information

This demo uses a hybrid retrieval-augmented diagnosis model that:
1. Extracts positive and negative symptoms from patient input
2. Ranks diseases based on symptom similarity
3. Returns top 3 most likely diagnoses with confidence scores

The model files are located in the `reranker_model` directory and called by:
- `inference_script.py` - Main inference engine
- `hybrid_model.py` - Hybrid diagnosis model
- `bayesian_core.py` - Bayesian reasoning core
- Various data files for disease mapping and symptom analysis

## Usage

1. Enter symptoms in the text area (e.g., "headache, fever, nausea")
2. Click "Run Diagnosis"
3. View the top 3 potential diseases with confidence percentages

## Security Notes

- This is a development-only server - not suitable for production use
- All processing happens locally on your machine
- No data is sent to external servers
- CORS is enabled for local development only

## File Structure

```
.
├── app.py                 # Flask backend
├── index.html             # Frontend HTML
├── style.css              # Frontend styling
├── script.js              # Frontend JavaScript
├── inference_script.py    # Main inference engine
├── hybrid_model.py        # Hybrid diagnosis model
├── bayesian_core.py       # Bayesian reasoning core
├── # Model data related files and directory
│   base_priors.json
│   combined_symptoms.csv
│   surface_to_canonical.csv
│   ── reranker_model
└── README.md              # This file
```

## License

This project is for demonstration purposes only. The medical AI model and its predictions should not be used for actual medical diagnosis.

## Contributing

This is a demo project. For educational purposes only.


# Reranker Model

This directory contains the reranker model used in the medical symptom diagnosis system. The reranker is responsible for re-ranking disease candidates based on symptom relevance and context.

## Model Overview

The reranker model is a transformer-based model fine-tuned for medical symptom diagnosis ranking. It takes symptom-text pairs and ranks diseases based on how well they match the given symptoms.

## Model Files

- `config.json` - Model configuration and hyperparameters
- `model.safetensors` - The actual model weights

## How the Model was Generated

The reranker model was trained using the following process:

1. **Data Preparation**: 
   - Collected medical symptom-disease pairs from various sources
   - Processed and cleaned the data for training
   - Created training samples with symptom-text and disease pairs

2. **Training Process**:
   - Fine-tuned a pre-trained transformer model on medical text data
   - Optimized for ranking accuracy in medical diagnosis contexts
   - Used specialized medical datasets for better domain understanding

3. **Evaluation**:
   - Tested on held-out medical datasets
   - Evaluated ranking performance metrics
   - Fine-tuned hyperparameters for optimal results

## Usage in the System

The reranker is used in the `HybridDX` class through the `NeuralReranker` component:

```python
from neural_reranker import NeuralReranker

# Initialize the reranker
reranker = NeuralReranker(model_path="reranker_model")

# Use in diagnosis
hybrid_ranked = self.hybrid.diagnose(
    input_symptoms=positive,
    symptom_text=text,
    top_k=3
)
```

## Model Architecture

The model architecture is designed for:
- Fast inference times
- Good performance on medical text
- Ability to handle various symptom combinations
- Integration with the Bayesian diagnostic framework

## Dependencies

- transformers (Hugging Face)
- torch
- numpy
