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
- **Error Handling**: Graceful handling of empty inputs

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
- All processing happens locally on the gpu server
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

This project is for demonstration purposes only. The medical AI model and its predictions should not be used for actual medical diagnosis. Consult a doctor if actual issues arise, not this diagnostic tool.

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

## Pipeline Overview
The system is designed as a multi-stage AI pipeline, each stage addressing a core challenge of medical knowledge extraction and inference.
1. Disease List and Web Sources
I began by curating a list of diseases and associated high-quality medical web pages describing symptoms, risk factors, and clinical context. This manual curation ensured the foundation of the system was grounded in reliable sources.
2. Knowledge Corpus Collection
I collected the full text from each source, producing a raw knowledge corpus of approximately 4.5 GB. This large-scale corpus represents a diverse range of diseases and clinical presentations.
3. AI-Assisted Extraction and Filtering
To process this large corpus efficiently, I used a large language model (LLM) to read and extract structured information. The LLM identified relevant features, such as symptoms, severity, risk factors, and comorbidities, while filtering out irrelevant content. Additionally, the LLM standardized terminology to produce a consistent, high-quality dataset suitable for training AI models.
4. Structured Dataset Preparation
The filtered data was stored in a structured format, creating a comprehensive dataset ready for machine learning. This step involved balancing disease classes, removing duplicates, and ensuring the data captured realistic symptom combinations.
5. Reranking Model: Design, Procedure, and Importance
A key innovation of this system is the reranking model, which improves prediction accuracy beyond simple retrieval methods.
Purpose:
When multiple candidate diseases match a patient’s symptoms, the reranking model evaluates them more deeply and ranks them by probability. This ensures the system presents the most likely diagnoses first, making predictions both accurate and clinically meaningful.
Procedure:
Candidate Retrieval: The system first retrieves a list of potential diseases based on symptom matching.
Feature Extraction: For each candidate, features are derived from the structured dataset, such as symptom overlap, disease severity, and prior probabilities.
Model Training: The reranking model is trained using supervised learning. The input is symptom-feature vectors, and the output is the correct ordering of disease candidates. Training includes:
Assigning higher weights to diseases with more symptom overlap
Incorporating prior knowledge (Bayesian priors) to reflect prevalence
Optimizing ranking loss to improve ordering accuracy
Evaluation: Model performance is assessed using ranking metrics such as Precision@1, Mean Reciprocal Rank (MRR), and Normalized Discounted Cumulative Gain (NDCG).
Why It’s Important:
Without reranking, the system could return multiple plausible diseases in random or suboptimal order, making it difficult for doctors or users to interpret. The reranker ensures the most relevant, evidence-supported diagnoses appear first, significantly improving the system’s usability and reliability. This mirrors real clinical reasoning, where physicians prioritize differential diagnoses based on likelihood and severity.
6. Bayesian Reasoning Layer
A Bayesian reasoning layer incorporates prior knowledge about disease prevalence and symptom likelihoods. By combining probabilistic reasoning with the reranking model, the system outputs confidence-based predictions, resembling the diagnostic process used by clinicians.
7. GPU-Based Inference
Finally, the trained model is deployed with GPU acceleration to provide real-time predictions. This ensures the system can handle multiple queries quickly, delivering instantaneous probabilistic outputs.
Pipeline Diagram
+---------------------------+
| Disease List + Web Links  |
+------------+--------------+
             |
             v
+---------------------------+
| Raw Knowledge Corpus      |
| (~4.5 GB)                 |
+------------+--------------+
             |
             v
+---------------------------+
| AI Extraction + Filtering |
| (LLM-assisted)            |
+------------+--------------+
             |
             v
+---------------------------+
| Structured Dataset        |
+------------+--------------+
             |
             v
+---------------------------+
| Reranking Model Training  |
| - Candidate retrieval     |
| - Feature extraction      |
| - Supervised training     |
| - Ranking evaluation      |
+------------+--------------+
             |
             v
+---------------------------+
| Bayesian Reasoning Layer  |
+------------+--------------+
             |
             v
+---------------------------+
| GPU-Based Inference       |
| → Probabilistic Diagnosis |
+---------------------------+


