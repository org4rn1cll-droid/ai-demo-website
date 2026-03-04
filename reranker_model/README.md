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