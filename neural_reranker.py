# neural_reranker.py

import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

class NeuralReranker:

    def __init__(self, model_path=None, device="cuda"):
        self.device = device
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

        if model_path:
            self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
        else:
            self.model = DistilBertForSequenceClassification.from_pretrained(
                "distilbert-base-uncased",
                num_labels=1
            )

        self.model.to(device) # type: ignore
        self.model.eval()

    def score(self, symptom_text, disease_name):
        text = f"[SYMPTOMS] {symptom_text} [DISEASE] {disease_name}"

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            score = torch.sigmoid(outputs.logits).item()

        return score