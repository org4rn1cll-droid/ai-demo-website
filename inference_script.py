# inference_script.py

import json
import numpy as np
import pandas as pd
import torch
import re
from math import log
from sentence_transformers import SentenceTransformer # type: ignore
import ollama # type: ignore

from bayesian_core import BayesianCore
from neural_reranker import NeuralReranker
from hybrid_model import HybridDX

EPS = 1e-9


# ==========================================================
# HYBRID EXTRACTOR (Segment-Based + Strict LLM)
# ==========================================================

class HybridCanonicalExtractor:

    def __init__(self, canonical_csv, top_k=30, model_name="llama3"):

        self.top_k = top_k
        self.llm_model = model_name

        df = pd.read_csv(canonical_csv)

        self.surface_forms = df["surface_form"].astype(str).tolist()
        self.surface_to_canonical = dict(zip(df["surface_form"], df["canonical"]))
        self.canonical_to_id = dict(zip(df["canonical"], df["symptom_id"]))

        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        self.surface_embeddings = self.embedder.encode(
            self.surface_forms,
            convert_to_tensor=True,
            normalize_embeddings=True
        )

    # Segment-based prefilter (major recall improvement)
    def _prefilter(self, text):

        segments = re.split(r"[,.]", text.lower())

        candidate_canonicals = set()

        for segment in segments:

            segment = segment.strip()
            if not segment:
                continue

            segment_embedding = self.embedder.encode(
                segment,
                convert_to_tensor=True,
                normalize_embeddings=True
            )

            similarities = torch.matmul(self.surface_embeddings, segment_embedding)

            top_indices = torch.topk(
                similarities,
                k=min(self.top_k, len(self.surface_forms))
            ).indices.tolist()

            for idx in top_indices:
                surface = self.surface_forms[idx]
                canonical = self.surface_to_canonical.get(surface)
                if canonical:
                    candidate_canonicals.add(canonical)

        return list(candidate_canonicals)

    def _llm_select(self, text, candidates):

        if not candidates:
            return []

        candidate_block = "\n".join([f"- {c}" for c in candidates])

        prompt = f"""
You are extracting symptoms explicitly stated by the patient.

STRICT RULES:
- Only select symptoms directly mentioned.
- Do NOT infer new symptoms.
- Do NOT assume disease patterns.
- If unsure, do NOT select it.

Return ONLY a raw JSON array.
No explanations.
No markdown.

Patient input:
"{text}"

Symptom list:
{candidate_block}
"""

        response = ollama.chat(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0}
        )

        content = response["message"]["content"]

        match = re.search(r"\[.*?\]", content, re.DOTALL)

        if not match:
            return []

        try:
            parsed = json.loads(match.group())
            return parsed if isinstance(parsed, list) else []
        except:
            return []

    def extract(self, text):

        candidates = self._prefilter(text)
        selected = self._llm_select(text, candidates)

        positive_ids = []

        for canonical in selected:
            if canonical in self.canonical_to_id:
                positive_ids.append(self.canonical_to_id[canonical])

        return list(set(positive_ids)), []


# ==========================================================
# ADDITIVE SPECIFICITY BAYESIAN WRAPPER
# ==========================================================

class BayesianWithSpecificity:

    def __init__(self, bayes_core, combined_csv):

        self.core = bayes_core

        df = pd.read_csv(combined_csv)
        # Simple fallback - if specificity_tier column doesn't exist, use 1 for all symptoms
        if "specificity_tier" in df.columns:
            tier_column = df["specificity_tier"]
        else:
            tier_column = [1] * len(df)  # Simple list of 1s
        self.tiers = dict(zip(df["symptom_id"], tier_column))

    def score(self, positive, negative=None):

        if negative is None:
            negative = []

        base_probs = self.core.score(positive, negative)

        # Add small additive specificity bonus
        boosted = {}

        for disease, prob in base_probs.items():

            log_p = log(prob + EPS)

            for s in positive:
                tier = self.tiers.get(s, 1)
                log_p += 0.12 * tier   # gentle boost

            boosted[disease] = log_p

        max_log = max(boosted.values())
        exp_scores = {d: np.exp(v - max_log) for d, v in boosted.items()}
        total = sum(exp_scores.values())

        return {d: float(v / total) for d, v in exp_scores.items()}


# ==========================================================
# INFERENCE ENGINE
# ==========================================================

class InferenceEngine:

    def __init__(self):

        disease_df = pd.read_csv("disease_ids.csv")
        self.disease_id_to_name = dict(zip(disease_df["disease_id"], disease_df["disease"]))

        self.base_bayes = BayesianCore("combined_symptoms.csv", "base_priors.json")

        self.extractor = HybridCanonicalExtractor(
            canonical_csv="surface_to_canonical.csv",
            top_k=30
        )

        # Check if combined_symptoms.csv has specificity_tier column, if not, create a fallback
        df_check = pd.read_csv("combined_symptoms.csv")
        if "specificity_tier" in df_check.columns:
            self.bayes = BayesianWithSpecificity(self.base_bayes, "combined_symptoms.csv")
        else:
            self.bayes = BayesianWithSpecificity(self.base_bayes, "combined_symptoms.csv")

        self.reranker = NeuralReranker(model_path="reranker_model")
        self.hybrid = HybridDX(self.base_bayes, self.reranker)

        canonical_df = pd.read_csv("surface_to_canonical.csv")
        self.symptom_id_to_name = dict(zip(canonical_df["symptom_id"], canonical_df["canonical"]))

    def diagnose(self, text):
        """
        Run diagnosis on the given symptom text.

        Returns a dict with:
          - "results": list of up to 3 diagnoses, each containing:
              - "disease":          str   disease name
              - "score":            float confidence (0–1)
              - "matched_symptoms": list  canonical symptom names that matched
              - "total_symptoms":   int   total symptoms extracted from input
          - "error": str (only present if something went wrong)
        """

        positive, negative = self.extractor.extract(text)

        if not positive:
            return {"error": "No recognizable symptoms found. Please describe your symptoms in more detail."}

        hybrid_ranked = self.hybrid.diagnose(
            input_symptoms=positive,
            symptom_text=text,
            top_k=3
        )

        results = []

        for disease_id, score in hybrid_ranked:

            disease_name = self.disease_id_to_name.get(disease_id, disease_id)
            disease_symptoms = set(self.base_bayes.disease_symptom_probs[disease_id].keys())

            matched = [s for s in positive if s in disease_symptoms]
            matched_names = [self.symptom_id_to_name.get(s, s) for s in matched]

            results.append({
                "disease": disease_name,
                "score": score,
                "matched_symptoms": matched_names,
                "total_symptoms": len(positive)
            })

        return {"results": results}


# ==========================================================
# CLI
# ==========================================================

if __name__ == "__main__":

    engine = InferenceEngine()

    while True:
        text = input("\nEnter symptoms: ")
        output = engine.diagnose(text)

        if "error" in output:
            print(output["error"])
            continue

        print("\nTop Diagnoses:\n")

        for i, item in enumerate(output["results"], 1):
            result = item  # type: ignore
            print(f"{i}. {result['disease']}") # type: ignore
            print(f"Confidence: {result['score'] * 100:.2f}%") # type: ignore
            print(f"Matched {len(result['matched_symptoms'])}/{result['total_symptoms']} symptoms: " + # type: ignore
                ", ".join(result["matched_symptoms"])) # type: ignore
            print("-" * 60)
