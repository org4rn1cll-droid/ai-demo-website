# bayesian_core.py

import pandas as pd
import numpy as np
import json
from collections import defaultdict
from math import log

EPS = 1e-9


class BayesianCore:

    def __init__(self, symptoms_csv, priors_json):

        self.df = pd.read_csv(symptoms_csv)
        self.base_priors = json.load(open(priors_json))

        self.disease_symptom_probs = defaultdict(dict)
        self.all_symptoms = set()

        for _, row in self.df.iterrows():
            d = row["disease_id"]
            s = row["symptom_id"]
            p = float(row["probability"])
            self.disease_symptom_probs[d][s] = max(min(p, 1 - EPS), EPS)
            self.all_symptoms.add(s)

        self.diseases = list(self.disease_symptom_probs.keys())
        self.priors = self._build_priors()

    def _build_priors(self):

        priors = {}

        for key, value in self.base_priors.items():
            disease_id = key.split(",")[0]
            priors[disease_id] = max(float(value), EPS)

        total = sum(priors.values())

        for d in priors:
            priors[d] /= total

        return priors

    def score(self, positive_symptoms, negative_symptoms=None):

        if negative_symptoms is None:
            negative_symptoms = []

        log_scores = {}

        for disease in self.diseases:

            prior = self.priors.get(disease, EPS)
            log_p = log(prior)

            symptom_probs = self.disease_symptom_probs[disease]

            # Positive evidence
            for s in positive_symptoms:
                p = symptom_probs.get(s, EPS)
                log_p += log(p)

            # Explicit negative evidence only
            for s in negative_symptoms:
                p = symptom_probs.get(s, EPS)
                log_p += log(1 - p)

            log_scores[disease] = log_p

        # Stable normalization
        max_log = max(log_scores.values())
        exp_scores = {d: np.exp(v - max_log) for d, v in log_scores.items()}
        total = sum(exp_scores.values())

        return {d: float(v / total) for d, v in exp_scores.items()}

    def top_k(self, positive_symptoms, negative_symptoms=None, k=20):
        scores = self.score(positive_symptoms, negative_symptoms)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:k]