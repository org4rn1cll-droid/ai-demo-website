# hybrid_model.py

class HybridDX:

    def __init__(self, bayesian_core, reranker, alpha=0.7, beta=0.3):
        self.bayes = bayesian_core
        self.reranker = reranker
        self.alpha = alpha
        self.beta = beta

    def diagnose(self, input_symptoms, symptom_text, top_k=10):

        top_candidates = self.bayes.top_k(input_symptoms, k=20)

        final_scores = []

        for disease, bayes_score in top_candidates:

            neural_score = self.reranker.score(symptom_text, disease)

            combined = (
                self.alpha * bayes_score +
                self.beta * neural_score
            )

            final_scores.append((disease, combined))

        ranked = sorted(final_scores, key=lambda x: x[1], reverse=True)

        return ranked[:top_k]
