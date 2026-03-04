from flask import Flask, request, jsonify # type: ignore
from flask_cors import CORS # type: ignore
from inference_script import InferenceEngine

# Tell Flask to serve static files from current directory
app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

print("Loading AI engine...")
engine = InferenceEngine()


# ✅ Serve index.html properly
@app.route("/")
def home():
    return app.send_static_file("index.html")


# ✅ Diagnose endpoint (unchanged logic)
@app.route("/diagnose", methods=["POST"])
def diagnose():

    data = request.get_json()
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No input text provided"}), 400

    positive, negative = engine.extractor.extract(text)

    if len(positive) == 0:
        return jsonify({"result": "No recognizable symptoms."})

    ranked = engine.hybrid.diagnose(
        input_symptoms=positive,
        symptom_text=text,
        top_k=3
    )

    results = []

    for disease_id, score in ranked:
        name = engine.disease_id_to_name.get(disease_id, disease_id)
        results.append({
            "disease": name,
            "score": float(score)
        })

    return jsonify({"results": results})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)