from flask import Flask, request, jsonify # type: ignore
from flask_cors import CORS # type: ignore
from inference_script import InferenceEngine

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

print("Loading AI engine...")
engine = InferenceEngine()


@app.route("/")
def home():
    return app.send_static_file("index.html")


@app.route("/diagnose", methods=["POST"])
def diagnose():

    data = request.get_json()
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No input text provided"}), 400

    result = engine.diagnose(text)
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)