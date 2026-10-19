"""Flask application for churn model inference."""

from __future__ import annotations

from flask import Flask, request, jsonify

from api.schemas import PredictionRequest, PredictionResponse
from src.serving.predict import make_prediction

app = Flask(__name__)
app.config['TITLE'] = "Customer Churn Prediction API"
app.config['DESCRIPTION'] = "Flask service for telecom customer churn prediction."
app.config['VERSION'] = "0.1.0"


@app.route("/")
def read_root():
    """Health-style root endpoint."""
    return jsonify({"message": "Customer Churn Prediction API is running."})


@app.route("/predict", methods=["POST"])
def predict():
    """Generate a churn prediction from engineered feature inputs."""
    try:
        req_data = request.get_json()
        features = req_data.get("features")
        
        prediction, probability = make_prediction(features)
        prediction_label = "Yes" if prediction == 1 else "No"

        return jsonify({
            "prediction": prediction,
            "prediction_label": prediction_label,
            "churn_probability": probability
        })
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 500
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400