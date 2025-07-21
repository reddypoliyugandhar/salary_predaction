from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import numpy as np
import os
import logging
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Threshold for salary correction
THRESHOLD_SALARY = 1000

# Define custom transformer
class SalaryCorrector(BaseEstimator, TransformerMixin):
    """Corrects salary values that appear to be missing zeros."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if 'Salary' in X.columns:
            X['Salary'] = X['Salary'].apply(lambda x: x if x > THRESHOLD_SALARY else x * 100)
        return X

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load model
model_path = 'model/salary_predictor_corrected.pkl'
model = None

def load_model():
    global model
    try:
        model = joblib.load(model_path)
        logger.info("Model loaded successfully.")
    except FileNotFoundError:
        logger.error(f"Model file not found at: {model_path}")
    except Exception as e:
        logger.exception(f"Failed to load model: {e}")

# Prediction logic
def make_prediction(input_data):
    if model is None:
        return None, "Model not loaded"

    try:
        df = pd.DataFrame([input_data])
        prediction = model.predict(df)
        return round(prediction[0], 2), None
    except Exception as e:
        logger.exception("Prediction failed")
        return None, str(e)

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = {
            'Age': int(request.form['age']),
            'Gender': request.form['gender'],
            'Education Level': request.form['education'],
            'Years of Experience': int(request.form['experience']),
            'Job Title': request.form['job_title']
        }
        prediction, error = make_prediction(data)
        if error:
            return jsonify({'success': False, 'error': error}), 500
        return jsonify({'success': True, 'prediction': prediction, 'input_data': data})
    except Exception as e:
        logger.exception("Form prediction failed")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()
        required_fields = ['Age', 'Gender', 'Education Level', 'Years of Experience', 'Job Title']

        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({'success': False, 'error': f"Missing fields: {', '.join(missing_fields)}"}), 400

        prediction, error = make_prediction(data)
        if error:
            return jsonify({'success': False, 'error': error}), 500
        return jsonify({'success': True, 'prediction': prediction, 'input_data': data})
    except Exception as e:
        logger.exception("API prediction failed")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })

# Main entry point
if __name__ == '__main__':
    os.makedirs('model', exist_ok=True)
    load_model()

    if model is None:
        logger.error("Model not loaded. Exiting...")
        exit(1)

    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)

