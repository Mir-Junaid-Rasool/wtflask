from flask import Flask, render_template, request, jsonify
import pandas as pd
from joblib import load
from sklearn.preprocessing import StandardScaler
import numpy as np
from flask_cors import CORS
# Load the trained model
model = load('water_treatment_model.joblib')
scaler = load('scaler.joblib')
#print(type(scaler))
regressor = load('water_treatment_regressor.joblib')

# Create Flask app
app = Flask(__name__)
CORS(app)

# @app.route('/predict')
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    Am = data['Am']
    COD = data['COD']
    TN = data['TN']

    # Create DataFrame from input
    input_data = pd.DataFrame([[Am, COD, TN]], columns=['Am', 'COD', 'TN'])
    input_scaled_data = scaler.transform(input_data)

    # Make predictions
    prediction = model.predict(input_scaled_data)[0]
    reg_pred = regressor.predict(input_scaled_data)[0]

    return jsonify({
        'classification_prediction': prediction,
        'regression_prediction': reg_pred
    })

if __name__ == '__main__':
    app.run(debug=True)

# def home():
#     return jsonify(message="Hello from Flask!")

# if __name__ == '__main__':
#     app.run(debug=True)
