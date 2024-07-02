from flask import Flask, request, jsonify
import pandas as pd
from joblib import load
from flask_cors import CORS

# Load the trained models and scaler
model = load('water_treatment_model.joblib')
scaler = load('scaler.joblib')
regressor = load('water_treatment_regressor.joblib')
codmodel = load('cod_prediction_model.joblib')
codscaler = load('codscaler.joblib')

# Create Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    Am = data['Am']
    COD = data['COD']
    TN = data['TN']

    input_data = pd.DataFrame([[Am, COD, TN]], columns=['Am', 'COD', 'TN'])
    input_scaled_data = scaler.transform(input_data)

    prediction = model.predict(input_scaled_data)[0]
    reg_pred = regressor.predict(input_scaled_data)[0]

    return jsonify({
        'classification_prediction': prediction,
        'regression_prediction': reg_pred
    })

@app.route('/codpredict', methods=['POST'])
def codpredict():
    try:
        data = request.get_json()
        BOD = data['BOD']
        TN = data['TN']
        Am = data['Am']
        avg_inflow = data['avg_inflow']
        avg_outflow = data['avg_outflow']
        T = data['T']
        H = data['H']
        PP = data['PP']

        input_data = pd.DataFrame([[BOD, TN, Am, avg_inflow, avg_outflow, T, H, PP]], 
                                  columns=['BOD', 'TN', 'Am', 'avg_inflow', 'avg_outflow', 'T', 'H', 'PP'])
        
        input_scaled_data = codscaler.transform(input_data)
        prediction = codmodel.predict(input_scaled_data)[0]

        return jsonify({
            'classification_prediction': prediction
        })
    except Exception as e:
        print(f"Error occurred: {e}")
        return jsonify({'error': str(e)}), 500
if __name__ == '__main__':
    app.run(debug=True)
