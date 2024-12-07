from flask import Blueprint, render_template, request
import joblib
import pandas as pd

main = Blueprint('main', __name__)

# Load the pre-trained model pipeline
model_pipeline = joblib.load('../model_development/models/model_pipeline.joblib')

@main.route('/')
def home():
    return render_template('index.html')

@main.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input from the form
        age = int(request.form['age'])
        bmi = float(request.form['bmi'])
        children = int(request.form['children'])
        smoker = 1 if request.form['smoker'] == 'yes' else 0  # Fixed typo in 'smkoer'
        sex = request.form['sex']
        region = request.form['region']

        # Prepare input as a DataFrame
        input_data = pd.DataFrame({
            'age': [age],
            'bmi': [bmi],
            'children': [children],
            'smoker': [smoker],
            'sex': [sex],
            'region': [region]
        })

        # Prediction using the loaded pipeline
        prediction = model_pipeline.predict(input_data)[0]

        return render_template('result.html', prediction=round(prediction, 2))

    except Exception as e:
        # Log or display the error for debugging
        return render_template('error.html', error_message=str(e))
