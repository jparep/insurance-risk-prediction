from flask import Blueprint, render_template, request
import joblib
import pandas as pd

main = Blueprint('main', __name__)

# Load the pre-trained model pipeline
model_pipeline = joblib.load('./model_development/models/model_pipeline.joblib')

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
        smoker = 1 if request.form['smoker'] == 'yes' else 0
        sex = request.form['sex']
        region = request.form['region']

        # Prepare input data
        input_data = {
            'age': [age],
            'bmi': [bmi],
            'children': [children],
            'smoker': [smoker],
            'sex': [sex],
            'region': [region],
        }

        # Predict using the model pipeline
        prediction = model_pipeline.predict(pd.DataFrame(input_data))[0]

        # Format prediction with comma as a thousands separator
        formatted_prediction = f"{prediction:,.2f}"

        return render_template('result.html', prediction=formatted_prediction)
    
    except Exception as e:
        return f"An error occurred: {str(e)}"
