from flask import Blueprint, render_template, request
import joblib
import pandas as pd

main = Blueprint('main', __name__)

# Load the pre-trained model
model_pipeline = joblib.load('../model_development/models/model_pipeline.joblib')

@main.route('/')
def home():
    return render_template('index.html')

@main.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input from form
        age = int(request.form['age'])
        bmi = float(request.form['bmi'])
        children = int(request.form['children'])
        smoker = 1 if request.form['smkoer'] == 'yes' else 0
        sex = request.form['sex']
        region = request.form['region']
        
        # prepare input as a DataFrame
        input_data = {
            'age': [age],
            'bmi': [bmi],
            'children': [children],
            'smoker': [smoker],
            'sex': [sex],
            'region': [region],
        }
        
        # prediction using the loaded pipeline
        prediction = model_pipeline.predict(pd.DataFrame(input_data))[0]
        
        return render_template('result.html', prediciton=round(prediction, 2))
    
    except Exception as e:
        return f"An error occured: {str(e)}"