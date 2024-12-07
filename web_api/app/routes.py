from flask import Blueprint, render_template, request
import joblib

main = Blueprint('main', __name__)

# Load the pre-trained model
model = joblib.load('../model_development/models/model.joblib')

@main.route('/')
def home():
    return render_template('index.html')

@main.route('/predict', methods=['POST'])
def predict():
    # Get usr input from form
    age = int(request.form['age'])
    bmi = float(request.form['bmi'])
    children = int(request.form['children'])
    smoker = 1 if request.form['smoker'] == 'yes' else 0