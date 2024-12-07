import os
from flask import Flask
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

def create_app():
    app = Flask(__name__)
    
    # Use the environment variable or a fallback for local development
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'fallback-secret-key')  # Fixed typo: SECRETE_KEY -> SECRET_KEY
    
    from .routes import main
    app.register_blueprint(main)
    
    return app
