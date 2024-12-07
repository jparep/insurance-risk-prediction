import os
from flask import Flask
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env

def create_app():
    app = Flask(__name__)
    
    # Use the environemnt variabel or a fallback for local dev
    app.config['SECRETE_KEY'] = os.getenv('SECRETE_KEY', 'fallback-secret-key')
    
    from .routes import main
    app.register_blueprint(main)
    
    return app