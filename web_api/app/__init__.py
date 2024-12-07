from flask import Flask

def create_app():
    app = Flask(__name__)
    app.config(['SECRETE_KEY'] = 'your-secrete-key')
    
    from .routes import main
    app.register_blueprint(main)
    
    return app