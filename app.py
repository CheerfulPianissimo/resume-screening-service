from flask import Flask
from resume_screening.routes import screening_blueprint

def create_app():
    app = Flask(__name__)
    app.register_blueprint(screening_blueprint, url_prefix='/analyze_resume')
    
    @app.errorhandler(Exception)
    def handle_exception(e):
        return {"error": str(e)}, 500

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host='::', port=5000)
