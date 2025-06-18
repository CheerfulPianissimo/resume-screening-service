from flask import Blueprint, request, jsonify
from .service import score_resume

screening_blueprint = Blueprint('screening', __name__)

@screening_blueprint.route('/', methods=['POST'])
def screen_resume():
    # Expect multipart/form-data with 'resume' file and 'description' text
    if 'resume' not in request.files or 'description' not in request.form:
        print("Missing resume file or job description in request")
        return jsonify({'error': 'Missing resume file or job description'}), 400

    resume_file = request.files['resume']
    description = request.form['description']
    pdf_bytes = resume_file.read()
    print(f"Received resume file of size: {len(pdf_bytes)} bytes with description length: {len(description)} characters")
    result = score_resume(pdf_bytes, description)
    return jsonify(result), 200
