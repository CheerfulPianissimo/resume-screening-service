import os
from .utils import extract_text
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util

import json

# Initialize OpenAI client and embedding model
_openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
_sim_model = SentenceTransformer('all-MiniLM-L6-v2')

def _generate_resume_brief(text: str) -> str:
    prompt = (
        "You are a helpful assistant specialized in HR recruitment. "
        "Summarize the following resume into a brief professional summary of key skills, "
        "experience, and technologies used:\n\n"
        f"{text}\n\nBrief Summary:"
    )
    response = _openai_client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=300
    )
    return response.choices[0].message.content.strip()

def _calculate_similarity(text1: str, text2: str, model) -> float:
    emb1 = model.encode(text1, convert_to_tensor=True)
    emb2 = model.encode(text2, convert_to_tensor=True)
    return util.pytorch_cos_sim(emb1, emb2).item()

def _extract_skills_comma_separated(text):
    prompt = (
        "You are an HR specialist with technical expertisereviewing a candidate's resume.\n"
        "Your task is to extract 8 key technical skills mentioned in the resume based on the job description requirement.\n"
        "Return 15 maximum key technical skills in a comma-separated list, no extra text or punctuation.\n\n"
        f"Resume:\n{text}\n\n"
        "Skills:"
    )
    response = _openai_client.chat.completions.create(
        model="gpt-4.1-mini",  # or gpt-3.5-turbo for lower cost
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=200
    )
    return response.choices[0].message.content.strip()

def _extract_skills_json(text: str) -> list:
    prompt = (
        "You are a helpful assistant specialized in HR recruitment. "
        "Extract the four most relevant skills from the following resume text and output them as a JSON list named skills, with no additional commentary:\n\n"
        f"{text}\n\n"
        "Output JSON format, e.g.: {\"skills\": [\"skill1\", \"skill2\", \"skill3\", \"skill4\"]}"
    )
    response = _openai_client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=60
    )
    content = response.choices[0].message.content.strip()
    try:
        data = json.loads(content)
        return data.get("skills", [])[:4]
    except:
        return []

def score_resume(pdf_bytes: bytes, description: str) -> dict:
    """
    Screen a resume PDF against a job description by:
      1. Extracting text from PDF bytes
      2. Generating a resume summary via OpenAI
      3. Computing a semantic similarity score against the description
    Returns a dict with:
      - summary: generated summary
      - score: similarity float
      - details: lengths of text and description
    """
    # 1. Extract raw text
    text = extract_text(pdf_bytes)
    print(f"Extracted {len(text)} characters from resume PDF.")
    
    # Generate brief summary via OpenAI
    resume_summary = _generate_resume_brief(text)
    print(f"Generated resume summary of {len(resume_summary)} characters.")

    # Extract skills from the resume brief text for similarity comparison
    skills_resume_for_similarity = _extract_skills_comma_separated(text)
    print(f"Extracted skills from resume for similarity: {skills_resume_for_similarity}")

    # Generate job description summary and skills for similarity comparison
    jd_summary = _generate_resume_brief(description)
    print(f"Generated JD summary of {len(jd_summary)} characters.")
    skills_jd_for_similarity = _extract_skills_comma_separated(description)
    print(f"Extracted skills from JD for similarity: {skills_jd_for_similarity}")

    # Compute similarity score between resume summary and job description summary
    similarity_summary = _calculate_similarity(resume_summary, jd_summary, _sim_model)
    print(f"Similarity score (summary): {similarity_summary:.3f}")

    # Compute similarity score between skills in resume and skills in jd
    similarity_skills = _calculate_similarity(skills_resume_for_similarity, skills_jd_for_similarity, _sim_model)
    print(f"Similarity score (skills): {similarity_skills:.3f}")

    # Compute weighted sum of similarity scores
    weighted_similarity = (0.3 * similarity_summary) + (0.7 * similarity_skills)
    print(f"Weighted similarity score: {weighted_similarity:.3f}")

    # Extract top 4 skills using the original function for the return value
    skills = _extract_skills_json(text) # Use original _extract_skills
    print(f"Extracted skills (original): {skills}")
    skills_str = ", ".join(skills)
    
    return {
        "resumeScore": weighted_similarity,
        "skills": skills_str
    }
