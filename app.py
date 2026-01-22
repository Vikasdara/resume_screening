import streamlit as st
import pickle
import re
import PyPDF2
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


# ---------------- LOAD FILES ----------------
model = pickle.load(open("resume_model.pkl", "rb"))
tfidf = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))
job_role_skills = pickle.load(open("job_role_skills.pkl", "rb"))

# ---------------- FUNCTIONS ----------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="Resume Screening System", page_icon="📄")

st.title("📄 Resume Screening System")
st.write("Upload your resume to predict job role and fit score.")

uploaded_file = st.file_uploader(
    "Upload Resume (PDF or TXT)",
    type=["pdf", "txt"]
)

if uploaded_file:
    if uploaded_file.type == "application/pdf":
        resume_text = extract_text_from_pdf(uploaded_file)
    else:
        resume_text = uploaded_file.read().decode("utf-8")

    cleaned_text = clean_text(resume_text)

    # -------- Job Role Prediction --------
    resume_vec = tfidf.transform([cleaned_text])
    prediction = model.predict(resume_vec)
    predicted_role = le.inverse_transform(prediction)[0]

    # -------- Match Score --------
    skills_text = " ".join(job_role_skills.get(predicted_role, []))

    if skills_text.strip():
        skills_vec = tfidf.transform([skills_text])
        score = cosine_similarity(resume_vec, skills_vec)[0][0]
        match_percentage = round(score * 100, 2)
    else:
        match_percentage = 0.0

    # -------- Fit Level --------
    if match_percentage >= 75:
        fit_level = "Good Match"
    elif match_percentage >= 50:
        fit_level = "Average Match"
    else:
        fit_level = "Poor Match"

    # -------- Missing Skills --------
    resume_words = set(cleaned_text.split())
    required_skills = job_role_skills.get(predicted_role, [])
    missing_skills = [
    s for s in required_skills
    if s not in resume_words
    and s not in ENGLISH_STOP_WORDS
    and len(s) > 3
][:5]


    # -------- OUTPUT --------
    st.success(f"✅ Predicted Job Role: **{predicted_role}**")
    st.info(f"📊 Resume–Job Match Score: **{match_percentage}%**")
    st.info(f"🎯 Fit Level: **{fit_level}**")

    if missing_skills:
        st.warning(f"⚠️ Missing Skills: {', '.join(missing_skills)}")
    else:
        st.success("🎉 All key skills are present!")
