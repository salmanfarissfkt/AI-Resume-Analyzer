from flask import Flask,render_template,request
import os
import re
from pypdf import PdfReader
from embedding import get_embedding
from sklearn.metrics.pairwise import cosine_similarity


app = Flask(__name__)

UPLOAD_FOLDER="uploads"
app.config['UPLOAD_FOLDER']=UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {"pdf"}
SKILLS =   [
    "python",
    "java",
    "c",
    "c++",
    "sql",
    "mysql",
    "flask",
    "fastapi",
    "django",
    "html",
    "css",
    "javascript",
    "react",
    "machine learning",
    "deep learning",
    "natural language processing",
    "nlp",
    "tensorflow",
    "pytorch",
    "git",
    "github",
    "docker",
    "rest api",
    "rest apis",
    "data analysis",
    "data science",
    "pandas",
    "numpy"
]
REQUIRED_SKILLS = [
    "python",
    "flask",
    "fastapi",
    "sql",
    "mysql",
    "machine learning",
    "nlp",
    "natural language processing",
    "git",
    "github"
]

PREFERRED_SKILLS = [
    "tensorflow",
    "pytorch",
    "docker",
    "react",
    "pandas",
    "numpy",
    "rest api"
]
SKILL_ALIASES = {
    "nlp": "natural language processing",
    "natural language processing": "natural language processing",
    "rest api": "rest api",
    "rest apis": "rest api",
    "ml": "machine learning",
    "machine learning": "machine learning",
}
SECTIONS = ["professional summary","techncal skill","education",   "projects","technical & academic projects",
    "internship","experience","work experience", "leadership & responsibilities", "soft skills"]
def allowed_filename(filename):
    return "." in filename and \
    filename.rsplit(".",1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(pdf_path):
    reader =PdfReader(pdf_path)
    text=""
    for page in reader.pages:
        text+=page.extract_text() or ""
    return text

def extract_skills(text):
    text=text.lower()
    found_skills=[]
    for skill in SKILLS:
        if skill in text:
            pattern =r"\b"+ re.escape(skill.lower())+r"\b"
            if re.search(pattern,text):
                normalized_skill = SKILL_ALIASES.get(skill, skill)
                if normalized_skill not in found_skills:
                  found_skills.append(normalized_skill)
    return found_skills

def get_relevant_resume_text(resume_text):
    sections = []

    resume_lower = resume_text.lower()

    # Professional Summary
    if "professional summary" in resume_lower:
        start = resume_lower.find("professional summary")
        sections.append(resume_text[start:start + 1000])

    # Technical Skills
    if "technical skills" in resume_lower:
        start = resume_lower.find("technical skills")
        sections.append(resume_text[start:start + 1000])

    # Projects
    if "technical & academic projects" in resume_lower:
        start = resume_lower.find("technical & academic projects")
        sections.append(resume_text[start:start + 3000])
    elif "projects" in resume_lower:
        start = resume_lower.find("projects")
        sections.append(resume_text[start:start + 3000])

    # Internship
    if "internship" in resume_lower:
        start = resume_lower.find("internship")
        sections.append(resume_text[start:start + 1500])

    return " ".join(sections)    

def calculate_ai_match(resume_text, job_description,normalize_embeddings=True):

    relevant_resume = get_relevant_resume_text(resume_text)

    resume_embedding = get_embedding(relevant_resume)

    job_embedding = get_embedding(job_description)

    similarity = cosine_similarity(
        [resume_embedding],
        [job_embedding]
    )[0][0]

    match_score = round(similarity * 100)

    return match_score   

def calculate_project_relevance(projects, job_description):

    if not projects.strip():
        return 0

    project_embedding = get_embedding(projects)
    job_embedding = get_embedding(job_description)

    similarity = cosine_similarity(
        [project_embedding],
        [job_embedding]
    )[0][0]

    project_score = round(similarity * 100)

    return project_score

def match_job_skills(skills, job_description):

    job_skills = extract_skills(job_description)

    matched_skills = []
    missing_skills = []

    for skill in job_skills:
        if skill in skills:
            matched_skills.append(skill)
        else:
            missing_skills.append(skill)

    required_job_skills = []
    preferred_job_skills = []

    for skill in job_skills:

        if skill in REQUIRED_SKILLS:
            required_job_skills.append(skill)

        elif skill in PREFERRED_SKILLS:
            preferred_job_skills.append(skill)

    matched_required = []

    for skill in required_job_skills:
        if skill in skills:
            matched_required.append(skill)

    matched_preferred = []

    for skill in preferred_job_skills:
        if skill in skills:
            matched_preferred.append(skill)

    if len(required_job_skills) > 0:
        required_score = (
            len(matched_required) / len(required_job_skills)
        ) * 70
    else:
        required_score = 0

    if len(preferred_job_skills) > 0:
        preferred_score = (
            len(matched_preferred) / len(preferred_job_skills)
        ) * 30
    else:
        preferred_score = 0

    match_score = int(required_score + preferred_score)

    return (
        job_skills,
        matched_skills,
        missing_skills,
        match_score
    )     

def detect_sections(text):
    text=text.lower()
    found_sections=[]
    for section in SECTIONS:
        if section in text:
            found_sections.append(section)
    return found_sections        
def extract_section(text, start_section, end_sections):

    lines = text.splitlines()

    start_index = -1

    for i, line in enumerate(lines):

        if line.strip().lower() == start_section.lower():
            start_index = i
            break

    if start_index == -1:
        return ""

    section_lines = []

    for line in lines[start_index + 1:]:

        clean_line = line.strip().lower()

        if clean_line in [section.lower() for section in end_sections]:
            break

        section_lines.append(line)

    return "\n".join(section_lines).strip()      


def clean_text(text):
    text = text.replace("•","")
    text = text.replace("◦","")
    text = text.replace("–","-")

    text = text.replace("\n", " ")

    text = " ".join(text.split())

    return text.strip()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload",methods=["POST"])    
def upload():
    file = request.files["resume"]
    job_description = request.form ["job_description"]
    
    

    if file.filename=="":
        return "no file  selected...!"

    if not allowed_filename(file.filename):
        return "only pdf file is allowed!!!"

    file_path =os.path.join(app.config['UPLOAD_FOLDER'],file.filename)
    file.save(file_path)
    resume_text=extract_text_from_pdf(file_path)
    skills = extract_skills(resume_text)
    sections =detect_sections(resume_text)
    job_skills, matched_skills, missing_skills, match_score = match_job_skills(skills,job_description
)   
    ai_match_score = calculate_ai_match(resume_text,job_description)
    project_relevance_score = calculate_project_relevance(projects,job_description)
    final_match_score =round( (match_score* 0.6)+(ai_match_score*0.4)) 
    education =extract_section(resume_text,"education",["technical & academic projects",
        "projects","internship", "experience","leadership & responsibilities","soft skills"])
    
    projects = extract_section( resume_text,
    "technical & academic projects",
    [
        "internship",
        "experience",
        "leadership & responsibilities",
        "soft skills"
    ]
)

    internship = extract_section(resume_text,
    "internship",
    [
        "leadership & responsibilities",
        "soft skills"
    ]
)
    education = clean_text(education)    
    projects = clean_text(projects)    
    internship = clean_text(internship)   
    recommendations = generate_recommendations (skills,sections,projects) 
    score,summary_score,skill_score,education_score,projects_score,internship_score,leadership_score,softskill_score= calculate_score(skills,sections,projects)

    return render_template("result.html",filename=file.filename,resume_text=resume_text,
    skills=skills,sections=sections,education=education,projects=projects,internship=internship,
    score = score,
    summary_score=summary_score,
    skill_score=skill_score,
    education_score=education_score,
    projects_score=projects_score,
    internship_score=internship_score,
    leadership_score=leadership_score,
    softskill_score=softskill_score,
    recommendations = recommendations,
    job_description = job_description,
    job_skills=job_skills,
    matched_skills=matched_skills,
    missing_skills=missing_skills,
    match_score=match_score,
    ai_match_score=ai_match_score,
    project_relevance_score = project_relevance_score,
    final_match_score=final_match_score)

def generate_recommendations(skills,sections,projects):
    recommendations=[]

    if len(skills) < 5:
        recommendations.append("add more relevent skills to your resume")

    if "professional summary" not in sections:
       recommendations.append("add professional summary at the begining of resume")
    
    if "education" not in sections:
        recommendations.append("Add an education section with your degree and institution.")

    if "projects" not in sections and "technical & academic projects" not in sections:
        recommendations.append(
            "Add technical projects to demonstrate your practical skills."
        )

    if len(projects) < 250:
        recommendations.append(
            "Add more details and achievements to your project descriptions."
        )

    if "internship" not in sections and "experience" not in sections:
        recommendations.append(
            "Consider adding internship or work experience."
        )

    if "soft skills" not in sections:
        recommendations.append(
            "Add relevant soft skills such as teamwork and communication."
        )

    if len(recommendations) == 0:
        recommendations.append(
            "Your resume has a strong overall structure. Keep improving your project details and skills."
        )

    return recommendations
def calculate_score(skills,sections,projects):
    score = 0
    summary_score = 0
    skill_score = 0
    education_score = 0
    projects_score = 0
    internship_score=0
    leadership_score = 0
    softskill_score = 0
    if "professional summary" in sections:
        summary_score = 10

    skill_score = min(len(skills) * 2, 25)
    
          

    if "education" in sections:
        education_score= 20

    if "projects" in sections or "technical & academic projects" in sections:
        if len(projects) > 500:
            projects_score = 20
        elif len(projects) > 250:
           projects_score= 15
        else :
            projects_score =10        
 
    if "internship" in sections or "experience" in sections:
        internship_score = 15

    if "leadership & responsibilities" in sections:
        leadership_score = 5

    if "soft skills" in sections:
        softskill_score = 5
    score = (summary_score + skill_score + education_score + projects_score + internship_score + leadership_score + softskill_score)
    return score,summary_score,skill_score,education_score,projects_score,internship_score,leadership_score,softskill_score

@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == "__main__":
    app.run(debug=True)

