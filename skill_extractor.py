DEFAULT_SKILLS = [
    "python","java","c++","sql","machine learning","deep learning",
    "scikit-learn","pandas","numpy","tensorflow","pytorch","nlp",
    "transformers","bert","docker","kubernetes","aws","azure","gcp",
    "git","linux","jenkins","ci/cd","mlflow","power bi","tableau",
    "excel","react","node.js","express","mongodb","mysql","django",
    "flask","rest api","html","css","bootstrap","statistics"
]

def extract_skills(text, skills=None):
    text_l = text.lower()
    skills = skills or DEFAULT_SKILLS
    found = []
    for skill in skills:
        if skill.lower() in text_l:
            found.append(skill)
    return sorted(set(found))
