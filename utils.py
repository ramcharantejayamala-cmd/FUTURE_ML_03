import re
import nltk

nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"http\S+", " ", text)

    text = re.sub(r"[^a-zA-Z ]", " ", text)

    words = text.split()

    words = [word for word in words if word not in stop_words]

    return " ".join(words)


skills = [
    "python",
    "java",
    "c++",
    "sql",
    "excel",
    "machine learning",
    "deep learning",
    "tensorflow",
    "keras",
    "nlp",
    "flask",
    "django",
    "git",
    "aws",
    "azure",
    "docker",
    "pandas",
    "numpy",
    "scikit",
    "statistics",
    "power bi",
    "tableau",
    "communication"
]


def extract_skills(text):

    text = text.lower()

    found = []

    for skill in skills:
        if skill in text:
            found.append(skill)

    return found