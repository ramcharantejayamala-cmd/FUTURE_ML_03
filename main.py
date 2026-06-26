import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils import clean_text, extract_skills

# Load dataset
df = pd.read_csv("data/Resume.csv")

# Show available categories
print("Available Categories:")
print(df["Category"].unique())
category = input("\nEnter Category: ").strip()

# Filter dataset (case-insensitive)
df = df[df["Category"].str.upper() == category.upper()]

if df.empty:
    print("No resumes found for this category.")
    exit()

print(f"\nNumber of resumes in {category}: {len(df)}")

df = df[["ID", "Resume_str", "Category"]]
# Clean resume text
df["Clean_Resume"] = df["Resume_str"].apply(clean_text)

# Read Job Description
with open("data/job_description.txt", "r", encoding="utf-8") as file:
    job_description = file.read()

clean_job = clean_text(job_description)

# TF-IDF
vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    max_features=5000
)

vectors = vectorizer.fit_transform(
    [clean_job] + df["Clean_Resume"].tolist()
)
# Similarity
scores = cosine_similarity(vectors[0:1], vectors[1:]).flatten()

df["Score"] = (scores * 100).round(2)

# Resume Skills
df["Resume Skills"] = df["Resume_str"].apply(extract_skills)

# JD Skills
jd_skills = extract_skills(job_description)

# Missing Skills
df["Missing Skills"] = df["Resume Skills"].apply(
    lambda x: list(set(jd_skills) - set(x))
)
# Matched Skills
df["Matched Skills"] = df["Resume Skills"].apply(
    lambda x: list(set(jd_skills).intersection(set(x)))
)

# Number of matched skills
df["Matched Count"] = df["Matched Skills"].apply(len)

# Number of missing skills
df["Missing Count"] = df["Missing Skills"].apply(len)

# Rank
df = df.sort_values("Score", ascending=False)

# Top 10
top = df.head(10)

print("\n==============================")
print("Top 10 Candidates")
print("==============================\n")

print("\n========== TOP 10 CANDIDATES ==========\n")

for i, (_, row) in enumerate(top.iterrows(), start=1):
    print(f"\nCandidate {i}")
    print("-" * 40)
    print(f"ID: {row['ID']}")
    print(f"Category: {row['Category']}")
    print(f"Similarity Score: {row['Score']:.2f}%")

    print("Matched Skills:", ", ".join(row["Matched Skills"]) if row["Matched Skills"] else "None")
    print("Resume Skills:", ", ".join(row["Resume Skills"]) if row["Resume Skills"] else "None")
    print("Missing Skills:", ", ".join(row["Missing Skills"]) if row["Missing Skills"] else "None")

    print(f"Matched Count: {row['Matched Count']}")
    print(f"Missing Count: {row['Missing Count']}")

best = top.iloc[0]

print("\n==============================")
print("Recommended Candidate")
print("==============================")

print(f"ID: {best['ID']}")
print(f"Category: {best['Category']}")
print(f"Similarity Score: {best['Score']:.2f}%")
print("Matched Skills:", ", ".join(best["Matched Skills"]))
print("Missing Skills:", ", ".join(best["Missing Skills"]))
# Save
os.makedirs("output", exist_ok=True)

result = df[
    [
        "ID",
        "Category",
        "Score",
        "Matched Skills",
        "Matched Count",
        "Missing Skills",
        "Missing Count"
    ]
]

result.to_csv(
    "output/ranked_candidates.csv",
    index=False
)

print("\nResults saved to output/ranked_candidates.csv")
import matplotlib.pyplot as plt

top10 = df.head(10)

plt.figure(figsize=(10,6))
plt.bar(top10["ID"].astype(str), top10["Score"])

plt.xticks(rotation=45)
plt.title("Top 10 Candidate Scores")
plt.xlabel("Candidate ID")
plt.ylabel("Similarity Score (%)")

plt.tight_layout()
plt.savefig("output/top10_scores.png")
plt.show()
