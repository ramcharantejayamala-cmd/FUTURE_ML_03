import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page Configuration
st.set_page_config(
    page_title="Resume Screening System",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Resume Screening System")
st.write("Rank resumes based on a Job Description using TF-IDF and Cosine Similarity.")

# Load Dataset
df = pd.read_csv("data/Resume.csv")

st.success(f"Loaded {len(df)} resumes successfully!")

# Category Selection
categories = sorted(df["Category"].unique())

category = st.selectbox(
    "Select Resume Category",
    categories
)

# Job Description Input
job_description = st.text_area(
    "Paste Job Description",
    height=250,
    placeholder="Paste the complete job description here..."
)

# Button
if st.button("Find Best Candidate"):

    if job_description.strip() == "":
        st.warning("Please enter a Job Description.")
        st.stop()

    # Filter resumes by category
    filtered_df = df[df["Category"] == category].copy()

    if filtered_df.empty:
        st.error("No resumes found for this category.")
        st.stop()

    # Resume text column is Resume_str
    resumes = filtered_df["Resume_str"].fillna("").tolist()

    # Combine Job Description + Resumes
    documents = [job_description] + resumes

    # TF-IDF
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=5000
    )

    tfidf_matrix = vectorizer.fit_transform(documents)

    # Cosine Similarity
    similarity_scores = cosine_similarity(
        tfidf_matrix[0:1],
        tfidf_matrix[1:]
    ).flatten()

    # Add scores
    filtered_df["Similarity Score"] = (similarity_scores * 100).round(2)

    # Sort candidates
    filtered_df = filtered_df.sort_values(
        by="Similarity Score",
        ascending=False
    )

    # Top 10
    top10 = filtered_df[["ID", "Category", "Similarity Score"]].head(10)

    st.subheader("🏆 Top 10 Matching Candidates")
    st.dataframe(top10, use_container_width=True)

    # Best Candidate
    best = filtered_df.iloc[0]

    st.subheader("✅ Best Candidate")
    st.write(f"**Resume ID:** {best['ID']}")
    st.write(f"**Category:** {best['Category']}")
    st.write(f"**Similarity Score:** {best['Similarity Score']:.2f}%")

    # Chart
    st.subheader("📊 Top 10 Candidate Scores")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(top10["ID"].astype(str), top10["Similarity Score"])
    ax.set_xlabel("Candidate ID")
    ax.set_ylabel("Similarity Score (%)")
    ax.set_title("Top 10 Candidate Scores")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Resume Preview
    st.subheader("📄 Best Candidate Resume Preview")
    st.write(best["Resume_str"][:3000])

    # Download CSV
    csv = filtered_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Ranked Candidates CSV",
        data=csv,
        file_name="ranked_candidates.csv",
        mime="text/csv"
    )