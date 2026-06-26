# 🤖 AI Resume Screening System

## 📌 Project Overview

The **AI Resume Screening System** is a Machine Learning project that helps recruiters automatically rank resumes based on a given Job Description.

The system uses **Natural Language Processing (NLP)** techniques such as **TF-IDF Vectorization** and **Cosine Similarity** to calculate how closely each resume matches the job description.

This project was developed as part of the **Future Interns Machine Learning Task 3 (2026)**.

---

# 🚀 Features

* Upload and analyze resume dataset
* Select resume category
* Paste any Job Description
* Rank resumes using TF-IDF
* Calculate Cosine Similarity
* Display Top 10 Matching Candidates
* Highlight Best Candidate
* Resume Preview
* Download Ranked Candidates as CSV
* Interactive Streamlit Web Application
* Candidate Score Visualization

---

# 🛠 Technologies Used

* Python
* Pandas
* Scikit-learn
* Streamlit
* Matplotlib

---

# 📂 Project Structure

```
FUTURE_ML_03
│
├── data
│   ├── Resume.csv
│   └── job_description.txt
│
├── output
│   ├── ranked_candidates.csv
│   └── top10_scores.png
│
├── screenshots
│   ├── home_page.png
│   ├── results.png
│   ├── best_candidate.png
│   └── bar_chart.png
│
├── app.py
├── main.py
├── utils.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# ⚙ Installation

Clone the repository

```bash
git clone https://github.com/your-username/FUTURE_ML_03.git
```

Move into the project folder

```bash
cd FUTURE_ML_03
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the Streamlit app

```bash
streamlit run app.py
```

---

# ▶️ How to Use

1. Launch the Streamlit application.
2. Select the desired resume category.
3. Paste the Job Description.
4. Click **Find Best Candidate**.
5. View the ranked candidates.
6. Download the ranked results as CSV.

---

# 📊 Machine Learning Workflow

* Resume Dataset Loading
* Category Filtering
* Text Cleaning
* TF-IDF Vectorization
* Cosine Similarity Calculation
* Resume Ranking
* Best Candidate Selection
* Visualization
* CSV Export

---

# 📷 Screenshots

### Home Page

(Add home_page.png)

### Top Matching Candidates

(Add results.png)

### Best Candidate

(Add best_candidate.png)

### Similarity Score Chart

(Add bar_chart.png)

---

# 🎯 Future Improvements

* Resume Upload Feature
* PDF Resume Parsing
* Skill Extraction using NLP
* BERT/Sentence Transformers based Matching
* Recruiter Dashboard
* Candidate Skill Gap Analysis

---

# 👨‍💻 Author

**Ram Charan Teja Yamala**

Machine Learning Enthusiast

Future Interns – ML Task 3 (2026)
