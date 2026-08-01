"""
Generates a synthetic dataset that mirrors the structure and relationships
of the Kaggle "Student Performance Factors" dataset:
https://www.kaggle.com/datasets/lainguyn123/student-performance-factors

(Kaggle isn't reachable from this environment, so we reproduce its schema
and realistic relationships/noise so the rest of the pipeline works the
same way it would on the real file. Swap in the real StudentPerformanceFactors.csv
if you have it — the rest of the code doesn't need to change.)
"""
import numpy as np
import pandas as pd

np.random.seed(42)
n = 1200

hours_studied = np.clip(np.random.normal(20, 6, n), 1, 44).round(0)
attendance = np.clip(np.random.normal(80, 10, n), 60, 100).round(0)
sleep_hours = np.clip(np.random.normal(7, 1.3, n), 4, 10).round(0)
previous_scores = np.clip(np.random.normal(75, 12, n), 40, 100).round(0)
tutoring_sessions = np.random.poisson(1.5, n).clip(0, 8)
physical_activity = np.clip(np.random.normal(3, 1.5, n), 0, 6).round(0)

parental_involvement = np.random.choice(["Low", "Medium", "High"], n, p=[0.25, 0.5, 0.25])
access_to_resources = np.random.choice(["Low", "Medium", "High"], n, p=[0.2, 0.5, 0.3])
extracurricular = np.random.choice(["Yes", "No"], n, p=[0.55, 0.45])
motivation_level = np.random.choice(["Low", "Medium", "High"], n, p=[0.3, 0.45, 0.25])
internet_access = np.random.choice(["Yes", "No"], n, p=[0.85, 0.15])
family_income = np.random.choice(["Low", "Medium", "High"], n, p=[0.3, 0.45, 0.25])
teacher_quality = np.random.choice(["Low", "Medium", "High"], n, p=[0.2, 0.55, 0.25])
school_type = np.random.choice(["Public", "Private"], n, p=[0.7, 0.3])
peer_influence = np.random.choice(["Positive", "Neutral", "Negative"], n, p=[0.4, 0.4, 0.2])
learning_disabilities = np.random.choice(["Yes", "No"], n, p=[0.1, 0.9])
gender = np.random.choice(["Male", "Female"], n, p=[0.5, 0.5])

# map ordinal categories to numeric effect sizes to build a realistic target
level_map = {"Low": -3, "Medium": 0, "High": 3}
yn_map = {"Yes": 1, "No": 0}
peer_map = {"Positive": 2, "Neutral": 0, "Negative": -3}

exam_score = (
    40
    + 1.15 * hours_studied
    + 0.18 * attendance
    + 0.20 * previous_scores
    + 0.6 * sleep_hours
    + 1.1 * tutoring_sessions
    + np.array([level_map[x] for x in parental_involvement])
    + np.array([level_map[x] for x in access_to_resources])
    + np.array([level_map[x] for x in motivation_level]) * 1.3
    + np.array([level_map[x] for x in teacher_quality])
    + np.array([peer_map[x] for x in peer_influence])
    + np.array([yn_map[x] for x in extracurricular]) * 1.5
    - np.array([yn_map[x] for x in learning_disabilities]) * 4
    + np.random.normal(0, 5, n)  # noise
)
exam_score = np.clip(exam_score, 35, 100).round(1)

df = pd.DataFrame({
    "Hours_Studied": hours_studied,
    "Attendance": attendance,
    "Parental_Involvement": parental_involvement,
    "Access_to_Resources": access_to_resources,
    "Extracurricular_Activities": extracurricular,
    "Sleep_Hours": sleep_hours,
    "Previous_Scores": previous_scores,
    "Motivation_Level": motivation_level,
    "Internet_Access": internet_access,
    "Tutoring_Sessions": tutoring_sessions,
    "Family_Income": family_income,
    "Teacher_Quality": teacher_quality,
    "School_Type": school_type,
    "Peer_Influence": peer_influence,
    "Physical_Activity": physical_activity,
    "Learning_Disabilities": learning_disabilities,
    "Gender": gender,
    "Exam_Score": exam_score,
})

# inject some realistic messiness for the "data cleaning" step
missing_idx = np.random.choice(df.index, size=40, replace=False)
df.loc[missing_idx[:15], "Teacher_Quality"] = np.nan
df.loc[missing_idx[15:28], "Parental_Involvement"] = np.nan
df.loc[missing_idx[28:], "Sleep_Hours"] = np.nan

dup_rows = df.sample(10, random_state=1)
df = pd.concat([df, dup_rows], ignore_index=True)

df.to_csv("StudentPerformanceFactors.csv", index=False)
print("Saved:", df.shape)
