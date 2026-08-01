import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sns.set_theme(style="whitegrid")
PLOTS = "plots"

# ---------------------------------------------------------------
# 1. LOAD + CLEAN
# ---------------------------------------------------------------
df = pd.read_csv("StudentPerformanceFactors.csv")
print("Raw shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum()[df.isnull().sum() > 0])

before = len(df)
df = df.drop_duplicates()
print(f"\nDropped {before - len(df)} duplicate rows")

# numeric NaNs -> median, categorical NaNs -> mode
num_cols = df.select_dtypes(include=np.number).columns
cat_cols = df.select_dtypes(exclude=np.number).columns

for c in num_cols:
    df[c] = df[c].fillna(df[c].median())
for c in cat_cols:
    df[c] = df[c].fillna(df[c].mode()[0])

print("\nMissing after cleaning:", df.isnull().sum().sum())
print("Final shape:", df.shape)
df.to_csv("StudentPerformanceFactors_clean.csv", index=False)

# ---------------------------------------------------------------
# 2. BASIC VISUALIZATION (EDA)
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(df["Exam_Score"], kde=True, ax=axes[0], color="#2563eb")
axes[0].set_title("Distribution of Exam Score")
sns.scatterplot(data=df, x="Hours_Studied", y="Exam_Score", alpha=0.5, ax=axes[1], color="#2563eb")
axes[1].set_title("Hours Studied vs Exam Score")
plt.tight_layout()
plt.savefig(f"{PLOTS}/01_eda_overview.png", dpi=150)
plt.close()

plt.figure(figsize=(9, 7))
corr = df.select_dtypes(include=np.number).corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Heatmap (numeric features)")
plt.tight_layout()
plt.savefig(f"{PLOTS}/02_correlation_heatmap.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 3. SIMPLE LINEAR REGRESSION: Exam_Score ~ Hours_Studied
# ---------------------------------------------------------------
X_simple = df[["Hours_Studied"]]
y = df["Exam_Score"]
Xs_train, Xs_test, ys_train, ys_test = train_test_split(X_simple, y, test_size=0.2, random_state=42)

simple_model = LinearRegression()
simple_model.fit(Xs_train, ys_train)
ys_pred = simple_model.predict(Xs_test)

simple_r2 = r2_score(ys_test, ys_pred)
simple_mae = mean_absolute_error(ys_test, ys_pred)
simple_rmse = mean_squared_error(ys_test, ys_pred) ** 0.5

print("\n=== Simple Linear Regression (Hours_Studied only) ===")
print(f"Intercept: {simple_model.intercept_:.2f}, Coef: {simple_model.coef_[0]:.3f}")
print(f"R2: {simple_r2:.3f}  MAE: {simple_mae:.2f}  RMSE: {simple_rmse:.2f}")

plt.figure(figsize=(7, 6))
plt.scatter(Xs_test, ys_test, alpha=0.5, label="Actual", color="#2563eb")
order = np.argsort(Xs_test["Hours_Studied"].values)
plt.plot(Xs_test["Hours_Studied"].values[order], ys_pred[order], color="#dc2626", linewidth=2, label="Predicted (fit)")
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.title("Simple Linear Regression: Hours Studied \u2192 Exam Score")
plt.legend()
plt.tight_layout()
plt.savefig(f"{PLOTS}/03_simple_regression_fit.png", dpi=150)
plt.close()

plt.figure(figsize=(6.5, 6))
plt.scatter(ys_test, ys_pred, alpha=0.5, color="#2563eb")
lims = [min(ys_test.min(), ys_pred.min()), max(ys_test.max(), ys_pred.max())]
plt.plot(lims, lims, color="#dc2626", linestyle="--", label="Perfect prediction")
plt.xlabel("Actual Exam Score")
plt.ylabel("Predicted Exam Score")
plt.title("Simple Model: Actual vs Predicted")
plt.legend()
plt.tight_layout()
plt.savefig(f"{PLOTS}/04_simple_actual_vs_predicted.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 4. MULTIPLE LINEAR REGRESSION (all features)
# ---------------------------------------------------------------
df_encoded = pd.get_dummies(df.drop(columns=["Exam_Score"]), drop_first=True)
y_full = df["Exam_Score"]

Xf_train, Xf_test, yf_train, yf_test = train_test_split(df_encoded, y_full, test_size=0.2, random_state=42)
full_model = LinearRegression()
full_model.fit(Xf_train, yf_train)
yf_pred = full_model.predict(Xf_test)

full_r2 = r2_score(yf_test, yf_pred)
full_mae = mean_absolute_error(yf_test, yf_pred)
full_rmse = mean_squared_error(yf_test, yf_pred) ** 0.5

print("\n=== Multiple Linear Regression (all features) ===")
print(f"R2: {full_r2:.3f}  MAE: {full_mae:.2f}  RMSE: {full_rmse:.2f}")

coef_series = pd.Series(full_model.coef_, index=df_encoded.columns).sort_values()
plt.figure(figsize=(8, 9))
coef_series.plot(kind="barh", color=np.where(coef_series > 0, "#2563eb", "#dc2626"))
plt.title("Feature Coefficients \u2014 Full Model")
plt.xlabel("Coefficient (effect on Exam Score)")
plt.tight_layout()
plt.savefig(f"{PLOTS}/05_full_model_coefficients.png", dpi=150)
plt.close()

plt.figure(figsize=(6.5, 6))
plt.scatter(yf_test, yf_pred, alpha=0.5, color="#059669")
lims = [min(yf_test.min(), yf_pred.min()), max(yf_test.max(), yf_pred.max())]
plt.plot(lims, lims, color="#dc2626", linestyle="--", label="Perfect prediction")
plt.xlabel("Actual Exam Score")
plt.ylabel("Predicted Exam Score")
plt.title("Full Model: Actual vs Predicted")
plt.legend()
plt.tight_layout()
plt.savefig(f"{PLOTS}/06_full_actual_vs_predicted.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 5. BONUS: Polynomial Regression (Hours_Studied) vs Linear
# ---------------------------------------------------------------
results = []
for degree in [1, 2, 3]:
    poly = PolynomialFeatures(degree=degree)
    Xp_train = poly.fit_transform(Xs_train)
    Xp_test = poly.transform(Xs_test)
    m = LinearRegression().fit(Xp_train, ys_train)
    pred = m.predict(Xp_test)
    results.append({
        "degree": degree,
        "R2": r2_score(ys_test, pred),
        "MAE": mean_absolute_error(ys_test, pred),
        "RMSE": mean_squared_error(ys_test, pred) ** 0.5,
    })
poly_results = pd.DataFrame(results)
print("\n=== Bonus: Polynomial Regression comparison (Hours_Studied) ===")
print(poly_results)

plt.figure(figsize=(8, 6))
x_range = np.linspace(Xs_train["Hours_Studied"].min(), Xs_train["Hours_Studied"].max(), 200).reshape(-1, 1)
plt.scatter(Xs_test, ys_test, alpha=0.35, color="gray", label="Test data")
colors = {1: "#2563eb", 2: "#059669", 3: "#dc2626"}
for degree in [1, 2, 3]:
    poly = PolynomialFeatures(degree=degree)
    Xp_train = poly.fit_transform(Xs_train)
    m = LinearRegression().fit(Xp_train, ys_train)
    y_line = m.predict(poly.transform(x_range))
    plt.plot(x_range, y_line, label=f"Degree {degree}", color=colors[degree], linewidth=2)
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.title("Bonus: Polynomial Regression Degree Comparison")
plt.legend()
plt.tight_layout()
plt.savefig(f"{PLOTS}/07_polynomial_comparison.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 6. BONUS: Feature combination experiments
# ---------------------------------------------------------------
feature_sets = {
    "Hours only": ["Hours_Studied"],
    "Hours + Attendance": ["Hours_Studied", "Attendance"],
    "Hours + Attendance + Previous_Scores": ["Hours_Studied", "Attendance", "Previous_Scores"],
    "Hours + Sleep": ["Hours_Studied", "Sleep_Hours"],
    "Hours + Attendance + Previous_Scores + Sleep + Tutoring": [
        "Hours_Studied", "Attendance", "Previous_Scores", "Sleep_Hours", "Tutoring_Sessions"
    ],
    "All numeric features": list(num_cols.drop("Exam_Score")),
}

combo_results = []
for name, feats in feature_sets.items():
    X_ = df[feats]
    Xtr, Xte, ytr, yte = train_test_split(X_, y, test_size=0.2, random_state=42)
    m = LinearRegression().fit(Xtr, ytr)
    pred = m.predict(Xte)
    combo_results.append({
        "Feature set": name,
        "n_features": len(feats),
        "R2": round(r2_score(yte, pred), 3),
        "MAE": round(mean_absolute_error(yte, pred), 2),
        "RMSE": round(mean_squared_error(yte, pred) ** 0.5, 2),
    })
combo_df = pd.DataFrame(combo_results)
print("\n=== Bonus: Feature combination experiments ===")
print(combo_df.to_string(index=False))

plt.figure(figsize=(9, 5))
plt.barh(combo_df["Feature set"], combo_df["R2"], color="#7c3aed")
plt.xlabel("R\u00b2 on test set")
plt.title("Bonus: R\u00b2 by Feature Combination")
plt.tight_layout()
plt.savefig(f"{PLOTS}/08_feature_combinations.png", dpi=150)
plt.close()

# save numeric summaries for the report
poly_results.to_csv("poly_results.csv", index=False)
combo_df.to_csv("combo_results.csv", index=False)
with open("metrics_summary.txt", "w") as f:
    f.write(f"Simple model (Hours_Studied only): R2={simple_r2:.3f}, MAE={simple_mae:.2f}, RMSE={simple_rmse:.2f}\n")
    f.write(f"Full model (all features): R2={full_r2:.3f}, MAE={full_mae:.2f}, RMSE={full_rmse:.2f}\n")

print("\nAll plots saved to", PLOTS)
