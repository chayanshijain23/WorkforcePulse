import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/processed_employee_attrition.csv")
REPORT_PATH = Path("reports/eda_analysis.txt")

df = pd.read_csv(DATA_PATH)

with open(REPORT_PATH, "w", encoding="utf-8") as f:

    f.write("WORKFORCEPULSE - EDA ANALYSIS\n")
    f.write("=" * 50 + "\n\n")

    # Attrition
    f.write("1. ATTRITION DISTRIBUTION\n")
    f.write("-" * 30 + "\n")
    f.write(str(df["Attrition"].value_counts()) + "\n\n")

    # Overtime
    f.write("2. OVERTIME VS ATTRITION\n")
    f.write("-" * 30 + "\n")
    overtime = pd.crosstab(
        df["OverTime"],
        df["Attrition"],
        normalize="index"
    ) * 100
    f.write(str(overtime.round(2)) + "\n\n")

    # Department
    f.write("3. DEPARTMENT VS ATTRITION\n")
    f.write("-" * 30 + "\n")
    department = pd.crosstab(
        df["Department"],
        df["Attrition"],
        normalize="index"
    ) * 100
    f.write(str(department.round(2)) + "\n\n")

    # Job Role
    f.write("4. JOB ROLE VS ATTRITION\n")
    f.write("-" * 30 + "\n")
    job_role = pd.crosstab(
        df["JobRole"],
        df["Attrition"],
        normalize="index"
    ) * 100
    f.write(str(job_role.round(2)) + "\n\n")

    # Job Satisfaction
    f.write("5. JOB SATISFACTION VS ATTRITION\n")
    f.write("-" * 30 + "\n")
    satisfaction = pd.crosstab(
        df["JobSatisfaction"],
        df["Attrition"],
        normalize="index"
    ) * 100
    f.write(str(satisfaction.round(2)) + "\n\n")

    # Numeric comparison
    f.write("6. NUMERICAL FEATURES BY ATTRITION\n")
    f.write("-" * 30 + "\n")

    numeric_cols = [
        "Age",
        "MonthlyIncome",
        "DistanceFromHome",
        "TotalWorkingYears",
        "YearsAtCompany",
        "YearsInCurrentRole",
        "YearsSinceLastPromotion",
        "YearsWithCurrManager"
    ]

    comparison = df.groupby("Attrition")[numeric_cols].mean().round(2)
    f.write(str(comparison))

print("EDA analysis report generated successfully!")
print(f"Saved to: {REPORT_PATH}")