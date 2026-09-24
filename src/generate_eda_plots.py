import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# Paths
DATA_PATH = Path("data/processed_employee_attrition.csv")
OUTPUT_DIR = Path("outputs/eda")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Load data
df = pd.read_csv(DATA_PATH)


# Style
sns.set_theme(style="whitegrid")


# =========================
# 1. ATTRITION DISTRIBUTION
# =========================

plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Attrition")
plt.title("Employee Attrition Distribution")
plt.xlabel("Attrition (0 = No, 1 = Yes)")
plt.ylabel("Number of Employees")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_attrition_distribution.png")
plt.close()


# =========================
# 2. AGE DISTRIBUTION
# =========================

plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="Age", bins=20, kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Employees")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_age_distribution.png")
plt.close()


# =========================
# 3. MONTHLY INCOME
# =========================

plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="MonthlyIncome", bins=30, kde=True)
plt.title("Monthly Income Distribution")
plt.xlabel("Monthly Income")
plt.ylabel("Number of Employees")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_monthly_income.png")
plt.close()


# =========================
# 4. YEARS AT COMPANY
# =========================

plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="YearsAtCompany", bins=20, kde=True)
plt.title("Years at Company Distribution")
plt.xlabel("Years at Company")
plt.ylabel("Number of Employees")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "04_years_at_company.png")
plt.close()


# =========================
# 5. OVERTIME VS ATTRITION
# =========================

plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="OverTime", hue="Attrition")
plt.title("Overtime vs Attrition")
plt.xlabel("Overtime")
plt.ylabel("Number of Employees")
plt.legend(title="Attrition", labels=["No", "Yes"])
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "05_overtime_vs_attrition.png")
plt.close()


# =========================
# 6. JOB ROLE VS ATTRITION
# =========================

plt.figure(figsize=(11, 6))
sns.countplot(
    data=df,
    y="JobRole",
    hue="Attrition"
)
plt.title("Job Role vs Attrition")
plt.xlabel("Number of Employees")
plt.ylabel("Job Role")
plt.legend(title="Attrition", labels=["No", "Yes"])
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "06_jobrole_vs_attrition.png")
plt.close()


# =========================
# 7. JOB SATISFACTION VS ATTRITION
# =========================

plt.figure(figsize=(8, 5))
sns.countplot(
    data=df,
    x="JobSatisfaction",
    hue="Attrition"
)
plt.title("Job Satisfaction vs Attrition")
plt.xlabel("Job Satisfaction (1 = Low, 4 = Very High)")
plt.ylabel("Number of Employees")
plt.legend(title="Attrition", labels=["No", "Yes"])
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "07_job_satisfaction_vs_attrition.png")
plt.close()


# =========================
# 8. DEPARTMENT VS ATTRITION
# =========================

plt.figure(figsize=(8, 5))
sns.countplot(
    data=df,
    x="Department",
    hue="Attrition"
)
plt.title("Department vs Attrition")
plt.xlabel("Department")
plt.ylabel("Number of Employees")
plt.legend(title="Attrition", labels=["No", "Yes"])
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "08_department_vs_attrition.png")
plt.close()


print("EDA plots generated successfully!")

for file in sorted(OUTPUT_DIR.glob("*.png")):
    print(file)