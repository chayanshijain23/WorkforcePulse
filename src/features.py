import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/processed_employee_attrition.csv")
OUTPUT_PATH = Path("data/final_employee_attrition.csv")

df = pd.read_csv(DATA_PATH)

# Binary encoding
df["OverTime_Flag"] = df["OverTime"].map({"Yes": 1, "No": 0})
df["Gender_Flag"] = df["Gender"].map({"Male": 1, "Female": 0})

# Experience-related features
df["PromotionGap"] = (
    df["YearsAtCompany"] - df["YearsSinceLastPromotion"]
)

df["RoleTenureRatio"] = (
    df["YearsInCurrentRole"] /
    (df["YearsAtCompany"] + 1)
)

df["ManagerTenureRatio"] = (
    df["YearsWithCurrManager"] /
    (df["YearsAtCompany"] + 1)
)

# Income relative to experience
df["IncomePerWorkingYear"] = (
    df["MonthlyIncome"] /
    (df["TotalWorkingYears"] + 1)
)

df.to_csv(OUTPUT_PATH, index=False)

print("Feature engineering completed!")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")
print(f"Saved to: {OUTPUT_PATH}")
print("\nNew features:")
print([
    "OverTime_Flag",
    "Gender_Flag",
    "PromotionGap",
    "RoleTenureRatio",
    "ManagerTenureRatio",
    "IncomePerWorkingYear"
])