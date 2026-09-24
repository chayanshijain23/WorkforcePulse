import pandas as pd
from pathlib import Path


# =========================
# 1. PROJECT PATHS
# =========================

RAW_PATH = Path("data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv")
PROCESSED_PATH = Path("data/processed_employee_attrition.csv")
REPORT_PATH = Path("reports/data_quality_report.txt")


# =========================
# 2. CREATE OUTPUT FOLDERS
# =========================

PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)


# =========================
# 3. LOAD RAW DATA
# =========================

df_raw = pd.read_csv(RAW_PATH)

print("Raw dataset loaded successfully!")
print(f"Rows: {df_raw.shape[0]}")
print(f"Columns: {df_raw.shape[1]}")


# Create a working copy
df = df_raw.copy()


# =========================
# 4. DATA QUALITY CHECKS
# =========================

missing_values = df.isnull().sum().sum()
duplicate_rows = df.duplicated().sum()


# =========================
# 5. IDENTIFY CONSTANT COLUMNS
# =========================

constant_columns = [
    column for column in df.columns
    if df[column].nunique() <= 1
]


# =========================
# 6. REMOVE IRRELEVANT COLUMNS
# =========================

# EmployeeNumber is an ID, not a useful predictive feature.
# EmployeeCount, Over18 and StandardHours are constant columns.

columns_to_remove = [
    "EmployeeNumber",
    "EmployeeCount",
    "Over18",
    "StandardHours"
]

df = df.drop(columns=columns_to_remove)


# =========================
# 7. CLEAN CATEGORICAL VALUES
# =========================

categorical_columns = df.select_dtypes(include="object").columns

for column in categorical_columns:
    df[column] = df[column].astype(str).str.strip()


# =========================
# 8. ENCODE TARGET VARIABLE
# =========================

df["Attrition"] = df["Attrition"].map({
    "Yes": 1,
    "No": 0
})


# =========================
# 9. TARGET DISTRIBUTION
# =========================

target_counts = df["Attrition"].value_counts().sort_index()

no_count = int(target_counts.get(0, 0))
yes_count = int(target_counts.get(1, 0))

total = len(df)

no_percentage = (no_count / total) * 100
yes_percentage = (yes_count / total) * 100


# =========================
# 10. SAVE CLEANED DATA
# =========================

df.to_csv(PROCESSED_PATH, index=False)


# =========================
# 11. CREATE DATA QUALITY REPORT
# =========================

report = f"""
WORKFORCEPULSE - DATA QUALITY REPORT
====================================

Rows before cleaning: {df_raw.shape[0]}
Columns before cleaning: {df_raw.shape[1]}

Rows after cleaning: {df.shape[0]}
Columns after cleaning: {df.shape[1]}

Missing values: {missing_values}
Duplicate rows: {duplicate_rows}

Constant columns detected:
{constant_columns}

Columns removed:
{columns_to_remove}

Target column:
Attrition

Target encoding:
Yes -> 1
No  -> 0

Attrition = No:
{no_count} ({no_percentage:.2f}%)

Attrition = Yes:
{yes_count} ({yes_percentage:.2f}%)

Raw dataset was NOT modified.
"""

REPORT_PATH.write_text(report, encoding="utf-8")


# =========================
# 12. FINAL OUTPUT
# =========================

print("\nCleaning completed successfully!")
print(f"Processed dataset saved to: {PROCESSED_PATH}")
print(f"Quality report saved to: {REPORT_PATH}")

print("\nTarget distribution:")
print(f"No  : {no_count} ({no_percentage:.2f}%)")
print(f"Yes : {yes_count} ({yes_percentage:.2f}%)")