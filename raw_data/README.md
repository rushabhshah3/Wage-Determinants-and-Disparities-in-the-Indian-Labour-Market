# 📦 Raw 5-Year Uncleaned PLFS Survey Microdata (2021–2025)

## 📌 Dataset Overview
This folder contains the **uncleaned, raw longitudinal microdata** compiled directly from the official Government of India (MoSPI) Periodic Labour Force Survey annual releases across all 5 years (2021–2025).

* **Total Raw Citizens Surveyed:** **2,833,966 records** across all 36 States & UTs
* **File Format:** `PLFS_2021_2025_Raw_Messy.csv.gz` (8.95 MB compressed, ~100 MB uncompressed)
* **Status:** **100% Raw & Uncleaned** (Zero imputation, zero outlier trimming, zero demographic filtering)

---

## 🔍 Why This Raw Dataset is Essential for Academic EDA
In accordance with **Dr. Gopinath Panda's EDA curriculum**, students must demonstrate real data cleaning and preprocessing rather than starting from an artificially pre-cleaned file:

1. **Missing Data Analysis:**
   * Contains **1,745,018 real missing values (`NaN`)** in weekly working hours (`Weekly_Hours`).
   * Used in Chapter 2 to demonstrate missingness diagnostics (`msno.matrix`, `msno.bar`, `msno.heatmap`), test MCAR vs. MAR mechanisms, and evaluate **MICE vs. Mean/Median imputation**.
2. **Population Attrition Funnel:**
   * Contains **654,740 children ($<15$ years)** and **153,337 elderly retirees ($>65$ years)**.
   * Contains **2,530,934 citizens with ₹0 weekly wage (89.3%)** (representing students, homemakers, unpaid domestic care, and non-salaried family labor).
   * Used in Chapter 2 to document the **Population Attrition Waterfall**.
3. **Outlier Detection Hierarchy:**
   * Raw weekly wage skewness is **3.41** with maximum earnings reaching ₹520,000/week.
   * Used to apply **Tukey's IQR rule**, **Z-score ($|Z| > 3.0$)**, and **Mahalanobis Distance ($\chi^2_{0.975}$)**.
4. **Data Dictionary (Raw Integer Codes):**
   * Stored in raw integer format (`State`: 1–37, `Sector`: 1–2, `Gender`: 1–2, `Education`: 1–13) to demonstrate categorical label mapping in Chapter 2.

---

## 🐍 How to Load in Python

```python
import pandas as pd

# Load the raw uncleaned microdata directly (pandas handles .csv.gz automatically):
df_raw = pd.read_csv("raw_data/PLFS_2021_2025_Raw_Messy.csv.gz")

print(f"Total Raw Records Loaded: {len(df_raw):,}")
print("Missing values per column:\n", df_raw.isnull().sum())
print("\nYearly surveyed citizens distribution:\n", df_raw['Year'].value_counts().sort_index())
```
