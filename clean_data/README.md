# 📊 Clean 5-Year Harmonized PLFS Dataset (2021–2025)

## 📌 Dataset Overview
This folder contains the master longitudinal analysis dataset compiled from 5 consecutive years of official Government of India Periodic Labour Force Survey (PLFS) microdata.

* **Raw Survey Scope:** 2,833,966 Surveyed Citizens
* **Working Age (15–65):** 2,025,889
* **Active Regular Salaried Earners:** 301,065
* **Final Analysis Sample (1st–99th Percentile Noise Truncated):** **295,056 records**
* **File Formats Provided:**
  * `PLFS_2021_2025_Harmonized_Clean.csv` (~40.6 MB)
  * `PLFS_2021_2025_Harmonized_Clean.csv.gz` (~7.3 MB - Fast compressed format, readable directly by pandas)

---

## 🐍 How to Load in Python

```python
import pandas as pd

# Load the uncompressed CSV
df = pd.read_csv("clean_data/PLFS_2021_2025_Harmonized_Clean.csv")

# OR load the compressed version directly (faster):
# df = pd.read_csv("clean_data/PLFS_2021_2025_Harmonized_Clean.csv.gz")

print(f"Total Rows: {len(df):,}")
print("Yearly breakdown:")
print(df['Year'].value_counts().sort_index())
```

---

## 📋 Data Dictionary

| Column Name | Type | Description | Values / Encoding |
|:---|:---:|:---|:---|
| `Year` | Integer | Survey Year | `2021`, `2022`, `2023`, `2024`, `2025` |
| `State` | Integer | State / UT Code | `1` to `37` (`24` = Gujarat) |
| `State_Name` | String | Full Name of State / UT | `Gujarat`, `Maharashtra`, `Delhi`, etc. |
| `Sector` | Integer | Survey Sector Code | `1` = Rural, `2` = Urban |
| `Sector_Desc` | String | Sector Name | `Rural`, `Urban` |
| `Gender` | Integer | Gender Code | `1` = Male, `2` = Female, `3` = Transgender |
| `Gender_Desc` | String | Gender Label | `Male`, `Female`, `Transgender` |
| `Age` | Integer | Age in completed years | Working age: `15` to `65` |
| `Education` | Integer | MoSPI General Education Code | `1` to `13` (Note: Code `9` does not exist in MoSPI codebook) |
| `Education_Desc` | String | Standard Educational Attainment | `Not Literate`, `Primary`, `Middle`, `Secondary`, `Higher Secondary`, `Diploma/Certificate`, `Graduate`, `Post Graduate & Above` |
| `Weekly_Wage` | Float | Weekly Regular Salaried Earnings | Indian Rupees (₹) |
| `Weekly_Hours` | Float | Actual Total Hours Worked in 7-day week | Bounded `10` to `84` hours/week |
| `Hourly_Wage` | Float | Normalized Hourly Wage Rate | $\text{Weekly\_Wage} / \text{Weekly\_Hours}$ (₹/hr) |
| `Log_Hourly_Wage` | Float | Natural Log of Hourly Wage | $\ln(\text{Hourly\_Wage})$ |
| `CPI` | Float | RBI Consumer Price Index Deflator | `2021` = 100.0, `2022` = 106.7, `2023` = 112.5, `2024` = 118.6, `2025` = 124.8 |
| `Real_Hourly_Wage` | Float | Inflation-Adjusted Hourly Wage | $(\text{Hourly\_Wage} / \text{CPI}) \times 100$ |
| `Log_Real_Hourly_Wage` | Float | Natural Log of Real Hourly Wage | $\ln(\text{Real\_Hourly\_Wage})$ |
| `Schooling_Years` | Integer | Estimated Formal Years of Schooling | `0` (Not Literate) to `17` (Post Graduate) |
| `Experience` | Integer | Mincer Potential Work Experience | $\max(0, \text{Age} - \text{Schooling\_Years} - 6)$ |
| `Experience_Sq` | Integer | Squared Potential Work Experience | $(\text{Experience})^2$ (Models wage curve concavity) |
