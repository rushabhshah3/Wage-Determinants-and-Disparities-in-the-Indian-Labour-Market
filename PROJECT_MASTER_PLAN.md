# 📊 Complete Project Plan: Indian Labor Market Wage Disparities
### M.Tech Machine Learning | EDA Project

---

## 🎓 LEARNING APPROACH (How We Work Together)

> For EVERY step, the explanation will follow this format:
> - **📖 WHAT** — What is this step?
> - **❓ WHY** — Why are we doing this? What happens if we skip it?
> - **💻 HOW** — The actual Python code
> - **💡 INSIGHT** — What does the result tell us?

This project is **50% of your coursework**. Every chart must have a title, axis labels, and a written insight explaining what it proves.

---

## 🔴 THE PROBLEM STATEMENT

> **"In India, everyone knows that women earn less, educated people earn more, and rural workers earn less than urban ones — but NOBODY can tell you the EXACT number. By HOW MUCH? We will prove it with data."**

### What problems we are solving:

| # | Problem Question | What We Prove |
|---|-----------------|---------------|
| 1 | Does gender affect wages? | Exact ₹ penalty for being female, same job/education |
| 2 | Does education pay off? | Exact ₹ return on Graduate vs Primary education |
| 3 | Is informal sector worse? | Exact ₹ loss for working in unregistered companies |
| 4 | When do wages peak? | Exact age at which Indian salaries hit maximum |
| 5 | Are states unequal? | Which states pay most/least and by how much |

---

## 🗂️ OUR DATA

- **File:** `Data_in_CSV/MPER_Q2_Q4_2025.csv`
- **Size:** ~1,696,741 individual records
- **Source:** Government of India — PLFS (Ministry of Statistics)
- **Key columns we'll use:**

| Column | Meaning |
|--------|---------|
| `sex` | 1 = Male, 2 = Female |
| `age` | Age of person |
| `gedu_lvl` | Education level (codes 01–13) |
| `st` | State code |
| `ern_reg` | Weekly earnings (our TARGET variable) |
| `nsc` | Formal/Informal sector |
| `ocu_cws` | Occupation code |
| `marst` | Marital status |

---

## 🛣️ FULL STEP-BY-STEP PLAN

---

### ✅ STEP 1 — Data Loading & First Look
**What we do:**
- Load the CSV into Python using `pandas`
- Check shape (rows × columns)
- Print first 5 rows
- Check data types of each column

**What problem it solves:**
> We confirm the data is loaded correctly and understand its structure before touching anything.

**Output:** Basic info printout, `.head()`, `.info()`, `.describe()`

---

### ✅ STEP 2 — Data Cleaning
**What we do:**
- Remove rows where `ern_reg = 0` (non-working / unemployed people)
- Drop rows with missing values in key columns (`age`, `sex`, `gedu_lvl`, `ern_reg`)
- Remove extreme outliers (people earning ₹0 or unrealistically high amounts)
- Filter only working-age population (age 15–65)

**What problem it solves:**
> Raw government data has codes and noise. If we include people who earned ₹0 (unemployed), our wage analysis will be completely wrong.

**Output:** Clean dataset with only valid working individuals

---

### ✅ STEP 3 — Data Decoding (Making it Human-Readable)
**What we do:**
- Map numeric codes to readable labels:
  - `sex`: `1 → Male`, `2 → Female`
  - `gedu_lvl`: `01 → Not Literate`, `08 → Graduate`, `10 → Post Graduate`, etc.
  - `nsc`: Formal vs Informal sector labels
  - `st`: State codes → State names

**What problem it solves:**
> The raw data says `sex = 1`. Nobody understands that in a chart. We need it to say "Male" and "Female" for clear visualizations.

**Output:** Decoded, labeled DataFrame ready for plotting

---

### ✅ STEP 4 — EDA: Wage Distribution Analysis
**What we do:**
- Plot histogram of `ern_reg` (weekly earnings)
- Show it is RIGHT-SKEWED (most people earn low, few earn very high)
- Apply **Log Transformation** → `log(ern_reg)`
- Show how the distribution becomes normal after log transform

**What problem it solves:**
> This proves we understand statistics. Wage data is always skewed. Log transformation is a standard technique to normalize it for better ML model performance.

**Charts:** Histogram (before) + Histogram (after log transform)

---

### ✅ STEP 5 — EDA: Gender Wage Gap Analysis
**What we do:**
- Box plot: Male vs Female weekly earnings
- Filter by education level → compare male/female at SAME education
- Heatmap: Education Level × Gender → color = average wage

**Questions answered:**
- How much less do women earn than men overall?
- Does getting a degree reduce the gender pay gap or widen it?

**Charts:** Box plots, Heatmap

---

### ✅ STEP 6 — EDA: Education Returns Analysis
**What we do:**
- Bar chart: Average wage by education level (Not Literate → Post Graduate)
- Separate lines for Rural vs Urban
- Calculate exact ₹ difference between each education step

**Questions answered:**
- How much more does a Graduate earn vs a Primary-educated worker?
- Is education more valuable in Rural or Urban areas?

**Charts:** Bar chart, Line chart

---

### ✅ STEP 7 — EDA: Mincer Earnings Curve (Age vs Wage)
**What we do:**
- Line plot: Age (X-axis) vs Average Weekly Earnings (Y-axis)
- Separate lines for Male and Female
- Identify the EXACT peak age

**Questions answered:**
- At what age do Indian wages peak?
- Does the peak age differ for men and women?

**Charts:** Line plot showing the classic "Inverted-U" shape

---

### ✅ STEP 8 — EDA: Formal vs Informal Sector Analysis
**What we do:**
- Violin plot: Wage distribution for Formal vs Informal workers
- Control for education → compare same-educated workers across sectors
- Calculate exact ₹ penalty for being in informal sector

**Questions answered:**
- How much does a worker lose just because their employer is unregistered?
- Is the informal penalty higher for educated or uneducated workers?

**Charts:** Violin plot, Bar chart

---

### ✅ STEP 9 — EDA: Geospatial / State-wise Analysis
**What we do:**
- Bar chart of average wages by state (top 10 and bottom 10)
- Map visualization of India colored by average wage per state

**Questions answered:**
- Which states have the highest and lowest wages?
- How large is the regional wage inequality in India?

**Charts:** Horizontal bar chart, India heatmap

---

### ✅ STEP 10 — Correlation & Feature Importance
**What we do:**
- Correlation heatmap of all numeric variables
- Identify which features are most strongly linked to wages

**Questions answered:**
- Which single factor (age, education, gender) has the STRONGEST effect on wages?

**Charts:** Correlation heatmap

---

### ✅ STEP 11 — Machine Learning: Wage Prediction Model
**What we do:**
1. Encode categorical variables
2. Split data → Train (80%) / Test (20%)
3. Train **Linear Regression** (baseline)
4. Train **Random Forest Regressor**
5. Train **XGBoost Regressor**
6. Compare models using R² score and RMSE

**What problem it solves:**
> EDA shows patterns visually. ML QUANTIFIES them mathematically with exact coefficients.

---

### ✅ STEP 12 — SHAP Explainability (Black Box Opening)
**What we do:**
- Apply SHAP values on the best ML model
- SHAP bar chart: which feature impacts wages the most?
- SHAP waterfall chart: explain ONE individual's wage prediction

**Example SHAP output:**
> *"Base wage = ₹3,500. Being Female = -₹800. Being Graduate = +₹2,200. Informal sector = -₹600. Final Prediction = ₹4,300"*

**This is the highlight of the project — Explainable AI!**

---

### ✅ STEP 13 — Final Report & Dashboard
**What we do:**
- Compile all charts into a clean PDF report
- Write conclusions for each research question
- Create a summary "Wage Penalty Dashboard"

---

## 🎯 FINAL DELIVERABLES

| Deliverable | Description |
|-------------|-------------|
| 📓 Jupyter Notebook | All code, charts, and analysis |
| 📊 PDF Report | Clean presentation of findings |
| 🤖 ML Model | Trained wage prediction model |
| 💡 SHAP Dashboard | Explainable AI output |
| 🗺️ State Wage Map | India geospatial visualization |

---

## 🧑‍🤝‍🧑 WHO DOES WHAT (Team Split)

| Task | Steps |
|------|-------|
| **Data Engineering** | Steps 1, 2, 3 |
| **EDA & Visualization** | Steps 4, 5, 6, 7, 8, 9, 10 |
| **Machine Learning** | Steps 11, 12 |
| **Report & Presentation** | Step 13 |

---

## 📌 PROGRESS TRACKER

| Step | Status | Notes |
|------|--------|-------|
| Step 1 — Data Loading | ✅ Completed | 1.69M records × 108 columns loaded |
| Step 2 — Data Cleaning | ✅ Completed | Age (15-65), Earnings > 0, IQR Outliers removed (414,272 earners left) |
| Step 3 — Data Decoding | ✅ Completed | NSSO codes mapped to Gender, Education, State, Marital Status (0 nulls) |
| Step 4 — Wage Distribution EDA | ✅ Completed | Raw vs. Log wage distribution, skewness analysis (+0.885 to -0.874) |
| Step 5 — Gender Wage Gap EDA | ✅ Completed | 57.3% female median penalty (₹8,600/week gap) |
| Step 6 — Education Returns EDA | ✅ Completed | Education × Gender matrix (gap compresses only at PG & Above) |
| Step 7 — Mincer Earnings Curve | ✅ Completed | Male peak age 39 (₹17,500), Female peak age 23 (₹8,000) |
| Step 8 — Formal vs Informal EDA | ✅ Completed | Casual labour (99.6%) vs regular salaried disparity |
| Step 9 — State-wise EDA | ✅ Completed | 2.78x inter-state disparity (Delhi/Mizoram vs. Nagaland/Odisha) |
| Step 10 — Correlation Analysis | ✅ Completed | Pearson matrix; resolved non-linear age correlation paradox |
| Step 11 — ML Model Training | ✅ Completed | Trained OLS, RF, and XGBoost on 331,417 rows (XGBoost R²=0.4000) |
| Step 12 — SHAP Explainability | ✅ Completed | Global Top 12 bar chart, Beeswarm plot, and Worker #0 Waterfall case study |
| Step 13 — Synthesis & Report | ✅ Completed | Section 9 empirical synthesis answering all research hypotheses |
| Step 14 — Interactive Dashboards | ✅ Completed | In-Notebook Widget Dashboard (Cell 78) + Standalone Web App (`app.py`) |

---

> **Core Message to Professor:**
> *"We are building an Explainable AI pipeline on India's official government data that mathematically proves the exact cost of gender discrimination, education gaps, and informal sector exploitation in the Indian labor market."*
