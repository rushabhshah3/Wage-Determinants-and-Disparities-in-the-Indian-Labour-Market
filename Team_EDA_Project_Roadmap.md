# 📊 Team EDA Project Roadmap: Indian Labor Market & Wage Disparities

## 1. Our Selected Project & Goal
*   **Project Title:** Exploratory Data Analysis of Wage Determinants in India.
*   **Our Core Goal:** To move beyond basic assumptions and visually prove the exact economic impact of education, gender, and the informal sector on Indian wages using advanced statistical data storytelling.

## 2. The Dataset Details
*   **Official Source:** Periodic Labour Force Survey (PLFS) via MoSPI.
*   **Database Link:** `https://microdata.gov.in/NADA/index.php/catalog/292/get-microdata`
*   **File Used:** Person-Level File (`MPER_Q2_Q4_2025.csv`)
*   **Database Size:** Over **419,000 surveyed individuals** and 130+ features.
*   **Key Columns:** Age, Gender, General Education, State, Industry Code (NIC), and Weekly Earnings.

---

## 3. Our Execution Plan (How to do it)

### Step A: Data Cleaning & Decoding (Data Engineering)
The raw government data uses numeric codes (e.g., Gender is listed as `1` or `2`).
1.  **Decoding:** We will use Python (`pandas`) and the official Data Dictionary to map these numbers into readable text (e.g., `1 -> Male`, `11 -> Graduate`).
2.  **Filtering:** We will clean the dataset by removing non-working individuals (filtering out rows where earnings = 0).
3.  **Missing Values:** We will impute or drop rows with missing crucial demographic data.

### Step B: Exploratory Data Analysis (EDA)
We will build high-level economic charts to answer real-world questions:
1.  **Mincer Earnings Curves (Line Plots):** Plotting Age vs. Weekly Earnings to visually identify the exact age where Indian salaries peak.
2.  **Gender Wage Gap Analysis (Box Plots):** Comparing male vs. female salaries, specifically filtering by education level to see if obtaining a degree actually closes the gender gap.
3.  **The Informal Penalty (Violin Plots):** Showing the wage density difference between workers in registered companies versus unregistered daily wage workers.
4.  **Geospatial Analysis:** Mapping average earnings by State Code to highlight regional wealth disparities.

### Step C: Presentation & Final Submission
*   We will compile all our Python visualizations into a clean, professional PDF report/dashboard.
*   In the final examination, our presentation will focus on the **Insights** we extracted, proving to the panel that we successfully executed complex Data Preprocessing, Data Transformation, and Statistical EDA on a massive real-world dataset.
