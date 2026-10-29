# M.Tech Machine Learning Project Proposal
## Topic: Quantifying Wage Determinants and Disparities in the Indian Labor Market

---

### 1. Introduction: What is the Problem?

In India, we constantly hear debates about the "Gender Pay Gap," the "Rural vs. Urban Divide," and whether "Education actually leads to higher income." However, most discussions rely on basic averages or guesswork. 

**The Real-World Problem:** 
Government policymakers, NGOs, and labor organizations do not have exact mathematical proof of *how much* a specific trait is worth. For example, if a woman and a man have the exact same education, age, and job, how much is the woman penalized simply for being a woman? If a worker moves from the informal (unregistered) sector to the formal sector, exactly how much does their wage increase?

Currently, there is a lack of deep, algorithmic analysis to isolate these variables. Our project solves this by treating the labor market as a mathematical function that can be solved using Machine Learning.

---

### 2. The Dataset (Why it is perfect)

*   **Source:** Periodic Labour Force Survey (PLFS) microdata from the Ministry of Statistics and Programme Implementation (MoSPI).
*   **Scale:** Over 400,000 surveyed individuals.
*   **Features:** Age, Gender, Education Level, Marital Status, State, Rural/Urban Sector, Industry Code (NIC), Occupation Code (NCO), and Enterprise Type (Formal/Informal).
*   **Target Variable:** Weekly Earnings (Regression).

---

### 3. What are the specific questions we will solve?

If we present this to our professors, we can guarantee that our project will answer these exact, high-level economic questions using data:

1.  **The Education Question:** What is the exact financial return on a Graduate degree compared to a Primary education in rural vs. urban areas?
2.  **The Gender Gap Question:** Controlling for all other factors (same age, same job, same education), what is the exact financial penalty for being female in the Indian workforce?
3.  **The Informal Sector Penalty:** How much income does a worker lose simply because their employer is unregistered (informal), even if they have the same skills as a formal worker?
4.  **The Experience Curve:** At what exact age do wages peak in India before they start declining?

---

### 4. How will we solve this using EDA?

Our Exploratory Data Analysis (EDA) will be the strongest part of the project. We will not just plot random bar charts. We will build specific statistical visualizations to prove our points before we even run the ML models:

*   **Mincer Earnings Profiles (Line Charts):** We will plot Age on the X-axis and Wage on the Y-axis. This will visually prove the "Inverted-U" shape of income (wages rise with experience, peak at age 45-50, and decline).
*   **Density Plots & Log Transformations:** We will plot the distribution of wages. Because wealth is unequal, the chart will be highly right-skewed. We will demonstrate our statistical knowledge by applying a Log-Transformation to normalize the curve.
*   **Interaction Heatmaps:** We will create a matrix showing Education Level vs. Gender, with the color representing the average wage. This will visually answer the question: *Does higher education bridge the gender wage gap, or widen it?*

---

### 5. How will we solve this using Machine Learning?

This is a **Regression Task**.
1.  **Baseline:** We will first train a Multiple Linear Regression model to get basic coefficients.
2.  **Advanced Algorithms:** Because human wages are not perfectly linear (e.g., age helps you until you are 50, then it hurts you), we will use non-linear, tree-based algorithms like **XGBoost** and **Random Forest Regressors**.
3.  **Opening the Black Box:** To impress the panel, we will use **SHAP (SHapley Additive exPlanations)**. This AI technique explains the model's math. For any prediction, SHAP will output a chart saying: *"The base wage is ₹4000. Being a Graduate added +₹2000. Being in the Informal sector subtracted -₹1000. Final prediction: ₹5000."*

---

### 6. What is the Final Output? (Why it won't be rejected)

If a team just says "We are predicting wages," a professor might reject it as "too simple." 

Instead, we will say: 
> *"We are building an Explainable AI (XAI) pipeline that takes the Indian Government's official labor data and outputs a dynamic 'Wage Penalty Dashboard'. It will allow policymakers to input any demographic profile and immediately see the exact financial discrimination or advantage that profile faces in the current economy."*

This guarantees a unique, highly impactful M.Tech thesis that blends Data Engineering, Statistical EDA, and Advanced Machine Learning.
