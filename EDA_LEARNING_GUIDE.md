# 📚 EDA Master Learning Guide
### What is EDA? Why do we do it? How do we do it?
> This guide teaches you EDA from scratch using your own PLFS project as the example.

---

## 🧠 WHAT IS EDA?

**EDA = Exploratory Data Analysis**

Before you build any Machine Learning model, you MUST understand your data.
EDA is the process of **visually and statistically exploring your dataset** to:
- Understand its structure
- Find patterns and relationships
- Detect problems (missing values, outliers)
- Form hypotheses to test

> Think of it like this: Before a doctor performs surgery, they first do an X-ray, blood test, and checkup. EDA is that checkup for your data.

---

## 🧱 THE 5 PILLARS OF EDA

Every proper EDA must cover all 5 pillars:

```
1. UNDERSTAND  →  What does the data look like?
2. CLEAN       →  Fix problems in the data
3. UNIVARIATE  →  Study ONE variable at a time
4. BIVARIATE   →  Study TWO variables together
5. MULTIVARIATE → Study MANY variables together
```

---

## 📐 PILLAR 1: UNDERSTAND THE DATA

### What we do:
```python
df.shape          # How many rows and columns?
df.head()         # First 5 rows
df.tail()         # Last 5 rows
df.info()         # Column names, data types, non-null counts
df.describe()     # Mean, min, max, std for numeric columns
df.columns        # List of all column names
df.dtypes         # Data type of each column
df.nunique()      # How many unique values per column
```

### Why we do it:
> You CANNOT analyze data you don't understand. This step tells you:
> - How BIG is the dataset?
> - Are columns numbers or text?
> - How many unique values does each column have?

### In our project:
> We need to know: How many people are in the survey? What columns exist? Are earnings stored as integers or floats?

---

## 🧹 PILLAR 2: DATA CLEANING

### Types of problems we fix:

#### A) Missing Values
```python
df.isnull().sum()           # Count missing values per column
df.isnull().mean() * 100    # % missing per column

# Fix options:
df.dropna()                 # Drop rows with ANY missing value
df.dropna(subset=['ern_reg']) # Drop only where earnings is missing
df['age'].fillna(df['age'].mean())  # Fill with mean
```

#### B) Duplicates
```python
df.duplicated().sum()       # Count duplicate rows
df.drop_duplicates()        # Remove duplicates
```

#### C) Wrong Data Types
```python
df['ern_reg'] = df['ern_reg'].astype(float)  # Convert to number
```

#### D) Outliers
```python
# IQR Method to detect outliers
Q1 = df['ern_reg'].quantile(0.25)
Q3 = df['ern_reg'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
# Keep only values within range
df_clean = df[(df['ern_reg'] >= lower) & (df['ern_reg'] <= upper)]
```

### Why we do it:
> GARBAGE IN = GARBAGE OUT. If your data has errors, your analysis will have errors. Your ML model will be wrong.

### In our project:
> We remove people with 0 earnings (unemployed) because we only want to study WORKING people's wages.

---

## 📊 PILLAR 3: UNIVARIATE ANALYSIS
### (Studying ONE variable at a time)

This answers: **"What does THIS column look like?"**

### For NUMERIC columns (like `age`, `ern_reg`):

| Chart | What it shows | When to use |
|-------|--------------|-------------|
| **Histogram** | Distribution of values | Always — first chart to make |
| **Box Plot** | Median, spread, outliers | To see outliers quickly |
| **KDE Plot** | Smooth distribution curve | Better looking than histogram |

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Histogram
plt.hist(df['ern_reg'], bins=50)
plt.title('Distribution of Weekly Earnings')
plt.xlabel('Weekly Earnings (₹)')
plt.ylabel('Number of People')
plt.show()

# Box Plot
sns.boxplot(x=df['ern_reg'])

# KDE Plot
sns.kdeplot(df['ern_reg'])
```

### For CATEGORICAL columns (like `sex`, `gedu_lvl`):

| Chart | What it shows | When to use |
|-------|--------------|-------------|
| **Bar Chart** | Count of each category | Always for categorical |
| **Pie Chart** | % share of each category | When showing proportions |
| **Value Counts** | Frequency table | Quick check |

```python
df['sex'].value_counts()                    # Frequency table
df['sex'].value_counts().plot(kind='bar')   # Bar chart
```

### Key concept — SKEWNESS:
```
Normal (Symmetric):     🔔  Mean = Median
Right Skewed:           📈  Few people earn VERY HIGH (wages always look like this!)
Left Skewed:            📉  Few people earn VERY LOW
```

> Wage data is ALWAYS right skewed. We fix this with LOG TRANSFORMATION.

```python
import numpy as np
df['log_earnings'] = np.log1p(df['ern_reg'])  # log(1 + x) to handle zeros
```

---

## 📊 PILLAR 4: BIVARIATE ANALYSIS
### (Studying TWO variables together)

This answers: **"How does X relate to Y?"**

### Numeric vs Numeric (e.g., Age vs Earnings):

| Chart | What it shows |
|-------|--------------|
| **Scatter Plot** | Individual dots — see pattern |
| **Line Plot** | Trend over a range |
| **Correlation** | Single number showing relationship strength |

```python
# Scatter plot
plt.scatter(df['age'], df['ern_reg'], alpha=0.3)

# Line plot (average earnings per age)
df.groupby('age')['ern_reg'].mean().plot()

# Correlation coefficient
df['age'].corr(df['ern_reg'])
# Output: 0.35 means moderate positive relationship
```

### Correlation values explained:
```
+1.0  →  Perfect positive (as X goes up, Y goes up)
+0.5  →  Moderate positive
 0.0  →  No relationship
-0.5  →  Moderate negative
-1.0  →  Perfect negative (as X goes up, Y goes down)
```

### Categorical vs Numeric (e.g., Gender vs Earnings):

| Chart | What it shows |
|-------|--------------|
| **Box Plot** | Median and spread per group |
| **Violin Plot** | Full distribution per group |
| **Bar Chart** | Average value per group |

```python
# Box plot: Male vs Female earnings
sns.boxplot(x='sex', y='ern_reg', data=df)

# Violin plot
sns.violinplot(x='sex', y='ern_reg', data=df)

# Average earnings by gender
df.groupby('sex')['ern_reg'].mean()
```

### Categorical vs Categorical (e.g., Gender vs Education):

| Chart | What it shows |
|-------|--------------|
| **Heatmap** | Color shows value at intersection |
| **Cross-tab** | Table of counts |
| **Grouped Bar** | Side-by-side bars |

```python
# Pivot table → Heatmap
pivot = df.pivot_table(values='ern_reg', index='gedu_lvl', columns='sex', aggfunc='mean')
sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlOrRd')
```

---

## 📊 PILLAR 5: MULTIVARIATE ANALYSIS
### (Studying MANY variables together)

This answers: **"How do multiple things interact?"**

| Chart | What it shows |
|-------|--------------|
| **Correlation Heatmap** | All numeric variable relationships |
| **Pair Plot** | All vs All scatter plots |
| **FacetGrid** | Same chart split by category |
| **Grouped Line Plot** | Multiple lines on same chart |

```python
# Correlation heatmap of all numeric columns
corr = df[['age', 'ern_reg', 'gedu_lvl']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')

# Pair plot
sns.pairplot(df[['age', 'ern_reg', 'gedu_lvl']])

# FacetGrid: Age vs Earnings, split by Gender
g = sns.FacetGrid(df, col='sex')
g.map(plt.plot, 'age', 'ern_reg')
```

---

## 🔑 KEY STATISTICAL CONCEPTS YOU MUST KNOW

### 1. Mean vs Median
```
Mean   = Sum of all / Count        → Pulled by outliers
Median = Middle value              → NOT affected by outliers

For wages: Always use MEDIAN because a few billionaires 
           pull the mean upward and distort it!
```

### 2. Standard Deviation
```
Low std  → Values are clustered near the mean
High std → Values are spread out wide

Example: If mean wage = ₹5000 and std = ₹500,
         most people earn between ₹4500 and ₹5500
```

### 3. Percentiles / Quantiles
```
25th percentile (Q1) → 25% of people earn LESS than this
50th percentile (Q2) → Median
75th percentile (Q3) → 75% of people earn LESS than this
IQR = Q3 - Q1       → The "middle 50%" range
```

### 4. Log Transformation
```
Problem:  Wages are right-skewed (most earn low, few earn very high)
Solution: Apply log(x) to compress the scale

Before: [100, 200, 500, 1000, 50000]  ← outlier pulls analysis
After:  [4.6, 5.3, 6.2, 6.9,  10.8]  ← now balanced
```

---

## 🎨 CHART SELECTION GUIDE

```
I have ONE variable:
  ├── It's a NUMBER → Histogram or KDE plot
  └── It's a CATEGORY → Bar chart or Pie chart

I have TWO variables:
  ├── Number vs Number → Scatter plot or Line plot
  ├── Category vs Number → Box plot or Violin plot
  └── Category vs Category → Heatmap or Grouped bar

I have MANY variables:
  ├── All numbers → Correlation Heatmap
  └── Mixed → FacetGrid or Pair plot
```

---

## 📏 EDA QUALITY CHECKLIST (For Your Submission)

Before submitting, make sure you have:

- [ ] ✅ Checked `.shape`, `.info()`, `.describe()`
- [ ] ✅ Checked missing values with `.isnull().sum()`
- [ ] ✅ Removed or handled missing values
- [ ] ✅ Checked and removed duplicates
- [ ] ✅ Detected and handled outliers
- [ ] ✅ Done Univariate analysis for ALL key columns
- [ ] ✅ Done Bivariate analysis for target variable vs each feature
- [ ] ✅ Applied log transformation where needed
- [ ] ✅ Created correlation heatmap
- [ ] ✅ Every chart has: Title, X-axis label, Y-axis label
- [ ] ✅ Written 2-3 lines of insight BELOW every chart
- [ ] ✅ Connected every chart back to the research question

---

## ✍️ HOW TO WRITE INSIGHTS (Most Important!)

After EVERY chart, you MUST write what you see. This is what separates
a good EDA from a bad one.

### ❌ Bad insight:
> "The chart shows the distribution of wages."

### ✅ Good insight:
> "The wage distribution is heavily right-skewed, with a median of ₹4,200 per week.
> Over 75% of workers earn below ₹8,000, while a small fraction earns above ₹50,000.
> This confirms the extreme income inequality in the Indian labor market and justifies
> our use of log transformation before model training."

---

> **Remember: EDA is not about making pretty charts.
> It is about TELLING A STORY WITH DATA.
> Every chart must answer a question and prove a point.**
