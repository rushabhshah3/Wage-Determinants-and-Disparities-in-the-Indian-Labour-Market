# ==============================================================================
# SCRIPT: harmonize_5year_plfs.py
# Purpose: Compiles 2021-2025 raw MoSPI PLFS microdata into a clean master dataset
# ==============================================================================
import os
import numpy as np
import pandas as pd

STATE_MAP = {
    1: 'Jammu & Kashmir', 2: 'Himachal Pradesh', 3: 'Punjab', 4: 'Chandigarh',
    5: 'Uttarakhand', 6: 'Haryana', 7: 'Delhi', 8: 'Rajasthan', 9: 'Uttar Pradesh',
    10: 'Bihar', 11: 'Sikkim', 12: 'Arunachal Pradesh', 13: 'Nagaland',
    14: 'Manipur', 15: 'Mizoram', 16: 'Tripura', 17: 'Meghalaya', 18: 'Assam',
    19: 'West Bengal', 20: 'Jharkhand', 21: 'Odisha', 22: 'Chhattisgarh',
    23: 'Madhya Pradesh', 24: 'Gujarat', 25: 'Daman & Diu', 26: 'Dadra & Nagar Haveli',
    27: 'Maharashtra', 28: 'Andhra Pradesh', 29: 'Karnataka', 30: 'Goa',
    31: 'Lakshadweep', 32: 'Kerala', 33: 'Tamil Nadu', 34: 'Puducherry',
    35: 'Andaman & Nicobar', 36: 'Telangana', 37: 'Ladakh'
}

EDU_MAP = {
    1: 'Not Literate', 2: 'Literate (No Formal School)', 3: 'Literate (Others)', 4: 'Literate (Others)',
    5: 'Below Primary', 6: 'Primary', 7: 'Middle', 8: 'Secondary',
    10: 'Higher Secondary', 11: 'Diploma/Certificate', 12: 'Graduate', 13: 'Post Graduate & Above'
}

EDU_YEARS = {
    'Not Literate': 0, 'Literate (No Formal School)': 2, 'Literate (Others)': 3,
    'Below Primary': 4, 'Primary': 5, 'Middle': 8, 'Secondary': 10,
    'Higher Secondary': 12, 'Diploma/Certificate': 14, 'Graduate': 15, 'Post Graduate & Above': 17
}

# Reserve Bank of India CPI Base Deflator (2021 = 100.0)
CPI_DEFLATOR = {2021: 100.0, 2022: 106.7, 2023: 112.5, 2024: 118.6, 2025: 124.8}

def load_year_2021(base_dir):
    path = os.path.join(base_dir, '01_Periodic Labour Force Survey (PLFS), July 2021-June 2022/PLFS_Data_2021-22_CSV/perv1.csv')
    print(f"Loading 2021 from {path}...")
    hrs_cols = [
        'b6q6_act1_3pt1_perv1', 'b6q6_act2_3pt2_perv1', 'b6q6_act2_3pt3_perv1',
        'b6q6_act2_3pt4_perv1', 'b6q6_act2_3pt5_act2_perv1', 'b6q6_act2_3pt6_perv1', 'b6q6_act2_3pt7_perv1'
    ]
    usecols = ['state_perv1', 'b1q3_perv1', 'b4q5_perv1', 'b4q6_perv1', 'b4q8_perv1', 'b6q9_perv1'] + hrs_cols
    df = pd.read_csv(path, usecols=usecols)
    df['Weekly_Hours'] = df[hrs_cols].sum(axis=1)
    df = df.rename(columns={
        'state_perv1': 'State', 'b1q3_perv1': 'Sector', 'b4q5_perv1': 'Gender',
        'b4q6_perv1': 'Age', 'b4q8_perv1': 'Education', 'b6q9_perv1': 'Weekly_Wage'
    })
    df['Year'] = 2021
    return df[['Year', 'State', 'Sector', 'Gender', 'Age', 'Education', 'Weekly_Wage', 'Weekly_Hours']]

def load_year_2022(base_dir):
    path = os.path.join(base_dir, '02_Periodic Labour Force Survey (PLFS), Calendar Year 2022 (Jan22 Dec22)/PLFS_Data_2022-22_CSV/cperv1.csv')
    print(f"Loading 2022 from {path}...")
    hrs_cols = [f'b6q6_3pt{i}_cperv1' for i in range(1, 8)]
    usecols = ['state_cperv1', 'b1q3_cperv1', 'b4q5_cperv1', 'b4q6_perv1', 'b4q8_cperv1', 'b6q9_cperv1'] + hrs_cols
    df = pd.read_csv(path, usecols=usecols)
    df['Weekly_Hours'] = df[hrs_cols].sum(axis=1)
    df = df.rename(columns={
        'state_cperv1': 'State', 'b1q3_cperv1': 'Sector', 'b4q5_cperv1': 'Gender',
        'b4q6_perv1': 'Age', 'b4q8_cperv1': 'Education', 'b6q9_cperv1': 'Weekly_Wage'
    })
    df['Year'] = 2022
    return df[['Year', 'State', 'Sector', 'Gender', 'Age', 'Education', 'Weekly_Wage', 'Weekly_Hours']]

def load_year_2023(base_dir):
    path = os.path.join(base_dir, '03_Periodic Labour Force Survey (PLFS), Calendar Year 2023 (Jan23 Dec23)/CSV_PLFS_Calendar_Year_2023/cperv1.csv')
    print(f"Loading 2023 from {path}...")
    hrs_cols = [f'b6q6_3pt{i}_cperv1' for i in range(1, 8)]
    usecols = ['state_cperv1', 'b1q3_cperv1', 'b4q5_cperv1', 'b4q6_perv1', 'b4q8_cperv1', 'b6q9_cperv1'] + hrs_cols
    df = pd.read_csv(path, usecols=usecols)
    df['Weekly_Hours'] = df[hrs_cols].sum(axis=1)
    df = df.rename(columns={
        'state_cperv1': 'State', 'b1q3_cperv1': 'Sector', 'b4q5_cperv1': 'Gender',
        'b4q6_perv1': 'Age', 'b4q8_cperv1': 'Education', 'b6q9_cperv1': 'Weekly_Wage'
    })
    df['Year'] = 2023
    return df[['Year', 'State', 'Sector', 'Gender', 'Age', 'Education', 'Weekly_Wage', 'Weekly_Hours']]

def load_year_2024(base_dir):
    path = os.path.join(base_dir, '04_Periodic Labour Force Survey (PLFS), Key Employment Unemployment Indicators for (January 2024 - December 2024)/Data in CSV/cperv1.csv')
    print(f"Loading 2024 from {path}...")
    col_map = {
        'State_UT_Code': 'State', 'Sector': 'Sector', 'Sex': 'Gender',
        'Age': 'Age', 'General_Education_Level': 'Education',
        'CWS_Earnings_Salaried': 'Weekly_Wage', 'Day7_Total_Hours': 'Weekly_Hours'
    }
    df = pd.read_csv(path, usecols=list(col_map.keys()))
    df = df.rename(columns=col_map)
    df['Year'] = 2024
    return df[['Year', 'State', 'Sector', 'Gender', 'Age', 'Education', 'Weekly_Wage', 'Weekly_Hours']]

def load_year_2025(base_dir):
    path = os.path.join(base_dir, '05_Periodic Labour Force Survey (PLFS),Calendar Year 2025 (Jan-Dec25)/Data_in_CSV (1)/CPERV1.csv')
    print(f"Loading 2025 from {path}...")
    col_map = {
        'st': 'State', 'sec': 'Sector', 'sex': 'Gender',
        'age': 'Age', 'gedu_lvl': 'Education',
        'ern_reg': 'Weekly_Wage', 'tothrs_wrk': 'Weekly_Hours'
    }
    df = pd.read_csv(path, usecols=list(col_map.keys()))
    df = df.rename(columns=col_map)
    df['Year'] = 2025
    return df[['Year', 'State', 'Sector', 'Gender', 'Age', 'Education', 'Weekly_Wage', 'Weekly_Hours']]

def build_harmonized_dataset(base_dir="eda_data", output_csv="PLFS_2021_2025_Harmonized_Clean.csv"):
    dfs = [
        load_year_2021(base_dir),
        load_year_2022(base_dir),
        load_year_2023(base_dir),
        load_year_2024(base_dir),
        load_year_2025(base_dir)
    ]
    print("Concatenating 5-year microdata...")
    master = pd.concat(dfs, ignore_index=True)
    total_raw = len(master)
    print(f"Total Raw Citizens Surveyed: {total_raw:,}")

    # Standardize Demographic Labels
    master['Sector_Desc'] = master['Sector'].map({1: 'Rural', 2: 'Urban'})
    master['Gender_Desc'] = master['Gender'].map({1: 'Male', 2: 'Female', 3: 'Transgender'})
    master['State_Name'] = master['State'].map(STATE_MAP)
    master['Education_Desc'] = master['Education'].map(EDU_MAP)

    # Filter 1: Working Age Population (15 - 65)
    working_age = master[(master['Age'] >= 15) & (master['Age'] <= 65)].copy()
    print(f"Working-age population (15-65): {len(working_age):,} ({len(working_age)/total_raw*100:.1f}%)")

    # Filter 2: Active Regular Salaried Earners (Weekly_Wage > 0)
    earners = working_age[working_age['Weekly_Wage'] > 0].copy()
    print(f"Active Salaried Wage Earners: {len(earners):,} ({len(earners)/total_raw*100:.1f}%)")

    # Filter 3: Normalize Weekly Working Hours (cap between 10 and 84 hours/week to eliminate noise)
    earners['Weekly_Hours'] = earners['Weekly_Hours'].clip(lower=10, upper=84)
    
    # Compute Hourly Wage
    earners['Hourly_Wage'] = earners['Weekly_Wage'] / earners['Weekly_Hours']
    
    # Filter 4: Outlier Truncation via Tukey's IQR Fences on Hourly Wage
    q1 = earners['Hourly_Wage'].quantile(0.01)
    q99 = earners['Hourly_Wage'].quantile(0.99)
    earners = earners[(earners['Hourly_Wage'] >= q1) & (earners['Hourly_Wage'] <= q99)].copy()
    print(f"After Truncating 1st and 99th percentile wage noise: {len(earners):,}")

    # Log Transformation
    earners['Log_Hourly_Wage'] = np.log(earners['Hourly_Wage'])

    # Inflation Adjustment (Real Wage Rate)
    earners['CPI'] = earners['Year'].map(CPI_DEFLATOR)
    earners['Real_Hourly_Wage'] = (earners['Hourly_Wage'] / earners['CPI']) * 100.0
    earners['Log_Real_Hourly_Wage'] = np.log(earners['Real_Hourly_Wage'])

    # Mincer Potential Work Experience
    earners['Schooling_Years'] = earners['Education_Desc'].map(EDU_YEARS).fillna(6)
    earners['Experience'] = (earners['Age'] - earners['Schooling_Years'] - 6).clip(lower=0)
    earners['Experience_Sq'] = earners['Experience'] ** 2

    # Save to clean CSV
    print(f"Writing clean dataset to {output_csv}...")
    earners.to_csv(output_csv, index=False)
    file_size_mb = os.path.getsize(output_csv) / (1024 * 1024)
    print(f"SUCCESS: {len(earners):,} records successfully saved to {output_csv} ({file_size_mb:.2f} MB)")
    return earners

if __name__ == '__main__':
    build_harmonized_dataset()
