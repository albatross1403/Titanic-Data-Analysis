# Titanic-Data-Analysis
Exploratory Data Analysis (EDA) on the Titanic dataset
# Titanic Dataset Analysis 🛳️

## 📌 Project Overview
This project performs **data cleaning** and **Exploratory Data Analysis (EDA)** on the Titanic dataset. It covers:
- Handling missing values, outliers, and duplicates
- Standardizing categorical values
- Generating statistical and graphical insights

## 📂 Files in This Repository
- `titanic_data_analysis.py` → Python script for data cleaning & EDA
- `cleaned_titanic.csv` → Cleaned dataset after processing
- `Titanic_Analysis_Report.md` → Detailed analysis report
- `README.md` → This file (Project description)

## 🔧 How to Run the Code
1. Install dependencies:  
   ```sh
   pip install pandas seaborn matplotlib
   ```
2. Run the script:  
   ```sh
   python titanic_data_analysis.py
   ```
3. The cleaned dataset will be generated as `cleaned_titanic.csv`.

## 📊 Key Insights
- **Most missing values were in the Cabin column, which was dropped**.
- **Age was filled with the median, and Embarked was filled with the mode**.
- **Outliers in Fare were removed using the IQR method**.
- **Correlation matrix shows strong relationships between Fare, Pclass, and Survival**.

## 📝 License
This project is for educational purposes only.

