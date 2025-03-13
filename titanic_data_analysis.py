import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load Titanic dataset
data = sns.load_dataset('titanic')

# Show dataset info
data.info()

# Handle missing values
data['age'].fillna(data['age'].median(), inplace=True) 
data['embarked'].fillna(data['embarked'].mode()[0], inplace=True)  
data['fare'].fillna(data['fare'].median(), inplace=True)  
data.drop(columns=['deck'], inplace=True, errors='ignore')  

# Remove duplicate rows
data.drop_duplicates(inplace=True)

# Remove outliers in 'fare' using IQR method
Q1, Q3 = data['fare'].quantile([0.25, 0.75])
IQR = Q3 - Q1
lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
data = data[(data['fare'] >= lower) & (data['fare'] <= upper)]

# Standardize categorical values
data['sex'] = data['sex'].str.lower() 
data['embark_town'] = data['embark_town'].str.title() 
data.to_csv('cleaned_titanic.csv', index=False)

# Plot Age Distribution
plt.figure(figsize=(10, 5))
sns.histplot(data['age'], bins=20, kde=True)
plt.title('Age Distribution')
plt.show()

# Compare Age by Gender
plt.figure(figsize=(8, 6))
sns.boxplot(x='sex', y='age', data=data)
plt.title('Age by Gender')
plt.show()

# Show Correlation Matrix (Only Numeric Columns)
plt.figure(figsize=(8, 6))
sns.heatmap(data.select_dtypes(include=['number']).corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
