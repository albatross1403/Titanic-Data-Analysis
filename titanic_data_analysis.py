import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Titanic dataset
data = sns.load_dataset('titanic')

# Display basic info
data.info()

# Handling Missing Values
# Fill missing age values with median
data['age'].fillna(data['age'].median(), inplace=True)
# Fill missing embarked values with mode
data['embarked'].fillna(data['embarked'].mode()[0], inplace=True)
# Drop 'deck' column due to many missing values
data.drop(columns=['deck'], inplace=True)

# Removing Duplicates
data.drop_duplicates(inplace=True)

# Detecting and Treating Outliers
# Using IQR method for 'fare' column
Q1 = data['fare'].quantile(0.25)
Q3 = data['fare'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
data = data[(data['fare'] >= lower_bound) & (data['fare'] <= upper_bound)]

# Standardizing Categorical Values
data['sex'] = data['sex'].str.lower()
data['embark_town'] = data['embark_town'].str.title()

# Save cleaned dataset
data.to_csv('cleaned_titanic.csv', index=False)

# Univariate Analysis
plt.figure(figsize=(10,5))
sns.histplot(data['age'], bins=20, kde=True)
plt.title('Age Distribution')
plt.show()

# Bivariate Analysis
plt.figure(figsize=(8,6))
sns.boxplot(x='sex', y='age', data=data)
plt.title('Age Distribution by Gender')
plt.show()

# Correlation Matrix
plt.figure(figsize=(8,6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
