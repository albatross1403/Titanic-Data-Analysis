import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load Titanic dataset
data = sns.load_dataset('titanic').copy()

# Handle missing values efficiently
data.fillna({
    'age': data['age'].median(),
    'embarked': data['embarked'].mode()[0],
    'fare': data['fare'].median()
}, inplace=True)

# Drop unnecessary column
data.drop(columns=['deck'], inplace=True, errors='ignore')

# Remove duplicates
data.drop_duplicates(inplace=True)

# Remove outliers in 'fare' using IQR method
Q1, Q3 = data['fare'].quantile([0.25, 0.75])
IQR = Q3 - Q1
data = data[(data['fare'].between(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR))]

# Standardize categorical values efficiently
data['sex'] = data['sex'].str.lower()
data['embark_town'] = data['embark_town'].str.title()

# Save cleaned dataset
data.to_csv('cleaned_titanic.csv', index=False)

# Visualization: Age Distribution, Age by Gender, and Correlation Matrix
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.histplot(data['age'], bins=20, kde=True, ax=axes[0])
axes[0].set_title('Age Distribution')

sns.boxplot(x='sex', y='age', data=data, ax=axes[1])
axes[1].set_title('Age by Gender')

sns.heatmap(data.select_dtypes(include=['number']).corr(), annot=True, cmap='coolwarm', ax=axes[2])
axes[2].set_title('Correlation Matrix')

plt.tight_layout()
plt.show()
