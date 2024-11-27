# Step 1: Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston

# Step 2: Load the Dataset
boston = load_boston()
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df['PRICE'] = boston.target  # Add target variable for prices

# Step 3: Basic Data Exploration
print("Dataset Info:")
print(df.info())
print("\nDataset Description:")
print(df.describe())

# Step 4: Data Visualization

# 1. Distribution of House Prices
plt.figure(figsize=(8, 6))
sns.histplot(df['PRICE'], bins=30, kde=True)
plt.title('Distribution of House Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()

# 2. Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

# 3. Boxplot for Price vs. Crime Rate
plt.figure(figsize=(8, 6))
sns.boxplot(x=df['CRIM'], y=df['PRICE'])
plt.title('Boxplot: Crime Rate vs. House Prices')
plt.xlabel('Crime Rate')
plt.ylabel('House Prices')
plt.show()
