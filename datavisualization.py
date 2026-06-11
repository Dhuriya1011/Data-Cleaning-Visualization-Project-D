import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Sample Raw Dataset
data = {
    'Student_ID': [1, 2, 3, 4, 5, 5, 6, 7, 8, 9],
    'Study_Hours': [2, 3, np.nan, 5, 6, 6, 7, 8, 20, 4],
    'Marks': [45, 50, 60, np.nan, 75, 75, 80, 90, 200, 58]
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df)

# -------------------------
# DATA CLEANING
# -------------------------

# Handle Missing Values
df['Study_Hours'].fillna(df['Study_Hours'].mean(), inplace=True)
df['Marks'].fillna(df['Marks'].mean(), inplace=True)

# Remove Duplicates
df.drop_duplicates(inplace=True)

print("\nDataset After Cleaning:")
print(df)

# -------------------------
# OUTLIER DETECTION
# -------------------------

plt.figure(figsize=(6,4))
sns.boxplot(y=df['Marks'])
plt.title("Outlier Detection in Marks")
plt.show()

# -------------------------
# VISUALIZATION 1
# -------------------------

plt.figure(figsize=(6,4))
plt.hist(df['Marks'], bins=5)
plt.title("Marks Distribution")
plt.xlabel("Marks")
plt.ylabel("Frequency")
plt.show()

# -------------------------
# VISUALIZATION 2
# -------------------------

plt.figure(figsize=(6,4))
plt.scatter(df['Study_Hours'], df['Marks'])
plt.title("Study Hours vs Marks")
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.show()

# -------------------------
# VISUALIZATION 3
# -------------------------

plt.figure(figsize=(6,4))
sns.heatmap(df[['Study_Hours','Marks']].corr(),
            annot=True,
            cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()