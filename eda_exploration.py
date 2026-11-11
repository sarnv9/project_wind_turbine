"""
Exploratory data analysis: wind turbine predictive maintenance
===============================================================
 
Goal: understand the sensor data well enough to take justified preprocessing
decisions before modelling the operational status (0 = Normal, 1 = Warning,
2 = Failure).
 
Steps: data quality (nulls, duplicated columns) -> variable types ->
correlations -> outliers -> categorical variables vs the target -> class balance.
 
Input : data/wind_turbine_data.csv
Output: images/*.png (figures used in the README)
        data/wind_turbine_cleaned.csv (same data without the duplicated sensor
        column; this is the input of the modelling script)
"""
 
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuration for visualizations
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
sns.set(font_scale=1.2)

os.makedirs('images', exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Load data and basic quality checks
# ---------------------------------------------------------------------------

df = pd.read_csv('data/wind_turbine_data.csv')

 
print("General information about the dataset:")
print(df.info())
 
print("\nFirst 5 rows of the dataset:")
print(df.head())
 
print("\nDescriptive statistics:")
print(df.describe())
 
print("\nMissing values per column:")
print(df.isnull().sum().to_string())


# ---------------------------------------------------------------------------
# 2. Duplicated columns
# ---------------------------------------------------------------------------
def getDuplicateColumns(df):
    """Return the names of columns that are exact copies of an earlier column.
 
    Uses Series.equals(), which is stricter than a correlation of 1.0: it
    confirms the values are identical, not just linearly related.
    """
    duplicateColumnNames = set()
 
    for x in range(df.shape[1]):
        col = df.iloc[:, x]
 
        for y in range(x + 1, df.shape[1]):
            otherCol = df.iloc[:, y]
 
            if col.equals(otherCol):
                duplicateColumnNames.add(df.columns.values[y])
 
    return list(duplicateColumnNames)
 
duplicateColNames = getDuplicateColumns(df)
 
for column in duplicateColNames:
    print('Duplicated Column Name: ', column)

df2 = df.drop(columns=['Rotor_Shaft_Vibration_Redundant'])
print(df2.dtypes)


# ---------------------------------------------------------------------------
# 3. Variable types and target
# ---------------------------------------------------------------------------
cat_columns = df2.select_dtypes(include=['object']).columns.tolist()
num_columns = df2.select_dtypes(include=['int64', 'float64']).columns.tolist()
 
print("\nCategorical variables:", cat_columns)
print("Numerical variables:", num_columns)
 
# The target is the last column of the dataset.
target_column = df2.columns[-1]
print("\nTarget variable:", target_column)


# ---------------------------------------------------------------------------
# 4. Correlations
# ---------------------------------------------------------------------------

corr_matrix = df2[num_columns].corr()

print(corr_matrix["Operational_Status"].sort_values(ascending=False))
 
# Correlation matrix of the numeric variables (the target is the last row and
# column of the matrix).
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='RdBu_r', center=0,
            linewidths=0.5, cbar_kws={"shrink": .8}, annot_kws={"size": 8})
plt.title('Correlation Matrix of Numeric Features')
plt.savefig('images/correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.close()



# ---------------------------------------------------------------------------
# 5. Outliers (IQR rule)
# ---------------------------------------------------------------------------

outlier_stats = pd.DataFrame()
 
active_num_columns = [col for col in num_columns if col in df2.columns]
 
# [:-1] skips the target, which is the last numeric column.
for column in active_num_columns[:-1]:
    Q1 = df2[column].quantile(0.25)
    Q3 = df2[column].quantile(0.75)
    IQR = Q3 - Q1
 
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
 
    outliers_count = df2[(df2[column] < lower_bound) | (df2[column] > upper_bound)].shape[0]
    outliers_percentage = (outliers_count / df2.shape[0]) * 100
 
    outlier_stats[column] = [outliers_count, outliers_percentage]
 
outlier_stats.index = ['Outliers Count', 'Outliers Percentage (%)']
outliers_sorted = outlier_stats.T.sort_values(by='Outliers Count', ascending=False)
print(outliers_sorted.round(2))

# ---------------------------------------------------------------------------
# 6. Categorical variables vs the target
# ---------------------------------------------------------------------------
for col in cat_columns:
    print(f"{col}': {df2[col].nunique()} unique values")
    print(df2[col].value_counts())
 
for cat_col in cat_columns:
    plt.figure(figsize=(8, 6)) # (width, height)
    prop_df = df.groupby(cat_col)[target_column].value_counts(normalize=True).mul(100).rename('percentage').reset_index()
    sns.barplot(x=cat_col, y='percentage', hue=target_column, data=prop_df, palette='viridis')
    plt.title(f'Proportion of "{target_column}" within each category of "{cat_col}"')
    plt.xlabel(cat_col)
    plt.ylabel('Percentage (%)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'images/proportion_by_{cat_col}.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# 7. Target balance
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
 # Counts per class
sns.countplot(x='Operational_Status', data=df2, ax=axes[0], palette='viridis')
axes[0].set_title('Real Distribution of Status')
 
# Same classes as percentages. The labels assume value_counts() orders the
# classes 0, 1, 2 (true here, since Normal > Warning > Failure in frequency).
df2['Operational_Status'].value_counts().plot.pie(
    autopct='%1.1f%%', ax=axes[1], startangle=90, labels=['Normal = 0', 'Warning = 1', 'Failure = 2']
)
axes[1].set_title('Class Balance')
plt.tight_layout()
plt.savefig('images/class_balance.png', dpi=300, bbox_inches='tight')
plt.close()
 
# ---------------------------------------------------------------------------
# 8. Save the cleaned dataset 
# ---------------------------------------------------------------------------
df2.to_csv('data/wind_turbine_cleaned.csv', index=False)