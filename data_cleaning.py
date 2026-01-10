import pandas as pd
import numpy as np

# Load raw data
df = pd.read_csv('../data/raw/customer_credit_data.csv')

# Create a copy
df_clean = df.copy()

# Handle missing numerical values
num_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns
for col in num_cols:
    df_clean[col].fillna(df_clean[col].median(), inplace=True)

# Handle missing categorical values
cat_cols = df_clean.select_dtypes(include='object').columns
for col in cat_cols:
    df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
    df_clean[col] = df_clean[col].str.strip().str.lower()

# Remove duplicates
df_clean.drop_duplicates(inplace=True)

# Feature Engineering
df_clean['Age_Group'] = pd.cut(
    df_clean['Age'],
    bins=[0, 25, 35, 50, 65, 120],
    labels=['Young', 'Early Career', 'Mid Career', 'Senior', 'Retired']
)

df_clean['High_Credit_Utilization'] = df_clean['Credit_Utilization'] > 0.5

# Save cleaned data
df_clean.to_csv('../data/cleaned/cleaned_customer_credit_data.csv', index=False)

print("Data cleaning completed successfully.")
