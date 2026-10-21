import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Create a sample dataset
data = {
    "Hours_Studied": [2, 4, 6, 8, 10],
    "Marks": [20, 35, 50, 65, 80]
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df)

# Min-Max Normalization
minmax_scaler = MinMaxScaler()
normalized_data = pd.DataFrame(
    minmax_scaler.fit_transform(df),
    columns=df.columns
)

print("\nMin-Max Normalized Data:")
print(normalized_data.round(2))

# Z-Score Standardization
standard_scaler = StandardScaler()
standardized_data = pd.DataFrame(
    standard_scaler.fit_transform(df),
    columns=df.columns
)

print("\nZ-Score Standardized Data:")
print(standardized_data.round(2))
