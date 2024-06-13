import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    return pd.read_csv(file_path, index_col=None)

def save_data(df, file_path):
    df.to_csv(file_path, index=None)

def fill_missing_values(df, value):
    return df.fillna(value)

def standardize_data(df, columns):
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df
