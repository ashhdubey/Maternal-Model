import pandas as pd

def preprocess_data(df):
    df = df.copy()

    df.columns = df.columns.str.lower().str.replace(" ", "_")

    numeric_cols = ["systolic_bp", "diastolic", "bs", "heart_rate"]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    return df