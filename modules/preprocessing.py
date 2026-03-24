import pandas as pd

def clean_data(df):
    df = df.copy()

    # 1. Fix the Currency (Removes ₹, Rs, INR, etc.)
    cols_to_fix = ['transaction_amount', 'amt', 'account_balance']
    for col in cols_to_fix:
        if col in df.columns:
            # Remove everything except numbers and dots
            df[col] = df[col].astype(str).str.replace(r'[^\d.]', '', regex=True)
            # Convert to actual float
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # Fill empty spots with the average (median)
            df[col] = df[col].fillna(df[col].median())

    # 2. Harmonize the Dates
    if 'transaction_timestamp' in df.columns:
        # Try to catch Unix timestamps and standard dates together
        df['transaction_timestamp'] = pd.to_datetime(df['transaction_timestamp'], errors='coerce', unit='s', origin='unix')
        df['transaction_timestamp'] = df['transaction_timestamp'].fillna(pd.to_datetime(df['transaction_timestamp'], errors='coerce'))

    return df