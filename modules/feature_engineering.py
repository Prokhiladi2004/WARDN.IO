import pandas as pd
import numpy as np
import ipaddress

def engineer_features(df):
    df = df.copy()
    
    # 1. HANDLE MISSING COLUMNS (The 73-79 missing columns issue)
    # We define the "Must-Have" columns for the model. 
    # If they are missing, we create them with 0s so the AI doesn't crash.
    required_cols = ['transaction_amount', 'hour', 'user_velocity', 'latitude', 'longitude']
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0 
            

    # 2. HANDLE NULLS (The 21 missing data points)
    # Fill numeric nulls with the median, and text nulls with "Unknown"
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    object_cols = df.select_dtypes(include=['object']).columns
    df[object_cols] = df[object_cols].fillna("Unknown")

    # 3. HANDLE INVALID IPs (The 107 invalid IPs issue)
    if 'ip_address' in df.columns:
        def validate_ip(ip):
            try:
                ipaddress.ip_address(str(ip))
                return 1 # Valid
            except ValueError:
                return 0 # Invalid
        
        # Create a 'is_valid_ip' feature. Fraudsters often use fake/malformed IPs.
        df['is_valid_ip'] = df['ip_address'].apply(validate_ip)
        # Convert IP to numeric format (First Octet) for the AI to read
        df['ip_prefix'] = df['ip_address'].apply(lambda x: str(x).split('.')[0] if '.' in str(x) else 0)
        df['ip_prefix'] = pd.to_numeric(df['ip_prefix'], errors='coerce').fillna(0)

    # 4. CLEAN CURRENCY (₹3159 Fix)
    for col in ['transaction_amount', 'amt', 'account_balance']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(r'[^\d.]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # 5. TARGET ENCODING
    if 'transaction_status' in df.columns:
        df['is_fraud'] = df['transaction_status'].apply(lambda x: 1 if str(x).lower() == 'failed' else 0)

    # 6. RETURN NUMERIC ONLY (Crucial for SMOTE and Training)
    return df.select_dtypes(include=[np.number])