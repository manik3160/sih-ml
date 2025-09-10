"""
Data Parser for Gas Leakage Detection Dataset
Parses the Excel file and creates a properly formatted DataFrame
"""

import pandas as pd
import numpy as np
from datetime import datetime

def parse_dataset(file_path):
    """
    Parse the gas leakage dataset from Excel file
    
    Args:
        file_path (str): Path to the Excel file
        
    Returns:
        pd.DataFrame: Properly formatted dataset
    """
    # Read the Excel file
    df = pd.read_excel(file_path)
    
    # Get the column name (it's all in one column)
    col_name = df.columns[0]
    
    # Split the data properly
    data_rows = []
    for idx, row in df.iterrows():
        values = str(row[col_name]).split(',')
        data_rows.append(values)
    
    # Create proper DataFrame
    df_parsed = pd.DataFrame(data_rows)
    
    # Set the first row as column headers
    df_parsed.columns = df_parsed.iloc[0]
    df_parsed = df_parsed.drop(0).reset_index(drop=True)
    
    # Define proper column names based on the actual data structure
    proper_columns = [
        'timestamp', 'location_id', 'location_name', 'process_area', 'equipment_id',
        'CO2_ppm', 'CO_ppm', 'SO2_ppm', 'CH4_ppm', 'H2S_ppm', 'NOx_ppm',
        'temperature_C', 'humidity_percent', 'pressure_kPa', 'wind_speed_ms',
        'vibration_level_mm_s', 'equipment_age_months', 'maintenance_days_ago',
        'production_rate_percent', 'leakage_detected', 'leakage_severity',
        'predicted_leak_location', 'warning_level'
    ]
    
    # Rename columns
    df_parsed.columns = proper_columns
    
    # Convert data types
    numeric_columns = [
        'CO2_ppm', 'CO_ppm', 'SO2_ppm', 'CH4_ppm', 'H2S_ppm', 'NOx_ppm',
        'temperature_C', 'humidity_percent', 'pressure_kPa', 'wind_speed_ms',
        'vibration_level_mm_s', 'equipment_age_months', 'maintenance_days_ago',
        'production_rate_percent', 'leakage_detected'
    ]
    
    for col in numeric_columns:
        df_parsed[col] = pd.to_numeric(df_parsed[col], errors='coerce')
    
    # Convert timestamp
    df_parsed['timestamp'] = pd.to_datetime(df_parsed['timestamp'])
    
    return df_parsed

def get_dataset_info(df):
    """
    Get comprehensive information about the dataset
    
    Args:
        df (pd.DataFrame): The parsed dataset
        
    Returns:
        dict: Dataset information
    """
    info = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'data_types': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'numeric_summary': df.describe().to_dict(),
        'categorical_summary': {}
    }
    
    # Get categorical column summaries
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        info['categorical_summary'][col] = {
            'unique_values': df[col].nunique(),
            'value_counts': df[col].value_counts().to_dict()
        }
    
    return info

if __name__ == "__main__":
    # Parse the dataset
    df = parse_dataset('dataset.xlsx')
    
    print("Dataset Shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nFirst few rows:")
    print(df.head())
    print("\nData types:")
    print(df.dtypes)
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nBasic statistics:")
    print(df.describe())
    
    # Save parsed dataset
    df.to_csv('parsed_dataset.csv', index=False)
    print("\nParsed dataset saved as 'parsed_dataset.csv'")
