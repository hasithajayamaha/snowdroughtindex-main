import pandas as pd
import numpy as np
import os

def analyze_csv_file(filepath):
    """Analyze a CSV file and return key insights"""
    print(f"\n{'='*60}")
    print(f"ANALYZING: {os.path.basename(filepath)}")
    print(f"{'='*60}")
    
    try:
        # Read the file
        df = pd.read_csv(filepath)
        
        # Basic info
        print(f"File size: {os.path.getsize(filepath) / (1024*1024):.2f} MB")
        print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        print(f"Memory usage: {df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB")
        
        # Column information
        print(f"\nColumns ({len(df.columns)}):")
        for i, col in enumerate(df.columns):
            dtype = df[col].dtype
            non_null = df[col].count()
            null_count = df[col].isnull().sum()
            print(f"  {i+1:2d}. {col:<30} | {str(dtype):<10} | {non_null:>8,} non-null | {null_count:>6,} null")
        
        # Data types summary
        print(f"\nData Types Summary:")
        dtype_counts = df.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            print(f"  {str(dtype):<15}: {count} columns")
        
        # Missing data analysis
        print(f"\nMissing Data Analysis:")
        missing_data = df.isnull().sum()
        missing_percent = (missing_data / len(df)) * 100
        missing_summary = pd.DataFrame({
            'Missing Count': missing_data,
            'Missing %': missing_percent
        }).sort_values('Missing %', ascending=False)
        
        # Show columns with missing data
        has_missing = missing_summary[missing_summary['Missing Count'] > 0]
        if len(has_missing) > 0:
            print("  Columns with missing data:")
            for col, row in has_missing.head(10).iterrows():
                print(f"    {col:<30}: {row['Missing Count']:>8,} ({row['Missing %']:>6.2f}%)")
        else:
            print("  No missing data found!")
        
        # Numeric columns analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print(f"\nNumeric Columns Analysis ({len(numeric_cols)} columns):")
            numeric_stats = df[numeric_cols].describe()
            print(numeric_stats.round(4))
        
        # Sample data
        print(f"\nFirst 5 rows:")
        print(df.head().to_string())
        
        # Unique values for categorical columns (if any)
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            print(f"\nCategorical Columns Analysis:")
            for col in categorical_cols[:5]:  # Limit to first 5 categorical columns
                unique_count = df[col].nunique()
                print(f"  {col}: {unique_count} unique values")
                if unique_count <= 10:
                    print(f"    Values: {list(df[col].unique())}")
        
        return df
        
    except Exception as e:
        print(f"Error analyzing file: {e}")
        return None

def compare_files(df1, df2, file1_name, file2_name):
    """Compare two dataframes"""
    print(f"\n{'='*60}")
    print(f"COMPARISON: {file1_name} vs {file2_name}")
    print(f"{'='*60}")
    
    if df1 is None or df2 is None:
        print("Cannot compare - one or both files failed to load")
        return
    
    print(f"Shape comparison:")
    print(f"  {file1_name}: {df1.shape[0]:,} rows × {df1.shape[1]} columns")
    print(f"  {file2_name}: {df2.shape[0]:,} rows × {df2.shape[1]} columns")
    
    # Column comparison
    cols1 = set(df1.columns)
    cols2 = set(df2.columns)
    
    common_cols = cols1.intersection(cols2)
    unique_to_1 = cols1 - cols2
    unique_to_2 = cols2 - cols1
    
    print(f"\nColumn comparison:")
    print(f"  Common columns: {len(common_cols)}")
    print(f"  Unique to {file1_name}: {len(unique_to_1)}")
    print(f"  Unique to {file2_name}: {len(unique_to_2)}")
    
    if unique_to_1:
        print(f"  Columns only in {file1_name}: {list(unique_to_1)}")
    if unique_to_2:
        print(f"  Columns only in {file2_name}: {list(unique_to_2)}")
    
    # Compare common numeric columns
    if common_cols:
        numeric_common = [col for col in common_cols if df1[col].dtype in ['int64', 'float64'] and df2[col].dtype in ['int64', 'float64']]
        if numeric_common:
            print(f"\nCommon numeric columns statistics comparison:")
            for col in numeric_common[:5]:  # Limit to first 5 for readability
                print(f"\n  {col}:")
                print(f"    {file1_name}: mean={df1[col].mean():.4f}, std={df1[col].std():.4f}, min={df1[col].min():.4f}, max={df1[col].max():.4f}")
                print(f"    {file2_name}: mean={df2[col].mean():.4f}, std={df2[col].std():.4f}, min={df2[col].min():.4f}, max={df2[col].max():.4f}")

if __name__ == "__main__":
    # File paths
    file1 = "data/output_data/elevation/elevation_extracted_full_CaSR_v3.1_A_PR24_SFC_combined_full.csv"
    file2 = "data/output_data/elevation/elevation_extracted_full_CaSR_v3.1_P_SWE_LAND_combined_full.csv"
    
    # Analyze both files
    df1 = analyze_csv_file(file1)
    df2 = analyze_csv_file(file2)
    
    # Compare the files
    compare_files(df1, df2, "PR24_SFC", "SWE_LAND")
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")
