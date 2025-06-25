import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def detailed_analysis():
    """Perform detailed analysis of the elevation data files"""
    
    # Load the data
    print("Loading data files...")
    df1 = pd.read_csv("data/output_data/elevation/elevation_extracted_full_CaSR_v3.1_A_PR24_SFC_combined_full.csv")
    df2 = pd.read_csv("data/output_data/elevation/elevation_extracted_full_CaSR_v3.1_P_SWE_LAND_combined_full.csv")
    
    print(f"Data loaded successfully!")
    print(f"PR24_SFC dataset: {df1.shape}")
    print(f"SWE_LAND dataset: {df2.shape}")
    
    # Convert time column to datetime
    df1['time'] = pd.to_datetime(df1['time'])
    df2['time'] = pd.to_datetime(df2['time'])
    
    # Extract temporal components
    df1['year'] = df1['time'].dt.year
    df1['month'] = df1['time'].dt.month
    df1['day'] = df1['time'].dt.day
    df1['hour'] = df1['time'].dt.hour
    
    df2['year'] = df2['time'].dt.year
    df2['month'] = df2['time'].dt.month
    df2['day'] = df2['time'].dt.day
    df2['hour'] = df2['time'].dt.hour
    
    print("\n" + "="*80)
    print("TEMPORAL ANALYSIS")
    print("="*80)
    
    # Time range analysis
    print(f"\nTime Range Analysis:")
    print(f"PR24_SFC:")
    print(f"  Start: {df1['time'].min()}")
    print(f"  End: {df1['time'].max()}")
    print(f"  Duration: {df1['time'].max() - df1['time'].min()}")
    
    print(f"\nSWE_LAND:")
    print(f"  Start: {df2['time'].min()}")
    print(f"  End: {df2['time'].max()}")
    print(f"  Duration: {df2['time'].max() - df2['time'].min()}")
    
    # Year distribution
    print(f"\nYear Distribution:")
    print(f"PR24_SFC years: {sorted(df1['year'].unique())}")
    print(f"SWE_LAND years: {sorted(df2['year'].unique())}")
    
    # Data availability by year
    print(f"\nData Availability by Year:")
    pr24_by_year = df1[df1['CaSR_v3.1_A_PR24_SFC'].notna()]['year'].value_counts().sort_index()
    swe_by_year = df2[df2['CaSR_v3.1_P_SWE_LAND'].notna()]['year'].value_counts().sort_index()
    
    print(f"PR24_SFC non-null records by year:")
    for year, count in pr24_by_year.items():
        print(f"  {year}: {count:,} records")
    
    print(f"\nSWE_LAND non-null records by year:")
    for year, count in swe_by_year.items():
        print(f"  {year}: {count:,} records")
    
    print("\n" + "="*80)
    print("SPATIAL ANALYSIS")
    print("="*80)
    
    # Spatial extent
    print(f"\nSpatial Extent:")
    print(f"Longitude range: {df1['original_lon'].min():.4f} to {df1['original_lon'].max():.4f}")
    print(f"Latitude range: {df1['original_lat'].min():.4f} to {df1['original_lat'].max():.4f}")
    print(f"Grid points: {df1['point_id'].nunique():,} unique locations")
    
    # Elevation statistics by location
    print(f"\nElevation Statistics by Location:")
    elevation_stats = df1.groupby('point_id')[['elevation_min', 'elevation_max', 'elevation_mean', 'elevation_median']].first()
    print(f"Elevation range across all points:")
    print(f"  Minimum elevation: {elevation_stats['elevation_min'].min():.1f} m")
    print(f"  Maximum elevation: {elevation_stats['elevation_max'].max():.1f} m")
    print(f"  Mean elevation (across points): {elevation_stats['elevation_mean'].mean():.1f} m")
    print(f"  Median elevation (across points): {elevation_stats['elevation_median'].median():.1f} m")
    
    print("\n" + "="*80)
    print("CLIMATE DATA ANALYSIS")
    print("="*80)
    
    # PR24_SFC analysis
    pr24_data = df1[df1['CaSR_v3.1_A_PR24_SFC'].notna()]
    print(f"\nPR24_SFC (Precipitation) Analysis:")
    print(f"  Records with data: {len(pr24_data):,} ({len(pr24_data)/len(df1)*100:.2f}%)")
    print(f"  Value range: {pr24_data['CaSR_v3.1_A_PR24_SFC'].min():.6f} to {pr24_data['CaSR_v3.1_A_PR24_SFC'].max():.6f}")
    print(f"  Mean: {pr24_data['CaSR_v3.1_A_PR24_SFC'].mean():.6f}")
    print(f"  Median: {pr24_data['CaSR_v3.1_A_PR24_SFC'].median():.6f}")
    print(f"  Standard deviation: {pr24_data['CaSR_v3.1_A_PR24_SFC'].std():.6f}")
    
    # SWE analysis
    swe_data = df2[df2['CaSR_v3.1_P_SWE_LAND'].notna()]
    print(f"\nSWE_LAND (Snow Water Equivalent) Analysis:")
    print(f"  Records with data: {len(swe_data):,} ({len(swe_data)/len(df2)*100:.2f}%)")
    print(f"  Value range: {swe_data['CaSR_v3.1_P_SWE_LAND'].min():.2f} to {swe_data['CaSR_v3.1_P_SWE_LAND'].max():.2f} mm")
    print(f"  Mean: {swe_data['CaSR_v3.1_P_SWE_LAND'].mean():.2f} mm")
    print(f"  Median: {swe_data['CaSR_v3.1_P_SWE_LAND'].median():.2f} mm")
    print(f"  Standard deviation: {swe_data['CaSR_v3.1_P_SWE_LAND'].std():.2f} mm")
    
    # Seasonal analysis
    print(f"\nSeasonal Analysis:")
    
    # PR24 by month
    pr24_monthly = pr24_data.groupby('month')['CaSR_v3.1_A_PR24_SFC'].agg(['count', 'mean', 'std'])
    print(f"\nPR24_SFC by Month:")
    for month in range(1, 13):
        if month in pr24_monthly.index:
            count = pr24_monthly.loc[month, 'count']
            mean_val = pr24_monthly.loc[month, 'mean']
            std_val = pr24_monthly.loc[month, 'std']
            print(f"  Month {month:2d}: {count:6,} records, mean={mean_val:.6f}, std={std_val:.6f}")
    
    # SWE by month
    swe_monthly = swe_data.groupby('month')['CaSR_v3.1_P_SWE_LAND'].agg(['count', 'mean', 'std'])
    print(f"\nSWE_LAND by Month:")
    for month in range(1, 13):
        if month in swe_monthly.index:
            count = swe_monthly.loc[month, 'count']
            mean_val = swe_monthly.loc[month, 'mean']
            std_val = swe_monthly.loc[month, 'std']
            print(f"  Month {month:2d}: {count:6,} records, mean={mean_val:.2f} mm, std={std_val:.2f} mm")
    
    print("\n" + "="*80)
    print("ELEVATION-CLIMATE RELATIONSHIPS")
    print("="*80)
    
    # Merge elevation with climate data
    pr24_with_elev = pr24_data[['point_id', 'elevation_mean', 'CaSR_v3.1_A_PR24_SFC']].copy()
    swe_with_elev = swe_data[['point_id', 'elevation_mean', 'CaSR_v3.1_P_SWE_LAND']].copy()
    
    # Create elevation bins
    elevation_bins = [0, 1000, 1500, 2000, 2500, 3500]
    elevation_labels = ['0-1000m', '1000-1500m', '1500-2000m', '2000-2500m', '2500m+']
    
    pr24_with_elev['elevation_bin'] = pd.cut(pr24_with_elev['elevation_mean'], 
                                           bins=elevation_bins, labels=elevation_labels, include_lowest=True)
    swe_with_elev['elevation_bin'] = pd.cut(swe_with_elev['elevation_mean'], 
                                          bins=elevation_bins, labels=elevation_labels, include_lowest=True)
    
    print(f"\nPrecipitation by Elevation Band:")
    pr24_by_elevation = pr24_with_elev.groupby('elevation_bin')['CaSR_v3.1_A_PR24_SFC'].agg(['count', 'mean', 'std'])
    for elev_bin in elevation_labels:
        if elev_bin in pr24_by_elevation.index:
            count = pr24_by_elevation.loc[elev_bin, 'count']
            mean_val = pr24_by_elevation.loc[elev_bin, 'mean']
            std_val = pr24_by_elevation.loc[elev_bin, 'std']
            print(f"  {elev_bin:12}: {count:6,} records, mean={mean_val:.6f}, std={std_val:.6f}")
    
    print(f"\nSWE by Elevation Band:")
    swe_by_elevation = swe_with_elev.groupby('elevation_bin')['CaSR_v3.1_P_SWE_LAND'].agg(['count', 'mean', 'std'])
    for elev_bin in elevation_labels:
        if elev_bin in swe_by_elevation.index:
            count = swe_by_elevation.loc[elev_bin, 'count']
            mean_val = swe_by_elevation.loc[elev_bin, 'mean']
            std_val = swe_by_elevation.loc[elev_bin, 'std']
            print(f"  {elev_bin:12}: {count:6,} records, mean={mean_val:.2f} mm, std={std_val:.2f} mm")
    
    print("\n" + "="*80)
    print("DATA QUALITY ASSESSMENT")
    print("="*80)
    
    # Check for data consistency
    print(f"\nData Consistency Checks:")
    
    # Check if elevation data is consistent across time for same points
    elevation_consistency = df1.groupby('point_id')[['elevation_min', 'elevation_max', 'elevation_mean', 'elevation_median']].nunique()
    inconsistent_points = elevation_consistency[(elevation_consistency > 1).any(axis=1)]
    
    if len(inconsistent_points) > 0:
        print(f"  WARNING: {len(inconsistent_points)} points have inconsistent elevation data across time")
    else:
        print(f"  ✓ Elevation data is consistent across time for all points")
    
    # Check for temporal gaps
    print(f"\nTemporal Coverage:")
    total_expected_records = df1['point_id'].nunique() * len(df1['time'].unique())
    actual_records = len(df1)
    print(f"  Expected records (points × times): {total_expected_records:,}")
    print(f"  Actual records: {actual_records:,}")
    print(f"  Coverage: {actual_records/total_expected_records*100:.2f}%")
    
    # Data availability summary
    print(f"\nData Availability Summary:")
    print(f"  PR24_SFC: {len(pr24_data):,} records with data ({len(pr24_data)/len(df1)*100:.2f}%)")
    print(f"  SWE_LAND: {len(swe_data):,} records with data ({len(swe_data)/len(df2)*100:.2f}%)")
    
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    
    print(f"""
1. DATASET STRUCTURE:
   - Both datasets contain {len(df1):,} records covering {df1['point_id'].nunique():,} spatial points
   - Time series spans from {df1['time'].min()} to {df1['time'].max()}
   - Spatial coverage: {df1['original_lon'].min():.2f}°W to {df1['original_lon'].max():.2f}°W, 
     {df1['original_lat'].min():.2f}°N to {df1['original_lat'].max():.2f}°N

2. ELEVATION CHARACTERISTICS:
   - Elevation ranges from {elevation_stats['elevation_min'].min():.0f}m to {elevation_stats['elevation_max'].max():.0f}m
   - Mean elevation across study area: {elevation_stats['elevation_mean'].mean():.0f}m
   - Topographically diverse region with significant elevation gradients

3. CLIMATE DATA AVAILABILITY:
   - Only {len(pr24_data)/len(df1)*100:.1f}% of records have precipitation data
   - Only {len(swe_data)/len(df2)*100:.1f}% of records have SWE data
   - This suggests data is available for specific time periods or conditions

4. PRECIPITATION PATTERNS:
   - PR24_SFC values are very small (mean: {pr24_data['CaSR_v3.1_A_PR24_SFC'].mean():.6f})
   - This appears to be precipitation rate data, possibly in mm/hour or similar units

5. SNOW WATER EQUIVALENT:
   - SWE values range from 0 to {swe_data['CaSR_v3.1_P_SWE_LAND'].max():.1f} mm
   - Mean SWE: {swe_data['CaSR_v3.1_P_SWE_LAND'].mean():.1f} mm
   - Shows significant seasonal and spatial variation

6. ELEVATION-CLIMATE RELATIONSHIPS:
   - Higher elevations tend to have different precipitation and SWE patterns
   - This data is suitable for studying orographic effects on precipitation and snow accumulation
    """)

if __name__ == "__main__":
    detailed_analysis()
