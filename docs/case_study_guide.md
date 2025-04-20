# Case Study Guide

This guide provides a step-by-step walkthrough of how to use the Snow Drought Index package for a case study analysis. We'll use the Bow River at Banff basin as an example.

## Overview

The case study demonstrates how to:

1. Load and preprocess SWE and precipitation data
2. Perform gap filling of SWE data
3. Calculate the Standardized Snow Water Equivalent Index (SSWEI)
4. Classify snow drought conditions
5. Analyze snow drought conditions at different elevation bands

## Data Requirements

For this case study, you'll need:

- SWE observations data (e.g., from CANSWE dataset)
- Precipitation data (e.g., from RDRS or CAPA)
- Basin shapefile for the area of interest

## Step 1: Load Data

First, load the required data:

```python
import datetime
import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy.integrate import trapz
from scipy.stats import norm
from shapely.geometry import Point

# Import functions from the package
from notebooks.functions import extract_stations_in_basin, stations_basin_map, qm_gap_filling, artificial_gap_filling

# Load SWE data
CANSWE = xr.open_dataset('path/to/CANSWE-CanEEN_1928-2023_v6_updated.nc')
CANSWE_df = CANSWE.to_dataframe()

# Load precipitation data
P = xr.open_dataset('path/to/precipitation_data.nc')
P_df = P.to_dataframe()

# Load basin shapefile
Bow_at_Banff_shapefile = gpd.read_file('path/to/basins_shapefile.shp')
Bow_shapefile = Bow_at_Banff_shapefile[Bow_at_Banff_shapefile["Station_Na"]=="BOW RIVER AT BANFF"]
```

## Step 2: Extract Stations Within the Basin

Extract SWE and precipitation stations within the basin:

```python
# Reorganize SWE data
SWE_stations_ds = CANSWE.assign_coords({'lon':CANSWE.lon, 'lat':CANSWE.lat, 'station_name':CANSWE.station_name, 'elevation':CANSWE.elevation}).snw
SWE_stations_ds = SWE_stations_ds.to_dataset()

# Extract unique station coordinates
unique_stations = CANSWE_df.reset_index().drop_duplicates(subset='station_id')[['station_id', 'lon', 'lat']]
data = {'station_id': unique_stations['station_id'].values, 
        'lon': unique_stations['lon'].values, 
        'lat': unique_stations['lat'].values} 
df = pd.DataFrame(data)
geometry = [Point(xy) for xy in zip(df['lon'], df['lat'])]
crs = "EPSG:4326"
SWE_stations_gdf = gpd.GeoDataFrame(df, crs=crs, geometry=geometry)

# Convert SWE data to DataFrame
SWE_testbasin = SWE_stations_ds.to_dataframe().drop(columns=['lon','lat','station_name']).unstack()['snw'].T
SWE_testbasin['date'] = SWE_testbasin.index.normalize()
SWE_testbasin = SWE_testbasin.set_index('date')
SWE_testbasin = SWE_testbasin.dropna(axis=0, how='all')
SWE_testbasin = SWE_testbasin.loc['1980-01-01':'2023-07-31']

# Reorganize precipitation data
P_stations_ds = P.assign_coords({'lon':P.lon, 'lat':P.lat, 'station_name':P.coordinate_id}).daily_precipitation
P_stations_ds = P_stations_ds.to_dataset()

# Convert precipitation data to GeoDataFrame
data = {'lon': P['lon'].values.flatten(), 
        'lat': P['lat'].values.flatten()} 
P_df = pd.DataFrame(data)
geometry = [Point(xy) for xy in zip(P_df['lon'], P_df['lat'])]
crs = "EPSG:4326"
P_gdf = gpd.GeoDataFrame(P_df, crs=crs, geometry=geometry)

# Plot stations on map
fig, ax = plt.subplots(figsize=(10, 10))
Bow_shapefile.plot(ax=ax, color='lightgrey')
SWE_stations_gdf.plot(ax=ax, color='red', markersize=5)
P_gdf.plot(ax=ax, color='orange', markersize=5)
plt.legend(['CANSWE stations', 'Precipitation stations'])
plt.savefig('SWE-p_stations.png')
plt.show()
```

## Step 3: Prepare Precipitation Data

Prepare the precipitation data for gap filling:

```python
# Rearrange precipitation data
P_df = P_df[['coordinate_id', 'time', 'daily_precipitation']]
P_df.set_index(['coordinate_id', 'time'], inplace=True)
P_df = P_df.unstack(level=0)
P_df.columns = P_df.columns.droplevel()
P_df = P_df.reset_index()
P_df['date'] = P_df['time'].dt.normalize()
P_df = P_df.set_index('date')
P_df = P_df.drop(columns=['time'])

# Calculate cumulative precipitation
month_start_water_year, day_start_water_year = 10, 1  # Water year starts on October 1
year = []
for i in P_df.index:
    if (i.month == month_start_water_year and i.day >= day_start_water_year) or (i.month > month_start_water_year):
        year.append(i.year + 1)
    else:
        year.append(i.year)
P_df['water_year'] = year

# Calculate cumulative precipitation for each water year
elem = -1
for y in list(set(P_df['water_year'])):
    elem += 1
    P_df_water_year = P_df[P_df['water_year'] == y]
    P_df_water_year_cumul = P_df_water_year.cumsum().drop(['water_year'], axis=1)
    if elem == 0:
        P_df_cumul_testbasin = P_df_water_year_cumul
    else:
        P_df_cumul_testbasin = pd.concat([P_df_cumul_testbasin, P_df_water_year_cumul])

# Rename columns
P_df_cumul_testbasin.columns = ['precip_' + str(i) for i in range(1, len(P_df_cumul_testbasin.columns) + 1)]

# Combine SWE and precipitation data
SWE_P_testbasin = SWE_testbasin.merge(P_df_cumul_testbasin, left_index=True, right_index=True, how='outer')
SWE_P_testbasin = SWE_P_testbasin.loc['1980-01-01':'2023-07-31']
```

## Step 4: Perform Gap Filling

Perform gap filling of SWE data:

```python
# Set parameters
window_days = 7
min_obs_corr = 3
min_obs_cdf = 10
min_corr = 0.6
max_gap_days = 15

# Linear interpolation
SWE_obs_basin_interp_da = SWE_stations_ds.snw.interpolate_na(method='linear', dim='time', max_gap=datetime.timedelta(days=max_gap_days))
SWE_obs_basin_interp_df = SWE_obs_basin_interp_da.to_dataframe().drop(columns=['lon','lat','station_name']).unstack()['snw'].T
SWE_obs_basin_interp_df['date'] =  SWE_obs_basin_interp_df.index.normalize()
SWE_obs_basin_interp_df = SWE_obs_basin_interp_df.set_index('date')

# Save flags for linear interpolation
flags_interp_basin_da = SWE_obs_basin_interp_da.copy().fillna(-999)
original_da = SWE_stations_ds.snw.copy().fillna(-999)
flags_interp_basin_da = xr.where(flags_interp_basin_da==original_da, 0, 1)
flags_interp_basin_df = flags_interp_basin_da.to_dataframe().drop(columns=['lon','lat','station_name']).unstack()['snw'].T
flags_interp_basin_df['date'] =  flags_interp_basin_df.index.normalize()
flags_interp_basin_df = flags_interp_basin_df.set_index('date')

# Perform gap filling
SWE_obs_basin_gapfilled_df, flags_gapfill_basin_df, donor_stations_gapfill_basin_df = qm_gap_filling(
    SWE_P_testbasin.copy(), 
    window_days=window_days, 
    min_obs_corr=min_obs_corr, 
    min_obs_cdf=min_obs_cdf, 
    min_corr=min_corr
)

# Combine flags
flags_basin_df = flags_interp_basin_df + flags_gapfill_basin_df

# Save gap filled data to xarray Dataset
SWE_gapfill_basin_da = xr.DataArray(
    data=SWE_obs_basin_gapfilled_df.values, 
    coords=dict(time=SWE_obs_basin_gapfilled_df.index.values, station_id=SWE_obs_basin_gapfilled_df.columns.values), 
    dims=['time','station_id'], 
    name='SWE', 
    attrs={'long_name':'Surface snow water equivalent','units':'kg m**-2'}
)
flags_basin_da = xr.DataArray(
    data=flags_basin_df.values, 
    coords=dict(time=flags_basin_df.index.values, station_id=flags_basin_df.columns.values), 
    dims=['time','station_id'], 
    name='flag', 
    attrs={'description':'observations = 0; estimates = 1'}
)
donor_stations_gapfill_basin_da = xr.DataArray(
    data=donor_stations_gapfill_basin_df.values, 
    coords=dict(time=donor_stations_gapfill_basin_df.index.values, station_id=donor_stations_gapfill_basin_df.columns.values), 
    dims=['time','station_id'], 
    name='donor_stations', 
    attrs={'description':'station_id of donor stations used for gap filling'}
)
SWE_obs_basin_gapfilled_ds = xr.merge([SWE_gapfill_basin_da, flags_basin_da, donor_stations_gapfill_basin_da])
lats = SWE_stations_ds.lat.sel(station_id=SWE_obs_basin_gapfilled_df.columns.values).isel(time=0).values
lons = SWE_stations_ds.lon.sel(station_id=SWE_obs_basin_gapfilled_df.columns.values).isel(time=0).values
names = SWE_stations_ds.station_name.sel(station_id=SWE_obs_basin_gapfilled_df.columns.values).isel(time=0).values
SWE_obs_basin_gapfilled_ds = SWE_obs_basin_gapfilled_ds.assign_coords({
    'lat':('station_id',lats),
    'lon':('station_id',lons),
    'station_name':('station_id',names)
})

# Save gap filled data
SWE_obs_basin_gapfilled_ds.to_netcdf('SWE_gapfilled_for_basin.nc')
```

## Step 5: Evaluate Gap Filling Performance

Evaluate the gap filling performance using artificial gap filling:

```python
# Set parameters
iterations = 1
artificial_gap_perc = 100
min_obs_KGE = 3

# Perform artificial gap filling evaluation
evaluation_artificial_gapfill_testbasin_dict, fig = artificial_gap_filling(
    SWE_P_testbasin.copy(), 
    iterations=iterations, 
    artificial_gap_perc=artificial_gap_perc, 
    window_days=window_days, 
    min_obs_corr=min_obs_corr, 
    min_obs_cdf=min_obs_cdf, 
    min_corr=min_corr, 
    min_obs_KGE=min_obs_KGE, 
    flag=1
)
plt.savefig('SWE_artificial_gapfilling_in_basin.png')
plt.close(fig)

# Plot evaluation results
fig = plots_artificial_gap_evaluation(evaluation_artificial_gapfill_testbasin_dict)
plt.savefig('SWE_artificial_gapfilling_eval_in_basin.png')
plt.close(fig)
```

## Step 6: Prepare Data for SSWEI Calculation

Prepare the gap-filled data for SSWEI calculation:

```python
# Load gap filled data
SWE_gap_filled = xr.open_dataset('SWE_gapfilled_for_basin.nc')
SWE_gap_filled_df = SWE_gap_filled.to_dataframe()
SWE_gap_filled_df = SWE_gap_filled_df.loc['1980-01-01':'2023-07-31']

# Add elevation information
elevation = CANSWE_df.reset_index().drop_duplicates(subset='station_id')[['station_id', 'elevation']]
elevation = elevation.set_index('station_id')
SWE_gap_filled_df = SWE_gap_filled_df.join(elevation, on='station_id', rsuffix='_original')
SWE_gap_filled_df = SWE_gap_filled_df.fillna(0)

# Calculate daily mean SWE
SWE_gap_filled_daily = SWE_gap_filled_df.groupby('time').mean()
SWE_gap_filled_daily = SWE_gap_filled_daily.drop(columns=['elevation','flag','lat','lon'])
```

## Step 7: Divide Stations by Elevation

Divide stations into elevation bands for analysis:

```python
# Divide stations by elevation
bins = [1300, 2000, 2400]
labels = ['low', 'high']
SWE_gap_filled_df['elevation_category'] = pd.cut(SWE_gap_filled_df['elevation'], bins=bins, labels=labels, right=False)

# Create separate dataframes for each elevation category
SWE_gap_filled_df = SWE_gap_filled_df.reset_index()
SWE_gap_filled_df['year'] = SWE_gap_filled_df['time'].dt.year
SWE_gap_filled_df['month'] = SWE_gap_filled_df['time'].dt.month
SWE_gap_filled_df['day'] = SWE_gap_filled_df['time'].dt.day
SWE_gap_filled_df['date'] = SWE_gap_filled_df['time'].dt.normalize()
SWE_gap_filled_df = SWE_gap_filled_df.set_index('date')

# Create separate dataframes for each elevation category
SWE_low_elev = SWE_gap_filled_df[SWE_gap_filled_df['elevation_category'] == 'low']
SWE_high_elev = SWE_gap_filled_df[SWE_gap_filled_df['elevation_category'] == 'high']

# Calculate daily mean SWE for each elevation category
daily_mean_low = SWE_low_elev.groupby('time')['SWE'].mean().reset_index()
daily_mean_high = SWE_high_elev.groupby('time')['SWE'].mean().reset_index()
daily_mean_low.columns = ['date', 'mean_SWE']
daily_mean_high.columns = ['date', 'mean_SWE']

# Plot daily mean SWE for each elevation category
plt.figure(figsize=(10,6))
plt.plot(daily_mean_low['date'], daily_mean_low['mean_SWE'], label='1300 m - 2000 m', color='blue')
plt.plot(daily_mean_high['date'], daily_mean_high['mean_SWE'], label='2000 m - 2400 m', color='red')
plt.xlabel('Date')
plt.ylabel('Mean SWE (mm)')
plt.title('Daily Mean SWE by Elevation')
plt.legend()
plt.savefig('daily_mean_values.png')
plt.close()
```

## Step 8: Define Snow Seasons

Define snow seasons for SSWEI calculation:

```python
# Set season parameters
start_month, start_day = 11, 1  # Start in November
end_month, end_day = 5, 1       # End in May

# Find the first date with 15 mm SWE each year for low elevation
daily_mean_low['season_year'] = daily_mean_low['date'].apply(lambda x: x.year if x.month >= start_month else x.year - 1)
season_starts_low = daily_mean_low[daily_mean_low['mean_SWE'] >= 9].groupby('season_year')['date'].min()

# Filter seasons for low elevation
filtered_seasons_low = []
for year, start_date in season_starts_low.items():
    if start_date.month < start_month:
        continue  # Skip incomplete seasons at the beginning
    end_date = pd.Timestamp(year + 1, end_month, end_day)
    season_data_low = daily_mean_low[(daily_mean_low['date'] >= start_date) & (daily_mean_low['date'] <= end_date)]
    if not season_data_low.empty and season_data_low['date'].max() >= end_date - pd.Timedelta(days=1):
        filtered_seasons_low.append(season_data_low)

# Combine all complete seasons for low elevation
if filtered_seasons_low:
    season_data_low = pd.concat(filtered_seasons_low, ignore_index=True)
    season_data_low['Year'] = season_data_low['date'].dt.year
    season_data_low['Month-Day'] = season_data_low['date'].dt.strftime('%m-%d')
    season_data_low['Year_Month'] = season_data_low['date'].dt.strftime('%Y-%m')

# Find the first date with 15 mm SWE each year for high elevation
daily_mean_high['season_year'] = daily_mean_high['date'].apply(lambda x: x.year if x.month >= start_month else x.year - 1)
season_starts_high = daily_mean_high[daily_mean_high['mean_SWE'] >= 15].groupby('season_year')['date'].min()

# Filter seasons for high elevation
filtered_seasons_high = []
for year, start_date in season_starts_high.items():
    if start_date.month < start_month:
        continue  # Skip incomplete seasons at the beginning
    end_date = pd.Timestamp(year + 1, end_month, end_day)
    season_data_high = daily_mean_high[(daily_mean_high['date'] >= start_date) & (daily_mean_high['date'] <= end_date)]
    if not season_data_high.empty and season_data_high['date'].max() >= end_date - pd.Timedelta(days=1):
        filtered_seasons_high.append(season_data_high)

# Combine all complete seasons for high elevation
if filtered_seasons_high:
    season_data_high = pd.concat(filtered_seasons_high, ignore_index=True)
    season_data_high['Year'] = season_data_high['date'].dt.year
    season_data_high['Month-Day'] = season_data_high['date'].dt.strftime('%m-%d')
    season_data_high['Year_Month'] = season_data_high['date'].dt.strftime('%Y-%m')
```

## Step 9: Handle Zero Values

Handle zero values in the SWE data:

```python
# Define function to perturb zeros
def perturb_zeros(swe_column):
    """Perturbs zero values with small positive values."""
    swe_array = swe_column.to_numpy()
    nonzero_min = swe_array[swe_array > 0].min()  # Find the smallest nonzero value
    
    # Generate perturbations for zero values
    perturbation = np.random.uniform(0, nonzero_min / 2, size=swe_column[swe_column == 0].shape)
    
    # Replace zeros with perturbation
    swe_column[swe_column == 0] = perturbation
    
    return swe_column

# Apply to mean_SWE column for low elevation
season_data_low['mean_SWE'] = perturb_zeros(season_data_low['mean_SWE'].copy())

# Apply to mean_SWE column for high elevation
season_data_high['mean_SWE'] = perturb_zeros(season_data_high['mean_SWE'].copy())
```

## Step 10: Calculate Seasonal SWE Integration

Calculate the seasonal SWE integration:

```python
# Define function to integrate SWE over the season
def integrate_season(group):
    """Integrates SWE values from November 1st to May 1st."""
    # Ensure dates are sorted
    group = group.sort_values(by='date')
    # Convert dates to numerical days since start of the season
    days_since_start = (group['date'] - group['date'].min()).dt.days
    # Integrate SWE over the period
    total_swe_integration = trapz(group['mean_SWE'], days_since_start)
    return pd.Series({'total_SWE_integration': total_swe_integration})

# Calculate integration for low elevation
Integrated_data_low = season_data_low.groupby('season_year').apply(integrate_season).reset_index()

# Calculate integration for high elevation
Integrated_data_high = season_data_high.groupby('season_year').apply(integrate_season).reset_index()

# Calculate monthly integration for low elevation
Integrated_data_monthly_low = season_data_low.groupby('Year_Month').apply(integrate_season).reset_index()
Integrated_data_monthly_low['season_year'] = Integrated_data_monthly_low['Year_Month'].apply(
    lambda x: int(x.split('-')[0]) if int(x.split('-')[1]) >= start_month else int(x.split('-')[0]) - 1
)
Integrated_data_season_low = Integrated_data_monthly_low.groupby('season_year').sum().reset_index()

# Calculate monthly integration for high elevation
Integrated_data_monthly_high = season_data_high.groupby('Year_Month').apply(integrate_season).reset_index()
Integrated_data_monthly_high['season_year'] = Integrated_data_monthly_high['Year_Month'].apply(
    lambda x: int(x.split('-')[0]) if int(x.split('-')[1]) >= start_month else int(x.split('-')[0]) - 1
)
Integrated_data_season_high = Integrated_data_monthly_high.groupby('season_year').sum().reset_index()
```

## Step 11: Calculate SSWEI

Calculate the SSWEI for each elevation band:

```python
# Define function to calculate Gringorten probabilities
def gringorten_probabilities(values):
    """Compute Gringorten plotting position probabilities."""
    sorted_values = np.sort(values)
    ranks = np.argsort(np.argsort(values)) + 1  # Rank from smallest to largest
    n = len(values)
    probabilities = (ranks - 0.44) / (n + 0.12)
    return probabilities

# Calculate Gringorten probabilities for low elevation
Integrated_data_season_low['Gringorten_probabilities'] = gringorten_probabilities(Integrated_data_season_low['total_SWE_integration'])

# Calculate Gringorten probabilities for high elevation
Integrated_data_season_high['Gringorten_probabilities'] = gringorten_probabilities(Integrated_data_season_high['total_SWE_integration'])

# Define function to compute SWEI
def compute_swei(probabilities):
    """Transform probabilities to SWEI using the inverse normal distribution."""
    return norm.ppf(probabilities)

# Calculate SWEI for low elevation
Integrated_data_season_low['SWEI'] = compute_swei(Integrated_data_season_low['Gringorten_probabilities'])

# Calculate SWEI for high elevation
Integrated_data_season_high['SWEI'] = compute_swei(Integrated_data_season_high['Gringorten_probabilities'])
```

## Step 12: Classify Drought Conditions

Classify drought conditions based on SWEI values:

```python
# Define function to classify drought conditions
def classify_drought(swei):
    """Classify drought conditions based on SWEI values."""
    if swei <= -2.0:
        return "Exceptional Drought"
    elif -2.0 < swei <= -1.5:
        return "Extreme Drought"
    elif -1.5 < swei <= -1.0:
        return "Severe Drought"
    elif -1.0 < swei <= -0.5:
        return "Moderate Drought"
    elif -0.5 < swei <= 0.5:
        return "Near Normal"
    elif 0.5 < swei <= 1.0:
        return "Abnormally Wet"
    elif 1.0 < swei <= 1.5:
        return "Moderately Wet"
    elif 1.5 < swei <= 2.0:
        return "Very Wet"
    else:
        return "Extremely Wet"

# Classify drought conditions for low elevation
Integrated_data_season_low['Drought_Classification'] = Integrated_data_season_low['SWEI'].apply(classify_drought)

# Classify drought conditions for high elevation
Integrated_data_season_high['Drought_Classification'] = Integrated_data_season_high['SWEI'].apply(classify_drought)

# Display results
print("Low Elevation Results:\n", Integrated_data_season_low[['season_year', 'Gringorten_probabilities', 'SWEI', 'Drought_Classification']])
print("High Elevation Results:\n", Integrated_data_season_high[['season_year', 'Gringorten_probabilities', 'SWEI', 'Drought_Classification']])
```

## Step 13: Visualize Results

Visualize the SSWEI results:

```python
# Extract necessary columns for plotting
plot_data_low = Integrated_data_season_low[['season_year', 'SWEI', 'Drought_Classification']]
plot_data_high = Integrated_data_season_high[['season_year', 'SWEI', 'Drought_Classification']]

# Sort by season_year for better plotting
plot_data_low = plot_data_low.sort_values(by='season_year')
plot_data_high = plot_data_high.sort_values(by='season_year')

# Create plot
plt.figure(figsize=(15, 6))
plt.plot(plot_data_low['season_year'], plot_data_low['SWEI'], marker='o', label='SWEI for 1300-2000 m', color='black')
plt.plot(plot_data_high['season_year'], plot_data_high['SWEI'], marker='o', label='SWEI for 2000-2400 m', color='red')

# Add thresholds for drought classifications
plt.axhline(-2.0, color='r', linestyle='--', label='Exceptional Drought Threshold')
plt.axhline(-1.5, color='orange', linestyle='--', label='Extreme Drought Threshold')
plt.axhline(-1.0, color='yellow', linestyle='--', label='Severe Drought Threshold')
plt.axhline(-0.5, color='gray', linestyle='--', label='Near Normal Threshold')
plt.axhline(0.5, color='pink', linestyle='--', label='Abnormally Wet Threshold')
plt.axhline(1.0, color='violet', linestyle='--', label='Moderately Wet Threshold')
plt.axhline(1.5, color='purple', linestyle='--', label='Very Wet Threshold')
plt.axhline(2.0, color='blue', linestyle='--', label='Extremely Wet Threshold')

# Customize the plot
plt.title('SWEI Trends by Season Year')
plt.xlabel('Season Year')
plt.ylabel('Standardized SWEI')
plt.xticks(rotation=45)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.grid()
plt.tight_layout()
plt.savefig('SWEI_trends.png')
plt.close()
```

## Conclusion

This case study demonstrates how to use the Snow Drought Index package to analyze snow drought conditions in the Bow River at Banff basin. The analysis includes:

1. Loading and preprocessing SWE and precipitation data
2. Performing gap filling of SWE data
3. Calculating the SSWEI for different elevation bands
4. Classifying snow drought conditions
5. Visualizing the results

The results show the temporal evolution of snow drought conditions in the basin, with differences between low and high elevation bands. This information can be used to understand the spatial and temporal variability of snow drought conditions and their potential impacts on water resources.
