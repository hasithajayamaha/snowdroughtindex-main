import nbformat

# Load the notebook
notebook_path = r'c:\Users\walimunige.rupasingh\OneDrive - University of Calgary\Documents\GitHub\snowdroughtindex-main\notebooks\workflows\0.2_gapfilling_data_preparation_workflow.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

# The cell is the 8th cell (index 7)
cell = nb.cells[7]

# New source
new_source = [
    "# Get station locations from canswe",
    "stations_df = pd.DataFrame({",
    "    'station_id': canswe_ds['station_id'].values,",
    "    'lat': canswe_ds['lat'].values,",
    "    'lon': canswe_ds['lon'].values,",
    "    'elevation': canswe_ds['elevation'].values",
    "})",
    "",
    "# Create Point geometries for each station",
    "stations_gdf = gpd.GeoDataFrame(",
    "    stations_df,",
    "    geometry=gpd.points_from_xy(stations_df['lon'], stations_df['lat']),",
    "    crs=basin_gdf.crs",
    ")",
    "",
    "# Find stations within any of the Bow basin polygons",
    "stations_in_basin = stations_gdf[stations_gdf.within(basin_gdf.unary_union)]",
    "",
    "# Select these stations from canswe",
    "bow_canswe = canswe_ds[canswe_ds['station_id'].isin(stations_in_basin['station_id'].values)]",
    "",
    "# Convert to DataFrame",
    "#bow_canswe_df = bow_canswe.to_dataframe().reset_index()",
    "",
    "# Save the extracted data to a nc file",
    "#bow_canswe.to_netcdf(output_data / 'bow_canswe.nc')",
    "",
    "display(stations_in_basin)"
]

cell.source = new_source

# Save the notebook
with open(notebook_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print("Notebook updated")