DEM and Shapefile Elevation Analysis
==================================

This guide explains how to join Digital Elevation Model (DEM) data with shapefiles to extract elevation statistics for geographic areas of interest.

Overview
--------

The DEM and Shapefile Elevation Analysis workflow allows you to:

1. Merge multiple DEM raster files into a single raster
2. Extract elevation statistics for areas defined in a shapefile
3. Visualize elevation data across your study area
4. Classify areas based on elevation ranges

This workflow is particularly useful for:

- Characterizing the topography of your study area
- Preparing elevation data for snow drought analysis
- Creating elevation-based zones for further analysis

Prerequisites
------------

- Digital Elevation Model (DEM) data in GeoTIFF format
- Shapefile defining your area(s) of interest
- Python packages: geopandas, rasterio, rasterstats, numpy, matplotlib

Workflow Steps
-------------

1. Load Shapefile
~~~~~~~~~~~~~~~~

First, load the shapefile that defines your study area:

.. code-block:: python

    import geopandas as gpd
    
    shapefile_path = "path/to/your/shapefile.shp"
    gdf = gpd.read_file(shapefile_path)

2. Merge DEM Files
~~~~~~~~~~~~~~~~~

If your study area spans multiple DEM files, merge them into a single raster:

.. code-block:: python

    import rasterio
    import glob
    from rasterio.merge import merge
    
    # Get all TIF files in the directory
    tif_files = glob.glob("path/to/dem/files/*.tif")
    
    # Open each file
    src_files_to_mosaic = [rasterio.open(fp) for fp in tif_files]
    
    # Merge the files
    mosaic, out_transform = merge(src_files_to_mosaic)
    
    # Update metadata
    mosaic_meta = src_files_to_mosaic[0].meta.copy()
    mosaic_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_transform
    })
    
    # Save the merged file
    merged_tif_path = "merged_dem.tif"
    with rasterio.open(merged_tif_path, "w", **mosaic_meta) as dest:
        dest.write(mosaic)

3. Extract Zonal Statistics
~~~~~~~~~~~~~~~~~~~~~~~~~

Calculate elevation statistics for each polygon in your shapefile:

.. code-block:: python

    from rasterstats import zonal_stats
    
    stats = zonal_stats(
        shapefile_path,
        merged_tif_path,
        stats=["min", "max", "mean", "std", "median", "count"],
        nodata=None,
        geojson_out=True
    )
    
    # Convert to GeoDataFrame
    stats_gdf = gpd.GeoDataFrame.from_features(stats)
    
    # Save results
    output_path = "path/to/output/directory"
    stats_gdf.to_file(os.path.join(output_path, "elevation_with_stats.shp"))
    stats_gdf.drop(columns="geometry").to_csv(os.path.join(output_path, "elevation_stats.csv"), index=False)

4. Visualize Elevation Data
~~~~~~~~~~~~~~~~~~~~~~~~~

Create maps to visualize the elevation data:

.. code-block:: python

    import matplotlib.pyplot as plt
    import numpy as np
    from rasterio.mask import mask
    
    # Plot mean elevation by polygon
    fig, ax = plt.subplots(figsize=(10, 8))
    stats_gdf.plot(column='mean', cmap='terrain', legend=True, edgecolor='black', ax=ax)
    ax.set_title("Mean Elevation per Polygon", fontsize=14)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "mean_elevation_plot.png"), dpi=300)
    
    # Clip and plot the DEM raster
    with rasterio.open(merged_tif_path) as src:
        shapes = [feature["geometry"] for feature in stats]
        clipped_dem, clipped_transform = mask(src, shapes=shapes, crop=True)
    
    nodata_value = mosaic_meta.get("nodata", 0)
    clipped_data = np.where(clipped_dem[0] == nodata_value, np.nan, clipped_dem[0])
    
    plt.figure(figsize=(12, 8))
    plt.imshow(clipped_data, cmap='terrain', vmin=0, vmax=3000)
    plt.colorbar(label="Elevation (m)")
    plt.title("Elevation Across Shapefile Area")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'clipped_elevation_map.png'), dpi=300)

5. Create Elevation Classes
~~~~~~~~~~~~~~~~~~~~~~~~~

Categorize your study area into elevation classes:

.. code-block:: python

    import pandas as pd
    
    # Define elevation bins
    elevation_bins = [0, 500, 1000, 1500, 2000, 2500, 3000]
    elevation_labels = [f"{elevation_bins[i]}_{elevation_bins[i+1]}m" for i in range(len(elevation_bins)-1)]
    
    # Classify polygons
    stats_gdf['elev_class'] = pd.cut(stats_gdf['mean'], bins=elevation_bins, 
                                     labels=elevation_labels, include_lowest=True)
    stats_gdf['elev_class'] = stats_gdf['elev_class'].astype(str)
    
    # Save separate shapefiles for each elevation class
    for label in elevation_labels:
        subset = stats_gdf[stats_gdf['elev_class'] == str(label)]
        if not subset.empty:
            out_path = os.path.join(output_path, f"elevation_{label}.shp")
            subset.to_file(out_path)

Complete Example
---------------

A complete example notebook is available in the package repository:

``notebooks/workflows/4_Join_DEM_and_shapefile_Elevation.ipynb``

This notebook demonstrates the full workflow with example data.

Integration with Snow Drought Analysis
-------------------------------------

The elevation data and statistics generated by this workflow can be used to:

1. **Stratify Analysis**: Analyze snow drought patterns across different elevation bands
2. **Improve Visualization**: Create more informative maps that include elevation context
3. **Refine Models**: Incorporate elevation as a predictor in snow drought models
4. **Target Interventions**: Identify elevation-specific management strategies

Next Steps
---------

After completing this workflow, you can:

- Use the elevation statistics in your snow drought analysis
- Combine elevation data with other environmental variables
- Analyze snow drought patterns across different elevation zones
- Create more detailed visualizations that incorporate elevation information
