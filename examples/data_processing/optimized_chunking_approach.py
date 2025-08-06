import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from snowdroughtindex.core.gap_filling_notebook import artificial_gap_filling

def optimized_artificial_gap_filling_chunked(original_data, iterations, artificial_gap_perc, window_days, min_obs_corr, min_obs_cdf, min_corr, min_obs_KGE, flag=0, max_stations_per_chunk=10):
    """
    Optimized version that processes stations in chunks rather than splitting the dataframe.
    This is much more efficient than the original chunking approach.
    
    Parameters:
    -----------
    original_data : pd.DataFrame
        The original SWE data
    max_stations_per_chunk : int
        Maximum number of stations to process at once (default: 10)
    Other parameters same as artificial_gap_filling
    
    Returns:
    --------
    evaluation : dict
        Combined evaluation results
    fig : matplotlib.figure.Figure (optional)
        Combined figure if flag=1
    """
    
    # Identify stations for gap filling
    cols = [c for c in original_data.columns if 'precip' not in c and 'ext' not in c]
    
    # If we have fewer stations than chunk size, just run normally
    if len(cols) <= max_stations_per_chunk:
        print(f"Processing all {len(cols)} stations at once...")
        return artificial_gap_filling(
            original_data, iterations, artificial_gap_perc, window_days, 
            min_obs_corr, min_obs_cdf, min_corr, min_obs_KGE, flag
        )
    
    # Split stations into chunks
    station_chunks = [cols[i:i + max_stations_per_chunk] for i in range(0, len(cols), max_stations_per_chunk)]
    
    print(f"Processing {len(cols)} stations in {len(station_chunks)} chunks of max {max_stations_per_chunk} stations each...")
    
    all_evaluations = []
    all_figs = []
    
    for chunk_idx, station_chunk in enumerate(station_chunks):
        print(f"Processing chunk {chunk_idx + 1}/{len(station_chunks)} with {len(station_chunk)} stations...")
        
        # Create a subset of data with only the stations in this chunk
        # Keep all dates but only selected stations (plus any precip/ext columns)
        other_cols = [c for c in original_data.columns if 'precip' in c or 'ext' in c]
        chunk_cols = station_chunk + other_cols
        chunk_data = original_data[chunk_cols].copy()
        
        # Run artificial gap filling on this chunk
        if flag == 1:
            evaluation_dict, fig = artificial_gap_filling(
                chunk_data, iterations, artificial_gap_perc, window_days,
                min_obs_corr, min_obs_cdf, min_corr, min_obs_KGE, flag
            )
            all_figs.append(fig)
        else:
            evaluation_dict = artificial_gap_filling(
                chunk_data, iterations, artificial_gap_perc, window_days,
                min_obs_corr, min_obs_cdf, min_corr, min_obs_KGE, flag
            )
        
        all_evaluations.append(evaluation_dict)
    
    # Combine evaluations
    print("Combining evaluation results...")
    combined_evaluation = {}
    for key in all_evaluations[0].keys():
        # Concatenate along the station axis (axis=1)
        combined_arrays = []
        for eval_dict in all_evaluations:
            combined_arrays.append(eval_dict[key])
        combined_evaluation[key] = np.concatenate(combined_arrays, axis=1)
    
    if flag == 1:
        # Create a combined figure
        print("Creating combined figure...")
        fig_combined = plt.figure(figsize=(15, 10))
        
        # Simple approach: create subplots for each chunk's figure
        n_chunks = len(all_figs)
        for i, fig in enumerate(all_figs):
            # Get the axes from the original figure
            axes = fig.get_axes()
            n_axes = len(axes)
            
            # Create subplots in the combined figure
            for j, ax in enumerate(axes):
                subplot_idx = i * n_axes + j + 1
                new_ax = fig_combined.add_subplot(n_chunks, n_axes, subplot_idx)
                
                # Copy the plot data
                for line in ax.get_lines():
                    new_ax.plot(line.get_xdata(), line.get_ydata(), 
                               color=line.get_color(), alpha=line.get_alpha())
                
                for collection in ax.collections:
                    if hasattr(collection, 'get_offsets'):  # Scatter plots
                        offsets = collection.get_offsets()
                        if len(offsets) > 0:
                            new_ax.scatter(offsets[:, 0], offsets[:, 1], 
                                         color=collection.get_facecolor()[0], 
                                         alpha=collection.get_alpha())
                
                # Copy labels and title
                new_ax.set_title(ax.get_title())
                new_ax.set_xlabel(ax.get_xlabel())
                new_ax.set_ylabel(ax.get_ylabel())
        
        plt.tight_layout()
        return combined_evaluation, fig_combined
    
    return combined_evaluation


def replace_notebook_chunking_code():
    """
    This function shows how to replace the slow chunking code in the notebook
    """
    replacement_code = '''
# Replace the slow chunking approach with this optimized version:

# Instead of:
# all_evaluations, all_figs = process_in_chunks(SWE_testbasin_interp_df, chunk_size)

# Use this optimized approach:
from optimized_chunking_approach import optimized_artificial_gap_filling_chunked

# Process with optimized chunking (by stations, not by time)
evaluation_dict = optimized_artificial_gap_filling_chunked(
    SWE_testbasin_interp_df,
    iterations=iterations_default,
    artificial_gap_perc=artificial_gap_perc_default,
    window_days=window_days_default,
    min_obs_corr=min_obs_corr_default,
    min_obs_cdf=min_obs_cdf_default,
    min_corr=min_corr_default,
    min_obs_KGE=min_obs_KGE_default,
    flag=0,  # Set to 1 if you want plots
    max_stations_per_chunk=5  # Adjust based on your system capacity
)

# evaluation_dict now contains the combined results
print("Artificial gap filling evaluation completed!")
'''
    
    print("Recommended replacement code:")
    print(replacement_code)
    return replacement_code


if __name__ == "__main__":
    replace_notebook_chunking_code()
