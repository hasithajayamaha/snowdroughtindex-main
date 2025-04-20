"""
DroughtAnalysis class for the Snow Drought Index package.

This module contains the DroughtAnalysis class, which encapsulates methods for analyzing
drought conditions, comparing across elevation bands, and performing temporal trend analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
from collections import defaultdict

from snowdroughtindex.core.dataset import SWEDataset
from snowdroughtindex.core.sswei_class import SSWEI
from snowdroughtindex.utils import visualization, statistics

class DroughtAnalysis:
    """
    A class for analyzing drought conditions, comparing across elevation bands,
    and performing temporal trend analysis.
    
    This class provides methods for analyzing drought conditions based on SSWEI values,
    comparing drought conditions across elevation bands, and performing temporal trend analysis.
    
    Attributes
    ----------
    datasets : dict
        Dictionary of SWEDataset objects, keyed by elevation band or region name.
    sswei_objects : dict
        Dictionary of SSWEI objects, keyed by elevation band or region name.
    """
    
    def __init__(self):
        """
        Initialize a DroughtAnalysis object.
        """
        self.datasets = {}
        self.sswei_objects = {}
        self.elevation_bands = []
        self.regions = []
        self.analysis_results = {}
    
    def add_dataset(self, name: str, dataset: SWEDataset, 
                   is_elevation_band: bool = True) -> 'DroughtAnalysis':
        """
        Add a SWEDataset to the analysis.
        
        Parameters
        ----------
        name : str
            Name of the dataset (e.g., elevation band or region name).
        dataset : SWEDataset
            SWEDataset object to add.
        is_elevation_band : bool, optional
            Whether the dataset represents an elevation band, by default True.
            
        Returns
        -------
        DroughtAnalysis
            The DroughtAnalysis object with the added dataset.
        """
        self.datasets[name] = dataset
        
        if is_elevation_band:
            if name not in self.elevation_bands:
                self.elevation_bands.append(name)
        else:
            if name not in self.regions:
                self.regions.append(name)
        
        return self
    
    def calculate_sswei(self, start_month: int, end_month: int, 
                       min_years: int = 10, distribution: str = 'gamma',
                       reference_period: Optional[Tuple[int, int]] = None) -> 'DroughtAnalysis':
        """
        Calculate SSWEI for all datasets.
        
        Parameters
        ----------
        start_month : int
            Starting month of the season (1-12).
        end_month : int
            Ending month of the season (1-12).
        min_years : int, optional
            Minimum number of years required for calculation, by default 10.
        distribution : str, optional
            Probability distribution to use, by default 'gamma'.
            Options: 'gamma', 'normal'.
        reference_period : tuple, optional
            Reference period (start_year, end_year) for standardization.
            If None, the entire period is used.
            
        Returns
        -------
        DroughtAnalysis
            The DroughtAnalysis object with calculated SSWEI values.
        """
        if not self.datasets:
            raise ValueError("No datasets have been added. Use add_dataset() first.")
        
        for name, dataset in self.datasets.items():
            sswei_obj = SSWEI(dataset)
            sswei_obj.calculate_sswei(
                start_month=start_month,
                end_month=end_month,
                min_years=min_years,
                distribution=distribution,
                reference_period=reference_period
            )
            self.sswei_objects[name] = sswei_obj
        
        return self
    
    def compare_elevation_bands(self) -> pd.DataFrame:
        """
        Compare drought conditions across elevation bands.
        
        Returns
        -------
        pandas.DataFrame
            DataFrame containing drought comparison across elevation bands.
        """
        if not self.elevation_bands:
            raise ValueError("No elevation bands have been added. Use add_dataset() with is_elevation_band=True.")
        
        if not self.sswei_objects:
            raise ValueError("SSWEI has not been calculated. Use calculate_sswei() first.")
        
        # Create a DataFrame to store the comparison results
        comparison_data = []
        
        for band in self.elevation_bands:
            if band not in self.sswei_objects:
                continue
                
            sswei_obj = self.sswei_objects[band]
            sswei_data = sswei_obj.sswei_data
            
            # Calculate drought statistics for this elevation band
            drought_years = sswei_data[sswei_data['SWEI'] < 0]
            drought_count = len(drought_years)
            severe_drought_count = len(sswei_data[sswei_data['SWEI'] < -1.5])
            mean_severity = drought_years['SWEI'].mean() if not drought_years.empty else 0
            
            # Calculate drought frequency
            total_years = len(sswei_data)
            drought_frequency = (drought_count / total_years) * 100 if total_years > 0 else 0
            
            # Add to comparison data
            comparison_data.append({
                'Elevation_Band': band,
                'Total_Years': total_years,
                'Drought_Count': drought_count,
                'Severe_Drought_Count': severe_drought_count,
                'Drought_Frequency': drought_frequency,
                'Mean_Severity': mean_severity
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        self.analysis_results['elevation_band_comparison'] = comparison_df
        
        return comparison_df
    
    def analyze_temporal_trends(self, window_size: int = 10) -> Dict[str, pd.DataFrame]:
        """
        Analyze temporal trends in drought conditions.
        
        Parameters
        ----------
        window_size : int, optional
            Size of the moving window in years, by default 10.
            
        Returns
        -------
        dict
            Dictionary of DataFrames containing temporal trend analysis for each dataset.
        """
        if not self.sswei_objects:
            raise ValueError("SSWEI has not been calculated. Use calculate_sswei() first.")
        
        trend_results = {}
        
        for name, sswei_obj in self.sswei_objects.items():
            trend_data = sswei_obj.analyze_drought_trends(window_size)
            trend_results[name] = trend_data
        
        self.analysis_results['temporal_trends'] = trend_results
        
        return trend_results
    
    def analyze_drought_characteristics(self) -> Dict[str, pd.DataFrame]:
        """
        Analyze drought characteristics for all datasets.
        
        Returns
        -------
        dict
            Dictionary of DataFrames containing drought characteristics for each dataset.
        """
        if not self.sswei_objects:
            raise ValueError("SSWEI has not been calculated. Use calculate_sswei() first.")
        
        characteristics_results = {}
        
        for name, sswei_obj in self.sswei_objects.items():
            drought_chars = sswei_obj.calculate_drought_characteristics()
            characteristics_results[name] = drought_chars
        
        self.analysis_results['drought_characteristics'] = characteristics_results
        
        return characteristics_results
    
    def analyze_drought_synchronicity(self) -> pd.DataFrame:
        """
        Analyze the synchronicity of drought events across elevation bands or regions.
        
        Returns
        -------
        pandas.DataFrame
            DataFrame containing drought synchronicity analysis.
        """
        if not self.sswei_objects or len(self.sswei_objects) < 2:
            raise ValueError("At least two datasets with calculated SSWEI are required for synchronicity analysis.")
        
        # Get all unique years across all datasets
        all_years = set()
        for sswei_obj in self.sswei_objects.values():
            all_years.update(sswei_obj.sswei_data['season_year'].values)
        
        all_years = sorted(all_years)
        
        # Create a DataFrame to track drought conditions for each year and dataset
        sync_data = pd.DataFrame(index=all_years)
        
        for name, sswei_obj in self.sswei_objects.items():
            # Create a series indicating drought conditions (True for drought, False otherwise)
            drought_series = pd.Series(False, index=all_years)
            
            # Get years with drought conditions (SWEI < 0)
            drought_years = sswei_obj.sswei_data[sswei_obj.sswei_data['SWEI'] < 0]['season_year'].values
            
            # Update the series
            drought_series[drought_series.index.isin(drought_years)] = True
            
            # Add to the DataFrame
            sync_data[name] = drought_series
        
        # Calculate synchronicity metrics
        sync_metrics = []
        
        # Calculate the number of datasets in drought for each year
        sync_data['Datasets_in_Drought'] = sync_data.sum(axis=1)
        
        # Calculate the percentage of datasets in drought for each year
        sync_data['Percent_in_Drought'] = (sync_data['Datasets_in_Drought'] / len(self.sswei_objects)) * 100
        
        # Calculate years with all datasets in drought
        all_drought_years = sync_data[sync_data['Datasets_in_Drought'] == len(self.sswei_objects)].index.tolist()
        
        # Calculate years with no datasets in drought
        no_drought_years = sync_data[sync_data['Datasets_in_Drought'] == 0].index.tolist()
        
        # Calculate pairwise agreement between datasets
        dataset_names = list(self.sswei_objects.keys())
        agreement_matrix = pd.DataFrame(index=dataset_names, columns=dataset_names)
        
        for i, name1 in enumerate(dataset_names):
            for j, name2 in enumerate(dataset_names):
                if i == j:
                    agreement_matrix.loc[name1, name2] = 1.0
                else:
                    # Calculate the percentage of years where both datasets agree on drought/non-drought
                    agreement = (sync_data[name1] == sync_data[name2]).mean()
                    agreement_matrix.loc[name1, name2] = agreement
        
        self.analysis_results['drought_synchronicity'] = {
            'sync_data': sync_data,
            'all_drought_years': all_drought_years,
            'no_drought_years': no_drought_years,
            'agreement_matrix': agreement_matrix
        }
        
        return sync_data
    
    def analyze_elevation_sensitivity(self) -> pd.DataFrame:
        """
        Analyze the sensitivity of drought conditions to elevation.
        
        Returns
        -------
        pandas.DataFrame
            DataFrame containing elevation sensitivity analysis.
        """
        if not self.elevation_bands or len(self.elevation_bands) < 2:
            raise ValueError("At least two elevation bands are required for elevation sensitivity analysis.")
        
        if not self.sswei_objects:
            raise ValueError("SSWEI has not been calculated. Use calculate_sswei() first.")
        
        # Extract elevation information from band names (assuming format like "1000-1500m")
        elevation_data = []
        
        for band in self.elevation_bands:
            if band not in self.sswei_objects:
                continue
                
            sswei_obj = self.sswei_objects[band]
            sswei_data = sswei_obj.sswei_data
            
            # Try to extract elevation from band name
            try:
                # Assuming format like "1000-1500m" or similar
                elevation_str = band.replace('m', '').split('-')
                if len(elevation_str) == 2:
                    min_elev = float(elevation_str[0])
                    max_elev = float(elevation_str[1])
                    mid_elev = (min_elev + max_elev) / 2
                else:
                    # If not in expected format, use the band name as is
                    mid_elev = float(band.replace('m', ''))
            except:
                # If extraction fails, use the index as a proxy
                mid_elev = self.elevation_bands.index(band)
            
            # Calculate drought statistics
            drought_years = sswei_data[sswei_data['SWEI'] < 0]
            drought_count = len(drought_years)
            severe_drought_count = len(sswei_data[sswei_data['SWEI'] < -1.5])
            mean_severity = drought_years['SWEI'].mean() if not drought_years.empty else 0
            
            # Calculate drought frequency
            total_years = len(sswei_data)
            drought_frequency = (drought_count / total_years) * 100 if total_years > 0 else 0
            
            # Add to elevation data
            elevation_data.append({
                'Elevation_Band': band,
                'Mid_Elevation': mid_elev,
                'Drought_Frequency': drought_frequency,
                'Mean_Severity': mean_severity,
                'Severe_Drought_Count': severe_drought_count
            })
        
        elevation_df = pd.DataFrame(elevation_data)
        
        # Sort by elevation
        elevation_df = elevation_df.sort_values('Mid_Elevation')
        
        # Calculate correlation between elevation and drought metrics
        corr_freq = np.corrcoef(elevation_df['Mid_Elevation'], elevation_df['Drought_Frequency'])[0, 1]
        corr_severity = np.corrcoef(elevation_df['Mid_Elevation'], elevation_df['Mean_Severity'])[0, 1]
        
        elevation_df['Corr_Elevation_Frequency'] = corr_freq
        elevation_df['Corr_Elevation_Severity'] = corr_severity
        
        self.analysis_results['elevation_sensitivity'] = elevation_df
        
        return elevation_df
    
    def plot_elevation_band_comparison(self, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot comparison of drought conditions across elevation bands.
        
        Parameters
        ----------
        figsize : tuple, optional
            Figure size, by default (12, 8).
            
        Returns
        -------
        matplotlib.figure.Figure
            Figure containing the plot.
        """
        if 'elevation_band_comparison' not in self.analysis_results:
            self.compare_elevation_bands()
        
        comparison_df = self.analysis_results['elevation_band_comparison']
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Plot drought frequency
        sns.barplot(x='Elevation_Band', y='Drought_Frequency', data=comparison_df, ax=axes[0, 0])
        axes[0, 0].set_title('Drought Frequency by Elevation Band')
        axes[0, 0].set_ylabel('Frequency (%)')
        axes[0, 0].set_xlabel('Elevation Band')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot mean severity
        sns.barplot(x='Elevation_Band', y='Mean_Severity', data=comparison_df, ax=axes[0, 1])
        axes[0, 1].set_title('Mean Drought Severity by Elevation Band')
        axes[0, 1].set_ylabel('Mean Severity (SSWEI)')
        axes[0, 1].set_xlabel('Elevation Band')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot drought count
        sns.barplot(x='Elevation_Band', y='Drought_Count', data=comparison_df, ax=axes[1, 0])
        axes[1, 0].set_title('Drought Count by Elevation Band')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_xlabel('Elevation Band')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot severe drought count
        sns.barplot(x='Elevation_Band', y='Severe_Drought_Count', data=comparison_df, ax=axes[1, 1])
        axes[1, 1].set_title('Severe Drought Count by Elevation Band')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_xlabel('Elevation Band')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        return fig
    
    def plot_temporal_trends(self, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot temporal trends in drought conditions.
        
        Parameters
        ----------
        figsize : tuple, optional
            Figure size, by default (12, 8).
            
        Returns
        -------
        matplotlib.figure.Figure
            Figure containing the plot.
        """
        if 'temporal_trends' not in self.analysis_results:
            self.analyze_temporal_trends()
        
        trend_results = self.analysis_results['temporal_trends']
        
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        
        # Plot drought frequency trends
        for name, trend_data in trend_results.items():
            axes[0].plot(trend_data['End_Year'], trend_data['Drought_Frequency'], 
                        label=name, marker='o')
        
        axes[0].set_title('Drought Frequency Trends')
        axes[0].set_ylabel('Frequency (%)')
        axes[0].set_xlabel('End Year of Moving Window')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot mean severity trends
        for name, trend_data in trend_results.items():
            axes[1].plot(trend_data['End_Year'], trend_data['Mean_Severity'], 
                        label=name, marker='o')
        
        axes[1].set_title('Mean Drought Severity Trends')
        axes[1].set_ylabel('Mean Severity (SSWEI)')
        axes[1].set_xlabel('End Year of Moving Window')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return fig
    
    def plot_drought_synchronicity(self, figsize: Tuple[int, int] = (12, 8)) -> Tuple[plt.Figure, plt.Figure]:
        """
        Plot drought synchronicity across elevation bands or regions.
        
        Parameters
        ----------
        figsize : tuple, optional
            Figure size, by default (12, 8).
            
        Returns
        -------
        tuple
            Tuple containing two figures: (heatmap_fig, timeseries_fig)
        """
        if 'drought_synchronicity' not in self.analysis_results:
            self.analyze_drought_synchronicity()
        
        sync_results = self.analysis_results['drought_synchronicity']
        sync_data = sync_results['sync_data']
        agreement_matrix = sync_results['agreement_matrix']
        
        # Create heatmap of agreement matrix
        heatmap_fig, ax1 = plt.subplots(figsize=figsize)
        sns.heatmap(agreement_matrix, annot=True, cmap='viridis', ax=ax1)
        ax1.set_title('Drought Agreement Between Datasets')
        
        # Create time series of datasets in drought
        timeseries_fig, ax2 = plt.subplots(figsize=figsize)
        ax2.bar(sync_data.index, sync_data['Percent_in_Drought'], color='orangered')
        ax2.set_title('Percentage of Datasets in Drought by Year')
        ax2.set_ylabel('Percentage of Datasets (%)')
        ax2.set_xlabel('Year')
        ax2.grid(True, alpha=0.3)
        
        # Highlight years with all datasets in drought
        if sync_results['all_drought_years']:
            for year in sync_results['all_drought_years']:
                ax2.axvline(x=year, color='red', linestyle='--', alpha=0.7)
        
        # Highlight years with no datasets in drought
        if sync_results['no_drought_years']:
            for year in sync_results['no_drought_years']:
                ax2.axvline(x=year, color='blue', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        return heatmap_fig, timeseries_fig
    
    def plot_elevation_sensitivity(self, figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Plot the sensitivity of drought conditions to elevation.
        
        Parameters
        ----------
        figsize : tuple, optional
            Figure size, by default (12, 6).
            
        Returns
        -------
        matplotlib.figure.Figure
            Figure containing the plot.
        """
        if 'elevation_sensitivity' not in self.analysis_results:
            self.analyze_elevation_sensitivity()
        
        elevation_df = self.analysis_results['elevation_sensitivity']
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Plot drought frequency vs. elevation
        axes[0].scatter(elevation_df['Mid_Elevation'], elevation_df['Drought_Frequency'], 
                       s=80, alpha=0.7)
        axes[0].set_title(f'Drought Frequency vs. Elevation\nCorrelation: {elevation_df["Corr_Elevation_Frequency"].iloc[0]:.2f}')
        axes[0].set_ylabel('Drought Frequency (%)')
        axes[0].set_xlabel('Elevation (m)')
        
        # Add regression line
        x = elevation_df['Mid_Elevation']
        y = elevation_df['Drought_Frequency']
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        axes[0].plot(x, p(x), "r--", alpha=0.7)
        
        # Plot mean severity vs. elevation
        axes[1].scatter(elevation_df['Mid_Elevation'], elevation_df['Mean_Severity'], 
                       s=80, alpha=0.7)
        axes[1].set_title(f'Mean Drought Severity vs. Elevation\nCorrelation: {elevation_df["Corr_Elevation_Severity"].iloc[0]:.2f}')
        axes[1].set_ylabel('Mean Severity (SSWEI)')
        axes[1].set_xlabel('Elevation (m)')
        
        # Add regression line
        x = elevation_df['Mid_Elevation']
        y = elevation_df['Mean_Severity']
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        axes[1].plot(x, p(x), "r--", alpha=0.7)
        
        plt.tight_layout()
        
        return fig
    
    def plot_sswei_comparison(self, year: int, figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Plot comparison of SSWEI values for a specific year across all datasets.
        
        Parameters
        ----------
        year : int
            Year to compare.
        figsize : tuple, optional
            Figure size, by default (12, 6).
            
        Returns
        -------
        matplotlib.figure.Figure
            Figure containing the plot.
        """
        if not self.sswei_objects:
            raise ValueError("SSWEI has not been calculated. Use calculate_sswei() first.")
        
        # Collect SSWEI values for the specified year
        sswei_values = []
        
        for name, sswei_obj in self.sswei_objects.items():
            year_data = sswei_obj.sswei_data[sswei_obj.sswei_data['season_year'] == year]
            
            if not year_data.empty:
                sswei_value = year_data['SWEI'].iloc[0]
                sswei_values.append({
                    'Dataset': name,
                    'SSWEI': sswei_value
                })
        
        if not sswei_values:
            raise ValueError(f"No SSWEI data found for year {year}.")
        
        comparison_df = pd.DataFrame(sswei_values)
        
        # Sort by SSWEI value
        comparison_df = comparison_df.sort_values('SSWEI')
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Define colors based on drought classification
        colors = []
        for sswei in comparison_df['SSWEI']:
            if sswei <= -2.0:
                colors.append('darkred')  # Exceptional drought
            elif sswei <= -1.5:
                colors.append('red')      # Extreme drought
            elif sswei <= -1.0:
                colors.append('orangered')  # Severe drought
            elif sswei <= 0:
                colors.append('orange')   # Moderate drought
            else:
                colors.append('green')    # No drought
        
        # Create the bar plot
        bars = ax.bar(comparison_df['Dataset'], comparison_df['SSWEI'], color=colors)
        
        # Add a horizontal line at SSWEI = 0
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Add horizontal lines for drought thresholds
        ax.axhline(y=-1.0, color='orangered', linestyle='--', alpha=0.5)
        ax.axhline(y=-1.5, color='red', linestyle='--', alpha=0.5)
        ax.axhline(y=-2.0, color='darkred', linestyle='--', alpha=0.5)
        
        # Add labels and title
        ax.set_title(f'SSWEI Comparison for Year {year}')
        ax.set_ylabel('SSWEI')
        ax.set_xlabel('Dataset')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            if height < 0:
                va = 'top'
                offset = -5
            else:
                va = 'bottom'
                offset = 5
            ax.text(bar.get_x() + bar.get_width()/2., height + np.sign(height) * 0.05,
                   f'{height:.2f}', ha='center', va=va, rotation=0)
        
        plt.tight_layout()
        
        return fig
    
    def export_results(self, output_dir: str) -> None:
        """
        Export analysis results to CSV files.
        
        Parameters
        ----------
        output_dir : str
            Directory to save the results to.
        """
        import os
        
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Export elevation band comparison
        if 'elevation_band_comparison' in self.analysis_results:
            self.analysis_results['elevation_band_comparison'].to_csv(
                os.path.join(output_dir, 'elevation_band_comparison.csv'), index=False
            )
        
        # Export temporal trends
        if 'temporal_trends' in self.analysis_results:
            for name, trend_data in self.analysis_results['temporal_trends'].items():
                trend_data.to_csv(
                    os.path.join(output_dir, f'temporal_trends_{name}.csv'), index=False
                )
        
        # Export drought characteristics
        if 'drought_characteristics' in self.analysis_results:
            for name, chars_data in self.analysis_results['drought_characteristics'].items():
                if not chars_data.empty:
                    chars_data.to_csv(
                        os.path.join(output_dir, f'drought_characteristics_{name}.csv'), index=False
                    )
        
        # Export drought synchronicity
        if 'drought_synchronicity' in self.analysis_results:
            sync_results = self.analysis_results['drought_synchronicity']
            
            # Export sync data
            sync_results['sync_data'].to_csv(
                os.path.join(output_dir, 'drought_synchronicity.csv')
            )
            
            # Export agreement matrix
            sync_results['agreement_matrix'].to_csv(
                os.path.join(output_dir, 'drought_agreement_matrix.csv')
            )
        
        # Export elevation sensitivity
        if 'elevation_sensitivity' in self.analysis_results:
            self.analysis_results['elevation_sensitivity'].to_csv(
                os.path.join(output_dir, 'elevation_sensitivity.csv'), index=False
            )
        
        print(f"Analysis results exported to {output_dir}")
    
    def __repr__(self) -> str:
        """
        Return a string representation of the DroughtAnalysis object.
        
        Returns
        -------
        str
            String representation of the DroughtAnalysis object.
        """
        datasets_str = ", ".join(self.datasets.keys())
        return f"DroughtAnalysis(datasets=[{datasets_str}])"
