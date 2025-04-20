"""
Command-line interface for the Snow Drought Index package.

This module provides a command-line interface for common operations in the
Snow Drought Index package, including data loading, gap filling, SSWEI calculation,
drought classification, visualization, and running complete workflows.
"""

import argparse
import os
import sys
import yaml
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional

from snowdroughtindex.core.dataset import SWEDataset
from snowdroughtindex.core.sswei_class import SSWEI
from snowdroughtindex.core.drought_analysis import DroughtAnalysis
from snowdroughtindex.core.configuration import Configuration


def parse_args():
    """
    Parse command-line arguments.
    
    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description='Snow Drought Index CLI',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Fill gaps command
    fill_gaps_parser = subparsers.add_parser(
        'fill-gaps',
        help='Fill gaps in SWE data'
    )
    fill_gaps_parser.add_argument(
        '--input-file',
        required=True,
        help='Path to input SWE data file'
    )
    fill_gaps_parser.add_argument(
        '--output-file',
        required=True,
        help='Path to output file for gap-filled data'
    )
    fill_gaps_parser.add_argument(
        '--window-days',
        type=int,
        default=15,
        help='Number of days to select data for around a certain day of year'
    )
    fill_gaps_parser.add_argument(
        '--min-obs-corr',
        type=int,
        default=10,
        help='Minimum number of overlapping observations required to calculate correlation'
    )
    fill_gaps_parser.add_argument(
        '--min-obs-cdf',
        type=int,
        default=5,
        help='Minimum number of stations required to calculate a station\'s CDF'
    )
    fill_gaps_parser.add_argument(
        '--min-corr',
        type=float,
        default=0.7,
        help='Minimum correlation value required to keep a donor station'
    )
    fill_gaps_parser.add_argument(
        '--parallel',
        action='store_true',
        help='Enable parallel processing'
    )
    fill_gaps_parser.add_argument(
        '--n-jobs',
        type=int,
        default=-1,
        help='Number of jobs for parallel processing (-1 for all available cores)'
    )
    fill_gaps_parser.add_argument(
        '--memory-efficient',
        action='store_true',
        help='Enable memory-efficient algorithms'
    )
    
    # Calculate SSWEI command
    calculate_sswei_parser = subparsers.add_parser(
        'calculate-sswei',
        help='Calculate SSWEI from SWE data'
    )
    calculate_sswei_parser.add_argument(
        '--input-file',
        required=True,
        help='Path to input SWE data file'
    )
    calculate_sswei_parser.add_argument(
        '--output-file',
        required=True,
        help='Path to output file for SSWEI results'
    )
    calculate_sswei_parser.add_argument(
        '--start-month',
        type=int,
        default=12,
        help='Starting month of the season (1-12)'
    )
    calculate_sswei_parser.add_argument(
        '--end-month',
        type=int,
        default=3,
        help='Ending month of the season (1-12)'
    )
    calculate_sswei_parser.add_argument(
        '--min-years',
        type=int,
        default=10,
        help='Minimum number of years required for calculation'
    )
    calculate_sswei_parser.add_argument(
        '--distribution',
        choices=['gamma', 'normal'],
        default='gamma',
        help='Probability distribution to use'
    )
    calculate_sswei_parser.add_argument(
        '--reference-period',
        nargs=2,
        type=int,
        metavar=('START_YEAR', 'END_YEAR'),
        help='Reference period (start_year end_year) for standardization'
    )
    calculate_sswei_parser.add_argument(
        '--parallel',
        action='store_true',
        help='Enable parallel processing'
    )
    calculate_sswei_parser.add_argument(
        '--n-jobs',
        type=int,
        default=-1,
        help='Number of jobs for parallel processing (-1 for all available cores)'
    )
    calculate_sswei_parser.add_argument(
        '--memory-efficient',
        action='store_true',
        help='Enable memory-efficient algorithms'
    )
    
    # Classify drought command
    classify_drought_parser = subparsers.add_parser(
        'classify-drought',
        help='Classify drought conditions based on SSWEI values'
    )
    classify_drought_parser.add_argument(
        '--input-file',
        required=True,
        help='Path to input SSWEI data file'
    )
    classify_drought_parser.add_argument(
        '--output-file',
        required=True,
        help='Path to output file for drought classification results'
    )
    classify_drought_parser.add_argument(
        '--thresholds',
        nargs='+',
        metavar='CLASS=THRESHOLD',
        help='Custom thresholds for drought classification (e.g., exceptional=-2.0 extreme=-1.5)'
    )
    
    # Plot SSWEI command
    plot_sswei_parser = subparsers.add_parser(
        'plot-sswei',
        help='Plot SSWEI time series'
    )
    plot_sswei_parser.add_argument(
        '--input-file',
        required=True,
        help='Path to input SSWEI data file'
    )
    plot_sswei_parser.add_argument(
        '--output-file',
        required=True,
        help='Path to output image file'
    )
    plot_sswei_parser.add_argument(
        '--title',
        default='SSWEI Time Series',
        help='Plot title'
    )
    plot_sswei_parser.add_argument(
        '--figsize',
        nargs=2,
        type=int,
        default=[10, 6],
        metavar=('WIDTH', 'HEIGHT'),
        help='Figure size (width height)'
    )
    plot_sswei_parser.add_argument(
        '--dpi',
        type=int,
        default=100,
        help='Figure DPI'
    )
    
    # Run workflow command
    run_workflow_parser = subparsers.add_parser(
        'run-workflow',
        help='Run a complete workflow'
    )
    run_workflow_parser.add_argument(
        '--input-file',
        required=True,
        help='Path to input SWE data file'
    )
    run_workflow_parser.add_argument(
        '--output-dir',
        required=True,
        help='Directory to save results'
    )
    run_workflow_parser.add_argument(
        '--config-file',
        help='Path to configuration file (YAML or JSON)'
    )
    run_workflow_parser.add_argument(
        '--workflow',
        choices=['sswei', 'drought-analysis', 'elevation-analysis'],
        default='sswei',
        help='Workflow to run'
    )
    
    return parser.parse_args()


def parse_thresholds(thresholds_list):
    """
    Parse threshold arguments.
    
    Parameters
    ----------
    thresholds_list : list
        List of threshold arguments in the format 'CLASS=THRESHOLD'.
        
    Returns
    -------
    dict
        Dictionary of thresholds.
    """
    thresholds = {}
    
    if thresholds_list:
        for threshold in thresholds_list:
            parts = threshold.split('=')
            if len(parts) == 2:
                class_name, value = parts
                try:
                    thresholds[class_name] = float(value)
                except ValueError:
                    print(f"Warning: Invalid threshold value for {class_name}: {value}")
    
    return thresholds


def fill_gaps_command(args):
    """
    Execute the fill-gaps command.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments.
    """
    print(f"Loading data from {args.input_file}...")
    
    # Create a configuration object
    config = Configuration()
    
    # Set performance parameters
    config.set_performance_params(
        parallel=args.parallel,
        n_jobs=args.n_jobs,
        memory_efficient=args.memory_efficient
    )
    
    # Create a SWEDataset object
    dataset = SWEDataset(config=config)
    dataset.load_from_file(args.input_file)
    
    # Preprocess data
    print("Preprocessing data...")
    dataset.preprocess()
    
    # Fill gaps
    print("Filling gaps...")
    dataset.gap_fill(
        window_days=args.window_days,
        min_obs_corr=args.min_obs_corr,
        min_obs_cdf=args.min_obs_cdf,
        min_corr=args.min_corr
    )
    
    # Save gap-filled data
    print(f"Saving gap-filled data to {args.output_file}...")
    dataset.save(args.output_file)
    
    print("Gap filling completed successfully.")


def calculate_sswei_command(args):
    """
    Execute the calculate-sswei command.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments.
    """
    print(f"Loading data from {args.input_file}...")
    
    # Create a configuration object
    config = Configuration()
    
    # Set performance parameters
    config.set_performance_params(
        parallel=args.parallel,
        n_jobs=args.n_jobs,
        memory_efficient=args.memory_efficient
    )
    
    # Create a SWEDataset object
    dataset = SWEDataset(config=config)
    dataset.load_from_file(args.input_file)
    
    # Preprocess data
    print("Preprocessing data...")
    dataset.preprocess()
    
    # Create an SSWEI object
    sswei_obj = SSWEI(dataset, config=config)
    
    # Parse reference period
    reference_period = None
    if args.reference_period:
        reference_period = tuple(args.reference_period)
    
    # Calculate SSWEI
    print("Calculating SSWEI...")
    sswei_obj.calculate_sswei(
        start_month=args.start_month,
        end_month=args.end_month,
        min_years=args.min_years,
        distribution=args.distribution,
        reference_period=reference_period
    )
    
    # Save SSWEI results
    print(f"Saving SSWEI results to {args.output_file}...")
    sswei_obj.save_results(args.output_file)
    
    print("SSWEI calculation completed successfully.")


def classify_drought_command(args):
    """
    Execute the classify-drought command.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments.
    """
    print(f"Loading SSWEI data from {args.input_file}...")
    
    # Load SSWEI data
    import pandas as pd
    sswei_data = pd.read_csv(args.input_file)
    
    # Create an SSWEI object
    sswei_obj = SSWEI()
    sswei_obj.sswei_data = sswei_data
    
    # Parse thresholds
    thresholds = parse_thresholds(args.thresholds)
    
    # Classify drought
    print("Classifying drought conditions...")
    sswei_obj.classify_drought(thresholds=thresholds if thresholds else None)
    
    # Save drought classification results
    print(f"Saving drought classification results to {args.output_file}...")
    sswei_obj.save_results(args.output_file)
    
    print("Drought classification completed successfully.")


def plot_sswei_command(args):
    """
    Execute the plot-sswei command.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments.
    """
    print(f"Loading SSWEI data from {args.input_file}...")
    
    # Load SSWEI data
    import pandas as pd
    sswei_data = pd.read_csv(args.input_file)
    
    # Create an SSWEI object
    sswei_obj = SSWEI()
    sswei_obj.sswei_data = sswei_data
    
    # Plot SSWEI time series
    print("Creating SSWEI time series plot...")
    fig = sswei_obj.plot_sswei_timeseries(figsize=tuple(args.figsize))
    
    # Set title
    fig.axes[0].set_title(args.title)
    
    # Save plot
    print(f"Saving plot to {args.output_file}...")
    fig.savefig(args.output_file, dpi=args.dpi)
    
    print("Plot created successfully.")


def run_workflow_command(args):
    """
    Execute the run-workflow command.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments.
    """
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load configuration if provided
    config = Configuration()
    if args.config_file:
        print(f"Loading configuration from {args.config_file}...")
        config.load_from_file(args.config_file)
    
    print(f"Loading data from {args.input_file}...")
    
    # Create a SWEDataset object
    dataset = SWEDataset(config=config)
    dataset.load_from_file(args.input_file)
    
    # Preprocess data
    print("Preprocessing data...")
    dataset.preprocess()
    
    # Fill gaps
    print("Filling gaps...")
    dataset.gap_fill()
    
    # Save gap-filled data
    gap_filled_file = os.path.join(args.output_dir, 'gap_filled_data.nc')
    print(f"Saving gap-filled data to {gap_filled_file}...")
    dataset.save(gap_filled_file)
    
    # Create an SSWEI object
    sswei_obj = SSWEI(dataset, config=config)
    
    # Calculate SSWEI
    print("Calculating SSWEI...")
    sswei_obj.calculate_sswei()
    
    # Classify drought
    print("Classifying drought conditions...")
    sswei_obj.classify_drought()
    
    # Save SSWEI results
    sswei_file = os.path.join(args.output_dir, 'sswei_results.csv')
    print(f"Saving SSWEI results to {sswei_file}...")
    sswei_obj.save_results(sswei_file)
    
    # Create plots
    print("Creating plots...")
    
    # SSWEI time series plot
    sswei_plot_file = os.path.join(args.output_dir, 'sswei_timeseries.png')
    fig = sswei_obj.plot_sswei_timeseries()
    fig.savefig(sswei_plot_file, dpi=100)
    
    # Drought classification heatmap
    heatmap_file = os.path.join(args.output_dir, 'drought_classification_heatmap.png')
    fig = sswei_obj.plot_drought_classification_heatmap()
    fig.savefig(heatmap_file, dpi=100)
    
    # Drought severity distribution
    severity_file = os.path.join(args.output_dir, 'drought_severity_distribution.png')
    fig = sswei_obj.plot_drought_severity_distribution()
    fig.savefig(severity_file, dpi=100)
    
    # Run additional workflow-specific steps
    if args.workflow == 'drought-analysis':
        print("Running drought analysis...")
        
        # Calculate drought characteristics
        drought_chars = sswei_obj.calculate_drought_characteristics()
        
        # Save drought characteristics
        drought_chars_file = os.path.join(args.output_dir, 'drought_characteristics.csv')
        drought_chars.to_csv(drought_chars_file, index=False)
        
        # Plot drought characteristics
        chars_plot_file = os.path.join(args.output_dir, 'drought_characteristics.png')
        fig = sswei_obj.plot_drought_characteristics()
        fig.savefig(chars_plot_file, dpi=100)
        
        # Analyze drought trends
        trend_data = sswei_obj.analyze_drought_trends()
        
        # Save trend data
        trend_file = os.path.join(args.output_dir, 'drought_trends.csv')
        trend_data.to_csv(trend_file, index=False)
        
        # Plot drought trends
        trend_plot_file = os.path.join(args.output_dir, 'drought_trends.png')
        fig = sswei_obj.plot_drought_trends()
        fig.savefig(trend_plot_file, dpi=100)
    
    elif args.workflow == 'elevation-analysis':
        print("Running elevation analysis...")
        
        # Create a DroughtAnalysis object
        analysis = DroughtAnalysis()
        
        # Add dataset
        analysis.add_dataset("All Stations", dataset)
        
        # Calculate SSWEI for all datasets
        analysis.calculate_sswei()
        
        # Compare elevation bands
        comparison_df = analysis.compare_elevation_bands()
        
        # Save comparison results
        comparison_file = os.path.join(args.output_dir, 'elevation_band_comparison.csv')
        comparison_df.to_csv(comparison_file, index=False)
        
        # Plot elevation band comparison
        comparison_plot_file = os.path.join(args.output_dir, 'elevation_band_comparison.png')
        fig = analysis.plot_elevation_band_comparison()
        fig.savefig(comparison_plot_file, dpi=100)
        
        # Analyze drought synchronicity
        sync_data = analysis.analyze_drought_synchronicity()
        
        # Save synchronicity results
        sync_file = os.path.join(args.output_dir, 'drought_synchronicity.csv')
        sync_data.to_csv(sync_file)
        
        # Plot drought synchronicity
        sync_plot_file = os.path.join(args.output_dir, 'drought_synchronicity.png')
        heatmap_fig, timeseries_fig = analysis.plot_drought_synchronicity()
        heatmap_fig.savefig(os.path.join(args.output_dir, 'drought_agreement_heatmap.png'), dpi=100)
        timeseries_fig.savefig(os.path.join(args.output_dir, 'drought_synchronicity_timeseries.png'), dpi=100)
    
    print(f"Workflow completed successfully. Results saved to {args.output_dir}")


def main():
    """
    Main entry point for the CLI.
    """
    args = parse_args()
    
    if args.command == 'fill-gaps':
        fill_gaps_command(args)
    elif args.command == 'calculate-sswei':
        calculate_sswei_command(args)
    elif args.command == 'classify-drought':
        classify_drought_command(args)
    elif args.command == 'plot-sswei':
        plot_sswei_command(args)
    elif args.command == 'run-workflow':
        run_workflow_command(args)
    else:
        print("Error: No command specified.")
        print("Run 'python -m snowdroughtindex.cli --help' for usage information.")
        sys.exit(1)


if __name__ == '__main__':
    main()
