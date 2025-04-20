"""
Performance utilities for the Snow Drought Index package.

This module contains utilities for benchmarking, profiling, and monitoring memory usage.
"""

import time
import functools
import os
import psutil
import numpy as np
import pandas as pd
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

def benchmark(func: Callable) -> Callable:
    """
    Decorator to benchmark a function's execution time.
    
    Parameters
    ----------
    func : callable
        Function to benchmark.
        
    Returns
    -------
    callable
        Wrapped function that prints execution time.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Function '{func.__name__}' executed in {execution_time:.4f} seconds")
        return result
    return wrapper

def memory_usage(func: Callable) -> Callable:
    """
    Decorator to monitor memory usage of a function.
    
    Parameters
    ----------
    func : callable
        Function to monitor.
        
    Returns
    -------
    callable
        Wrapped function that prints memory usage.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Get current process
        process = psutil.Process(os.getpid())
        
        # Get memory usage before function call
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Call function
        result = func(*args, **kwargs)
        
        # Get memory usage after function call
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        
        # Calculate memory usage
        mem_used = mem_after - mem_before
        
        print(f"Function '{func.__name__}' memory usage: {mem_used:.2f} MB")
        print(f"Total memory usage: {mem_after:.2f} MB")
        
        return result
    return wrapper

def profile_function(func: Callable, *args, **kwargs) -> Tuple[Any, Dict[str, Any]]:
    """
    Profile a function's execution time and memory usage.
    
    Parameters
    ----------
    func : callable
        Function to profile.
    *args : tuple
        Positional arguments to pass to the function.
    **kwargs : dict
        Keyword arguments to pass to the function.
        
    Returns
    -------
    tuple
        Tuple containing (result, profile_stats).
    """
    # Get current process
    process = psutil.Process(os.getpid())
    
    # Get memory usage before function call
    mem_before = process.memory_info().rss / 1024 / 1024  # MB
    
    # Start timer
    start_time = time.time()
    
    # Call function
    result = func(*args, **kwargs)
    
    # End timer
    end_time = time.time()
    
    # Get memory usage after function call
    mem_after = process.memory_info().rss / 1024 / 1024  # MB
    
    # Calculate statistics
    execution_time = end_time - start_time
    mem_used = mem_after - mem_before
    
    # Create profile statistics
    profile_stats = {
        'execution_time': execution_time,
        'memory_before': mem_before,
        'memory_after': mem_after,
        'memory_used': mem_used
    }
    
    return result, profile_stats

def compare_performance(funcs: List[Callable], args_list: List[Tuple], kwargs_list: List[Dict],
                       labels: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Compare the performance of multiple functions.
    
    Parameters
    ----------
    funcs : list
        List of functions to compare.
    args_list : list
        List of positional arguments to pass to each function.
    kwargs_list : list
        List of keyword arguments to pass to each function.
    labels : list, optional
        List of labels for each function, by default None.
        
    Returns
    -------
    pandas.DataFrame
        DataFrame containing performance comparison.
    """
    if labels is None:
        labels = [func.__name__ for func in funcs]
    
    results = []
    
    for i, (func, args, kwargs) in enumerate(zip(funcs, args_list, kwargs_list)):
        _, stats = profile_function(func, *args, **kwargs)
        stats['function'] = labels[i]
        results.append(stats)
    
    return pd.DataFrame(results)

def benchmark_chunking(func: Callable, data: pd.DataFrame, chunk_sizes: List[int],
                      *args, **kwargs) -> pd.DataFrame:
    """
    Benchmark a function with different chunk sizes.
    
    Parameters
    ----------
    func : callable
        Function to benchmark.
    data : pandas.DataFrame
        Data to process.
    chunk_sizes : list
        List of chunk sizes to test.
    *args : tuple
        Additional positional arguments to pass to the function.
    **kwargs : dict
        Additional keyword arguments to pass to the function.
        
    Returns
    -------
    pandas.DataFrame
        DataFrame containing benchmark results.
    """
    results = []
    
    for chunk_size in chunk_sizes:
        # Split data into chunks
        n_chunks = max(1, len(data) // chunk_size)
        chunks = [data.iloc[i*chunk_size:(i+1)*chunk_size] for i in range(n_chunks)]
        
        # Process each chunk and measure time
        start_time = time.time()
        
        for chunk in chunks:
            func(chunk, *args, **kwargs)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        results.append({
            'chunk_size': chunk_size,
            'n_chunks': n_chunks,
            'execution_time': execution_time
        })
    
    return pd.DataFrame(results)

def benchmark_parallel(func: Callable, data: pd.DataFrame, n_jobs_list: List[int],
                      *args, **kwargs) -> pd.DataFrame:
    """
    Benchmark a function with different numbers of parallel jobs.
    
    Parameters
    ----------
    func : callable
        Function to benchmark.
    data : pandas.DataFrame
        Data to process.
    n_jobs_list : list
        List of numbers of jobs to test.
    *args : tuple
        Additional positional arguments to pass to the function.
    **kwargs : dict
        Additional keyword arguments to pass to the function.
        
    Returns
    -------
    pandas.DataFrame
        DataFrame containing benchmark results.
    """
    results = []
    
    for n_jobs in n_jobs_list:
        # Update kwargs with n_jobs
        kwargs_copy = kwargs.copy()
        kwargs_copy['n_jobs'] = n_jobs
        
        # Process data and measure time
        start_time = time.time()
        func(data, *args, **kwargs_copy)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        results.append({
            'n_jobs': n_jobs,
            'execution_time': execution_time
        })
    
    return pd.DataFrame(results)

def plot_benchmark_results(results: pd.DataFrame, x_col: str, y_col: str = 'execution_time',
                          title: str = 'Benchmark Results', figsize: Tuple[int, int] = (10, 6)):
    """
    Plot benchmark results.
    
    Parameters
    ----------
    results : pandas.DataFrame
        DataFrame containing benchmark results.
    x_col : str
        Column to use for x-axis.
    y_col : str, optional
        Column to use for y-axis, by default 'execution_time'.
    title : str, optional
        Plot title, by default 'Benchmark Results'.
    figsize : tuple, optional
        Figure size, by default (10, 6).
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the plot.
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(results[x_col], results[y_col], 'o-')
    
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    
    ax.grid(True, alpha=0.3)
    
    return fig

def monitor_memory_usage(interval: float = 1.0, duration: float = 60.0) -> pd.DataFrame:
    """
    Monitor memory usage over time.
    
    Parameters
    ----------
    interval : float, optional
        Sampling interval in seconds, by default 1.0.
    duration : float, optional
        Monitoring duration in seconds, by default 60.0.
        
    Returns
    -------
    pandas.DataFrame
        DataFrame containing memory usage over time.
    """
    # Get current process
    process = psutil.Process(os.getpid())
    
    # Initialize results
    results = []
    
    # Monitor memory usage
    start_time = time.time()
    end_time = start_time + duration
    
    while time.time() < end_time:
        # Get memory usage
        mem_usage = process.memory_info().rss / 1024 / 1024  # MB
        
        # Record time and memory usage
        elapsed_time = time.time() - start_time
        results.append({
            'time': elapsed_time,
            'memory_usage': mem_usage
        })
        
        # Wait for next sample
        time.sleep(interval)
    
    return pd.DataFrame(results)

def plot_memory_usage(memory_usage: pd.DataFrame, figsize: Tuple[int, int] = (10, 6)):
    """
    Plot memory usage over time.
    
    Parameters
    ----------
    memory_usage : pandas.DataFrame
        DataFrame containing memory usage over time.
    figsize : tuple, optional
        Figure size, by default (10, 6).
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the plot.
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(memory_usage['time'], memory_usage['memory_usage'])
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Memory Usage (MB)')
    ax.set_title('Memory Usage Over Time')
    
    ax.grid(True, alpha=0.3)
    
    return fig
