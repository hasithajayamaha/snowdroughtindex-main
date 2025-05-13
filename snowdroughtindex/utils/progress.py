"""
Progress tracking utilities for the Snow Drought Index package.

This module provides tools for tracking long-running operations,
monitoring memory usage, and displaying progress bars.
"""

import time
import psutil
import gc
from typing import Optional, Callable
from functools import wraps
from tqdm import tqdm
import numpy as np

class ProgressTracker:
    """
    A class for tracking progress of long-running operations.
    
    Attributes
    ----------
    total : int
        Total number of steps to complete
    desc : str
        Description of the operation
    unit : str
        Unit of progress (e.g., 'steps', 'stations')
    memory_monitoring : bool
        Whether to monitor memory usage
    start_time : float
        Time when tracking started
    current_step : int
        Current step number
    step_times : list
        List of times taken for each step
    memory_usage : list
        List of memory usage at each step
    """
    
    def __init__(
        self,
        total: int,
        desc: str = "Processing",
        unit: str = "items",
        memory_monitoring: bool = True
    ):
        """
        Initialize the progress tracker.
        
        Parameters
        ----------
        total : int
            Total number of steps to complete
        desc : str, optional
            Description of the operation, by default "Processing"
        unit : str, optional
            Unit of progress, by default "items"
        memory_monitoring : bool, optional
            Whether to monitor memory usage, by default True
        """
        self.total = total
        self.desc = desc
        self.unit = unit
        self.memory_monitoring = memory_monitoring
        self.start_time = time.time()
        self.current_step = 0
        self.step_times = []
        self.memory_usage = []
        self.pbar = tqdm(total=total, desc=desc, unit=unit)
        
    def update(self, n: int = 1, message: Optional[str] = None):
        """
        Update the progress tracker.
        
        Parameters
        ----------
        n : int, optional
            Number of steps completed, by default 1
        message : str, optional
            Additional message to display, by default None
        """
        self.current_step += n
        current_time = time.time()
        step_time = current_time - (self.start_time + sum(self.step_times))
        self.step_times.append(step_time)
        
        if self.memory_monitoring:
            process = psutil.Process()
            memory_info = process.memory_info()
            self.memory_usage.append(memory_info.rss / 1024 / 1024)  # Convert to MB
            
        # Calculate estimated time remaining
        if len(self.step_times) > 0:
            avg_step_time = np.mean(self.step_times)
            remaining_steps = self.total - self.current_step
            est_time_remaining = avg_step_time * remaining_steps
            
            # Format time remaining
            if est_time_remaining > 3600:
                time_str = f"{est_time_remaining/3600:.1f} hours"
            elif est_time_remaining > 60:
                time_str = f"{est_time_remaining/60:.1f} minutes"
            else:
                time_str = f"{est_time_remaining:.1f} seconds"
        else:
            time_str = "calculating..."
            
        # Update progress bar with additional information
        if message:
            self.pbar.set_postfix({
                'message': message,
                'est. time remaining': time_str,
                'memory': f"{self.memory_usage[-1]:.1f}MB" if self.memory_usage else "N/A"
            })
        else:
            self.pbar.set_postfix({
                'est. time remaining': time_str,
                'memory': f"{self.memory_usage[-1]:.1f}MB" if self.memory_usage else "N/A"
            })
            
        self.pbar.update(n)
        
    def close(self):
        """Close the progress tracker and display summary."""
        self.pbar.close()
        total_time = time.time() - self.start_time
        
        if self.memory_monitoring and self.memory_usage:
            max_memory = max(self.memory_usage)
            avg_memory = np.mean(self.memory_usage)
            print(f"\nMemory usage summary:")
            print(f"  Maximum: {max_memory:.1f} MB")
            print(f"  Average: {avg_memory:.1f} MB")
            
        print(f"\nOperation completed in {total_time:.1f} seconds")
        print(f"Average time per {self.unit}: {total_time/self.total:.2f} seconds")

def track_progress(func: Callable) -> Callable:
    """
    Decorator for tracking progress of a function.
    
    Parameters
    ----------
    func : Callable
        Function to track progress of
        
    Returns
    -------
    Callable
        Wrapped function with progress tracking
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Check if progress_tracker is provided
        progress_tracker = kwargs.get('progress_tracker')
        if progress_tracker is None:
            # Create a new progress tracker if none provided
            total = kwargs.get('total', 100)
            progress_tracker = ProgressTracker(
                total=total,
                desc=func.__name__,
                unit="steps"
            )
            kwargs['progress_tracker'] = progress_tracker
            
        try:
            result = func(*args, **kwargs)
            progress_tracker.close()
            return result
        except Exception as e:
            progress_tracker.close()
            raise e
            
    return wrapper

def monitor_memory(func: Callable) -> Callable:
    """
    Decorator for monitoring memory usage of a function.
    
    Parameters
    ----------
    func : Callable
        Function to monitor memory usage of
        
    Returns
    -------
    Callable
        Wrapped function with memory monitoring
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024 / 1024  # Convert to MB
        
        try:
            result = func(*args, **kwargs)
            
            end_memory = process.memory_info().rss / 1024 / 1024
            memory_diff = end_memory - start_memory
            
            print(f"\nMemory usage for {func.__name__}:")
            print(f"  Start: {start_memory:.1f} MB")
            print(f"  End: {end_memory:.1f} MB")
            print(f"  Difference: {memory_diff:.1f} MB")
            
            # Force garbage collection
            gc.collect()
            
            return result
        except Exception as e:
            end_memory = process.memory_info().rss / 1024 / 1024
            memory_diff = end_memory - start_memory
            
            print(f"\nMemory usage for {func.__name__} (error occurred):")
            print(f"  Start: {start_memory:.1f} MB")
            print(f"  End: {end_memory:.1f} MB")
            print(f"  Difference: {memory_diff:.1f} MB")
            
            # Force garbage collection
            gc.collect()
            
            raise e
            
    return wrapper 