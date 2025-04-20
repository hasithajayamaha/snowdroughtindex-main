"""
Snow Drought Index Package

A package for analyzing snow drought conditions using various indices and methods.
"""

__version__ = '0.1.0'

# Import core modules
from snowdroughtindex.core.dataset import SWEDataset
from snowdroughtindex.core.sswei_class import SSWEI
from snowdroughtindex.core.drought_analysis import DroughtAnalysis
from snowdroughtindex.core.configuration import Configuration

# Import CLI module
from snowdroughtindex import cli
