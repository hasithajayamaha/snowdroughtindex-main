#!/usr/bin/env python3
"""
Enhanced setup.py for Snow Drought Index package
Includes virtual environment management and multi-requirements installation
"""

import os
import sys
import subprocess
import venv
import platform
from pathlib import Path
from setuptools import setup, find_packages

# Package metadata
PACKAGE_NAME = "snowdroughtindex"
VERSION = "0.1.0"
VENV_DIR = "venv"

# Requirements files in installation order
REQUIREMENTS_FILES = [
    "requirements.txt",
    "requirements_extraction.txt", 
    "requirements_notebook.txt"
]

def print_status(message):
    """Print status message with formatting"""
    print(f"\n{'='*60}")
    print(f"  {message}")
    print(f"{'='*60}")

def print_step(step, message):
    """Print step message"""
    print(f"\n[Step {step}] {message}")

def run_command(command, description="", check=True):
    """Run a command with error handling"""
    if description:
        print(f"  → {description}")
    
    try:
        if isinstance(command, str):
            result = subprocess.run(command, shell=True, check=check, 
                                  capture_output=True, text=True)
        else:
            result = subprocess.run(command, check=check, 
                                  capture_output=True, text=True)
        
        if result.stdout:
            print(f"    {result.stdout.strip()}")
        return result
    except subprocess.CalledProcessError as e:
        print(f"    ERROR: {e}")
        if e.stderr:
            print(f"    STDERR: {e.stderr}")
        if check:
            sys.exit(1)
        return e

def create_virtual_environment():
    """Create virtual environment if it doesn't exist"""
    venv_path = Path(VENV_DIR)
    
    if venv_path.exists():
        print(f"  → Virtual environment already exists at {venv_path.absolute()}")
        return venv_path
    
    print(f"  → Creating virtual environment at {venv_path.absolute()}")
    try:
        venv.create(venv_path, with_pip=True)
        print(f"  → Virtual environment created successfully")
        return venv_path
    except Exception as e:
        print(f"  → ERROR creating virtual environment: {e}")
        sys.exit(1)

def get_activation_command():
    """Get the appropriate activation command for the platform"""
    venv_path = Path(VENV_DIR)
    
    if platform.system() == "Windows":
        activate_script = venv_path / "Scripts" / "activate.bat"
        python_exe = venv_path / "Scripts" / "python.exe"
        pip_exe = venv_path / "Scripts" / "pip.exe"
    else:
        activate_script = venv_path / "bin" / "activate"
        python_exe = venv_path / "bin" / "python"
        pip_exe = venv_path / "bin" / "pip"
    
    return str(activate_script), str(python_exe), str(pip_exe)

def upgrade_pip(pip_exe):
    """Upgrade pip to latest version"""
    print("  → Upgrading pip to latest version")
    run_command([pip_exe, "install", "--upgrade", "pip"], 
                "Upgrading pip")

def install_requirements(pip_exe):
    """Install requirements from all requirements files"""
    for i, req_file in enumerate(REQUIREMENTS_FILES, 1):
        req_path = Path(req_file)
        if req_path.exists():
            print(f"  → Installing requirements from {req_file}")
            run_command([pip_exe, "install", "-r", str(req_path)], 
                       f"Installing {req_file}")
        else:
            print(f"  → WARNING: {req_file} not found, skipping")

def install_package_dev(pip_exe):
    """Install the package in development mode"""
    print("  → Installing package in development mode")
    run_command([pip_exe, "install", "-e", "."], 
                "Installing package in development mode")

def print_completion_message():
    """Print completion message with activation instructions"""
    activate_script, python_exe, pip_exe = get_activation_command()
    
    print_status("SETUP COMPLETED SUCCESSFULLY!")
    print("\nTo activate your virtual environment:")
    
    if platform.system() == "Windows":
        print(f"  {VENV_DIR}\\Scripts\\activate.bat")
        print("\nOr in PowerShell:")
        print(f"  {VENV_DIR}\\Scripts\\Activate.ps1")
    else:
        print(f"  source {VENV_DIR}/bin/activate")
    
    print(f"\nPython executable: {python_exe}")
    print(f"Pip executable: {pip_exe}")
    
    print("\nInstalled requirements from:")
    for req_file in REQUIREMENTS_FILES:
        if Path(req_file).exists():
            print(f"  ✓ {req_file}")
        else:
            print(f"  ✗ {req_file} (not found)")
    
    print(f"\nPackage '{PACKAGE_NAME}' installed in development mode")
    print("\nYou can now run:")
    print("  python -c \"import snowdroughtindex; print('Package imported successfully!')\"")

def setup_environment():
    """Main setup function for virtual environment and requirements"""
    print_status("SNOW DROUGHT INDEX - ENVIRONMENT SETUP")
    
    # Step 1: Create virtual environment
    print_step(1, "Creating virtual environment")
    venv_path = create_virtual_environment()
    
    # Step 2: Get activation commands
    print_step(2, "Configuring virtual environment")
    activate_script, python_exe, pip_exe = get_activation_command()
    
    # Step 3: Upgrade pip
    print_step(3, "Upgrading pip")
    upgrade_pip(pip_exe)
    
    # Step 4: Install requirements
    print_step(4, "Installing requirements")
    install_requirements(pip_exe)
    
    # Step 5: Install package in development mode
    print_step(5, "Installing package")
    install_package_dev(pip_exe)
    
    # Step 6: Print completion message
    print_completion_message()

def show_help():
    """Show help message"""
    print(f"""
Snow Drought Index Setup Script

Usage:
    python setup.py                    - Full setup with virtual environment
    python setup.py install           - Full setup with virtual environment  
    python setup.py develop           - Development mode setup
    python setup.py --help           - Show this help message

This script will:
1. Create a virtual environment in './{VENV_DIR}/'
2. Activate the virtual environment
3. Upgrade pip to the latest version
4. Install requirements from:
   - requirements.txt
   - requirements_extraction.txt
   - requirements_notebook.txt
5. Install the package in development mode

Requirements files installation order ensures compatibility,
especially for NumPy version constraints in notebook requirements.
""")

# Setuptools configuration (for standard package installation)
SETUP_CONFIG = {
    "name": PACKAGE_NAME,
    "version": VERSION,
    "packages": find_packages(),
    "install_requires": [
        "numpy>=1.20.0,<2.0.0",
        "pandas>=1.3.0",
        "xarray>=0.19.0",
        "netCDF4>=1.5.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "scikit-learn>=1.0.0",
        "geopandas>=0.10.0",
        "shapely>=1.8.0",
        "h5py>=3.7.0",
        "statsmodels>=0.13.0",
        "properscoring>=0.1",
        "rasterio>=1.2.0",
    ],
    "author": "Snow Drought Index Team",
    "author_email": "example@example.com",
    "description": "A package for analyzing snow drought conditions using various indices and methods",
    "long_description": "A comprehensive package for analyzing snow drought conditions using various indices and methods, including SSWEI calculations, gap filling, and drought classification.",
    "keywords": "snow, drought, climate, hydrology, SWE",
    "url": "https://github.com/example/snowdroughtindex",
    "classifiers": [
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    "python_requires": ">=3.7",
}

if __name__ == "__main__":
    # Handle command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] in ["--help", "-h", "help"]:
            show_help()
            sys.exit(0)
        elif sys.argv[1] == "develop":
            # Development mode - just run setuptools
            setup(**SETUP_CONFIG)
            sys.exit(0)
        elif sys.argv[1] == "install":
            # Full setup mode
            setup_environment()
            sys.exit(0)
        else:
            # Let setuptools handle other commands
            setup(**SETUP_CONFIG)
            sys.exit(0)
    else:
        # Default behavior - full setup
        setup_environment()
