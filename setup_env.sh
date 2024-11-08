#!/bin/bash

# Specify the required Python version
required_version="3.12.2"

# Step 1: Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python $required_version to proceed."
    exit 1
fi

# Check the Python version
installed_version=$(python3 --version | awk '{print $2}')
if [[ "$installed_version" < "$required_version" ]]; then
    echo "Warning: You are using Python $installed_version."
    echo "This script requires Python $required_version or higher. Please install Python $required_version for best compatibility."
elif [[ "$installed_version" > "$required_version" ]]; then
    echo "Warning: You are using Python $installed_version."
    echo "This is beyond the recommended version $required_version. Compatibility issues may occur."
fi

# Step 2: Print message indicating the start of the installation process
echo "Installing required Python packages globally or into the current Python environment with Python $installed_version..."

# Step 3: Install the required Python packages using pip
python3 -m pip install --upgrade pip  # Upgrade pip to the latest version
python3 -m pip install numpy scipy torch numba py-markdown-table scikit-learn tensorly pandas

# Step 4: Install the tntorch package from GitHub
echo "Installing tntorch from GitHub..."
python3 -m pip install git+https://github.com/awkhan3/tntorch.git

# Print completion message
echo "Installation complete! All packages installed in the current Python environment."

