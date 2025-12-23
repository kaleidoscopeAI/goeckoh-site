# Create and activate a virtual environment
python3 -m venv env
source env/bin/activate

# Install dependencies
pip install numpy networkx matplotlib scikit-learn

# Export PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/project"

