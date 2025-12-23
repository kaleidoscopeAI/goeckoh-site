# Step 1: Create and activate a virtual environment
python3 -m venv node_demo_env
source node_demo_env/bin/activate

# Step 2: Install required packages
pip install numpy networkx scikit-learn

# Step 3: Set the Python path (if modules are in a specific directory)
export PYTHONPATH="${PYTHONPATH}:/home/studio/Desktop/node-demo"

# Step 4: Run the demo script
python3 /home/studio/Desktop/node-demo/core_demo.py

