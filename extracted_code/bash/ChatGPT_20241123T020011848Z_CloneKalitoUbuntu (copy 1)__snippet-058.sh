# Navigate to the project directory
cd ~/Documents/AI.AI/codebase

# Create necessary directories
mkdir -p core functionality scripts/{building-blocks,modules,main_scripts,test,docs,visualization,data,logs}

# Move scripts to appropriate directories
mv core_node.py core functionality scripts/modules/
mv dynamic_visualization.py core functionality scripts/visualization/
mv data_crawler.py core functionality scripts/modules/
mv mirrored_network.py core functionality scripts/main_scripts/
mv organic_ai_demo.py core functionality scripts/main_scripts/
mv resource_managment.py core functionality scripts/modules/

# Move log files and temporary data
mv node_logs.json core functionality scripts/logs/

