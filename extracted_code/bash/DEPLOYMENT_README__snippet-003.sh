# Extract and run directly
tar -xzf goeckoh-system-1.0.0.tar.gz
cd goeckoh-system-1.0.0
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
./bin/launch_bubble_system.sh gui
