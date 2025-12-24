#!/bin/bash

# Run backend from project root
echo "Starting backend..."
cd ..
export PYTHONPATH=$(pwd):$PYTHONPATH
source venv/bin/activate
python backend/server.py &

# Run frontend
echo "Starting frontend..."
cd frontend
npm run dev &

# Wait a moment for frontend to start
sleep 3



echo "System started!"
