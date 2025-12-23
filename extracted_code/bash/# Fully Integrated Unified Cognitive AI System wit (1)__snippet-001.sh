#!/bin/bash
# Create virtual environment and install backend dependencies

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r ../backend/requirements.txt
