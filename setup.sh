#!/bin/bash
# Goeckoh System Setup Script
# Quick setup for development or production environments

set -e

echo "======================================================================="
echo "Goeckoh System Setup"
echo "======================================================================="
echo ""

# Detect environment
ENVIRONMENT="${GOECKOH_ENV:-development}"
echo "Environment: $ENVIRONMENT"
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version || { echo "Error: Python 3 not found"; exit 1; }
echo ""

# Install Python dependencies
echo "Installing Python dependencies..."
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
    echo "✓ Python dependencies installed"
else
    echo "⚠️  requirements.txt not found, skipping..."
fi
echo ""

# Run configuration validation
echo "Validating configuration..."
python3 validate_config.py
echo ""

# Run configuration integration
echo "Integrating subsystems..."
python3 integrate_config.py
echo ""

# Setup subsystems based on environment
if [ "$ENVIRONMENT" = "production" ]; then
    echo "Setting up for PRODUCTION..."
    
    # Check for voice profile
    if [ ! -d "voice_profiles" ] || [ -z "$(ls -A voice_profiles/*.wav 2>/dev/null)" ]; then
        echo ""
        echo "⚠️  WARNING: No voice profile found!"
        echo "   Production mode requires a voice profile."
        echo "   Please create one in voice_profiles/ directory."
        echo ""
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    # Use production config
    if [ -f config.prod.yaml ]; then
        echo "✓ Using config.prod.yaml"
    fi
    
else
    echo "Setting up for DEVELOPMENT..."
    
    # Use development config
    if [ -f config.dev.yaml ]; then
        echo "✓ Using config.dev.yaml (voice cloning disabled for faster dev)"
    fi
fi
echo ""

# Setup Cognitive Nebula (if exists)
if [ -d "project/cognitive-nebula(8)" ]; then
    echo "Setting up Cognitive Nebula..."
    cd "project/cognitive-nebula(8)"
    
    if [ -f package.json ]; then
        if [ ! -d "node_modules" ]; then
            echo "Installing Cognitive Nebula dependencies..."
            npm install
            echo "✓ Cognitive Nebula dependencies installed"
        else
            echo "✓ Cognitive Nebula already set up"
        fi
    fi
    
    cd - > /dev/null
    echo ""
fi

# Setup Goeckoh web app (if exists)
if [ -d "project/goeckoh" ]; then
    echo "Setting up Goeckoh web app..."
    cd "project/goeckoh"
    
    if [ -f package.json ]; then
        if [ ! -d "node_modules" ]; then
            echo "Installing Goeckoh web app dependencies..."
            npm install
            echo "✓ Goeckoh web app dependencies installed"
        else
            echo "✓ Goeckoh web app already set up"
        fi
        
        # Check for .env.local
        if [ ! -f .env.local ]; then
            echo ""
            echo "⚠️  Note: Create .env.local and set GEMINI_API_KEY for Goeckoh web app"
        fi
    fi
    
    cd - > /dev/null
    echo ""
fi

echo "======================================================================="
echo "Setup Complete!"
echo "======================================================================="
echo ""
echo "Next steps:"
echo ""

if [ "$ENVIRONMENT" = "production" ]; then
    echo "  1. Create/verify your voice profile in voice_profiles/"
    echo "  2. Update config.prod.yaml with correct voice_profile_path"
    echo "  3. Run: python -m cli start"
else
    echo "  Development mode (voice cloning disabled):"
    echo "  1. Run: python -m cli start"
    echo "  2. Or run Cognitive Nebula: cd project/cognitive-nebula\\(8\\) && npm run dev"
    echo "  3. Or run Goeckoh web: cd project/goeckoh && npm run dev"
fi

echo ""
echo "For configuration help, see: CONFIGURATION_GUIDE.md"
echo "======================================================================="
