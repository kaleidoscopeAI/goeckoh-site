#!/bin/bash
# Build Goeckoh Desktop Application for All Platforms
# This script creates distributable packages for Windows, macOS, and Linux

set -e

echo "==================================="
echo "Goeckoh Desktop App Builder"
echo "==================================="
echo ""

# Change to electron-app directory
cd "$(dirname "$0")/electron-app"

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "Error: Node.js is not installed. Please install Node.js first."
    exit 1
fi

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "Error: npm is not installed. Please install npm first."
    exit 1
fi

echo "Installing dependencies..."
npm install

echo ""
echo "Dependencies installed successfully!"
echo ""

# Build for all platforms or specific platform
PLATFORM=${1:-all}

case $PLATFORM in
  windows|win)
    echo "Building for Windows..."
    npm run dist:win
    ;;
  
  macos|mac)
    echo "Building for macOS..."
    npm run dist:mac
    ;;
  
  linux)
    echo "Building for Linux..."
    npm run dist:linux
    ;;
  
  all)
    echo "Building for all platforms..."
    echo "Note: macOS builds require macOS, Windows builds work on all platforms."
    echo ""
    
    # Linux
    echo "Building Linux packages..."
    npm run dist:linux
    
    # Windows
    if [ "$(uname)" != "Darwin" ]; then
      echo "Building Windows packages..."
      npm run dist:win
    else
      echo "Skipping Windows build (requires macOS or Linux)"
    fi
    
    # macOS (only on macOS)
    if [ "$(uname)" == "Darwin" ]; then
      echo "Building macOS packages..."
      npm run dist:mac
    else
      echo "Skipping macOS build (requires macOS)"
    fi
    ;;
  
  *)
    echo "Usage: $0 [windows|macos|linux|all]"
    exit 1
    ;;
esac

echo ""
echo "==================================="
echo "Build Complete!"
echo "==================================="
echo ""
echo "Build artifacts are in: dist/electron/"
echo ""

# Show what was built
if [ -d "../dist/electron" ]; then
  echo "Built packages:"
  ls -lh ../dist/electron/
fi
