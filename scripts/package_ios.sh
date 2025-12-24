#!/bin/bash
#
# Package Goeckoh System for iOS
# Creates Xcode project for iOS build
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VERSION="1.0.0"
BUILD_DIR="build/ios"
DIST_DIR="dist/ios"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     Packaging Goeckoh System for iOS                     ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if on macOS (required for iOS builds)
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo -e "${RED}❌ iOS builds require macOS and Xcode${NC}"
    echo "   Please run this script on a Mac with Xcode installed"
    exit 1
fi

# Check for Xcode
if ! command -v xcodebuild &> /dev/null; then
    echo -e "${RED}❌ Xcode not found. Please install Xcode from App Store.${NC}"
    exit 1
fi

# Check for Python-for-iOS (briefcase or kivy-ios)
if ! command -v briefcase &> /dev/null && ! command -v toolchain &> /dev/null; then
    echo -e "${YELLOW}📦 iOS build tools not found.${NC}"
    echo ""
    echo "Options:"
    echo "  1. Use BeeWare Briefcase: pip install briefcase"
    echo "  2. Use Kivy iOS toolchain: pip install kivy-ios"
    echo ""
    read -p "Install Briefcase? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        pip3 install briefcase
    else
        echo "Please install build tools manually and run again"
        exit 1
    fi
fi

# Clean previous builds
rm -rf "$BUILD_DIR" "$DIST_DIR"
mkdir -p "$BUILD_DIR" "$DIST_DIR"

echo -e "${YELLOW}📦 Creating iOS package...${NC}"

# Option 1: Use Briefcase (BeeWare)
if command -v briefcase &> /dev/null; then
    echo -e "${BLUE}Using BeeWare Briefcase...${NC}"
    
    # Initialize briefcase if needed
    if [ ! -f "pyproject.toml" ]; then
        briefcase create
    fi
    
    # Update pyproject.toml
    cat > pyproject.toml << 'TOML_EOF'
[tool.briefcase]
project_name = "Goeckoh System"
bundle = "com.goeckoh.system"
version = "1.0.0"
url = "https://goeckoh.com"
license = "Proprietary"
author = "Goeckoh Team"
author_email = "support@goeckoh.com"

[tool.briefcase.app.goeckoh]
formal_name = "Goeckoh Neuro-Acoustic System"
description = "Therapeutic voice therapy and AGI system"
sources = ["goeckoh"]
requires = [
    "numpy>=1.20.0",
    "scipy>=1.7.0",
    "kivy>=2.0.0",
    "kivymd>=1.0.0",
]

[tool.briefcase.app.goeckoh.iOS]
requires = [
    "numpy>=1.20.0",
    "scipy>=1.7.0",
    "kivy>=2.0.0",
    "kivymd>=1.0.0",
]
TOML_EOF
    
    # Build iOS app
    echo -e "${YELLOW}🔨 Building iOS app (this takes time)...${NC}"
    briefcase build ios
    
    # Create IPA
    briefcase package ios
    
    echo -e "${GREEN}✅ iOS package created!${NC}"
    
# Option 2: Use Kivy iOS
elif command -v toolchain &> /dev/null; then
    echo -e "${BLUE}Using Kivy iOS toolchain...${NC}"
    
    # Create iOS project structure
    IOS_PROJECT="$BUILD_DIR/goeckoh-ios"
    mkdir -p "$IOS_PROJECT"
    
    echo -e "${YELLOW}📝 Creating iOS project...${NC}"
    echo -e "${YELLOW}   Note: Kivy iOS requires manual Xcode project setup${NC}"
    echo ""
    echo "To complete iOS build:"
    echo "  1. Run: toolchain build python3 kivy"
    echo "  2. Create Xcode project: toolchain create <title> <app_directory>"
    echo "  3. Open in Xcode and build"
    echo ""
    echo "See: https://github.com/kivy/kivy-ios for detailed instructions"
    
else
    echo -e "${RED}❌ No iOS build tool found${NC}"
    exit 1
fi

echo ""
echo -e "${BLUE}📱 iOS Package Instructions:${NC}"
echo ""
echo "After building:"
echo "  1. Open Xcode project"
echo "  2. Configure signing with your Apple Developer account"
echo "  3. Build and archive"
echo "  4. Export IPA for distribution"
echo ""

