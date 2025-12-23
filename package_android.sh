#!/bin/bash
#
# Package Goeckoh System for Android
# Uses Buildozer to create APK
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VERSION="1.0.0"
BUILD_DIR="build/android"
DIST_DIR="dist/android"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     Packaging Goeckoh System for Android                ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check for buildozer
if ! command -v buildozer &> /dev/null; then
    echo -e "${YELLOW}📦 Installing Buildozer...${NC}"
    pip3 install buildozer
fi

# Check for Android SDK/NDK
if [ -z "$ANDROID_SDK_ROOT" ] && [ -z "$ANDROID_HOME" ]; then
    echo -e "${YELLOW}⚠️  Android SDK not found in environment${NC}"
    echo "   Buildozer will download SDK/NDK automatically on first build"
    echo "   This may take a while..."
fi

# Clean previous builds
rm -rf "$BUILD_DIR" "$DIST_DIR"
mkdir -p "$BUILD_DIR" "$DIST_DIR"

# Update buildozer.spec with current version
echo -e "${YELLOW}📝 Updating buildozer.spec...${NC}"
if [ -f "buildozer.spec" ]; then
    # Backup original
    cp buildozer.spec buildozer.spec.bak
    
    # Update version
    sed -i "s/^version = .*/version = $VERSION/" buildozer.spec
    
    # Ensure requirements are correct
    if ! grep -q "goeckoh" buildozer.spec; then
        # Add our package to requirements
        sed -i '/^requirements = /s/$/,goeckoh/' buildozer.spec
    fi
fi

# Build APK
echo -e "${YELLOW}🔨 Building Android APK...${NC}"
echo -e "${YELLOW}   This will take 10-30 minutes on first build...${NC}"
echo ""

# Check if we should build debug or release
BUILD_TYPE="${1:-debug}"

if [ "$BUILD_TYPE" == "release" ]; then
    echo -e "${BLUE}Building RELEASE APK (signed)...${NC}"
    buildozer android release
    APK_FILE="bin/goeckoh-${VERSION}-release.apk"
else
    echo -e "${BLUE}Building DEBUG APK...${NC}"
    buildozer android debug
    APK_FILE="bin/goeckoh-${VERSION}-debug.apk"
fi

# Copy APK to dist
if [ -f "$APK_FILE" ]; then
    mkdir -p "$DIST_DIR"
    cp "$APK_FILE" "$DIST_DIR/"
    echo ""
    echo -e "${GREEN}✅ Android APK created!${NC}"
    echo -e "   📦 APK: ${DIST_DIR}/$(basename $APK_FILE)"
    echo ""
    echo -e "${BLUE}To install on device:${NC}"
    echo -e "   ${YELLOW}adb install ${DIST_DIR}/$(basename $APK_FILE)${NC}"
    echo -e "   Or transfer APK to device and install manually"
else
    echo -e "${RED}❌ APK build failed. Check buildozer output above.${NC}"
    exit 1
fi

# Restore original buildozer.spec
if [ -f "buildozer.spec.bak" ]; then
    mv buildozer.spec.bak buildozer.spec
fi

echo ""
echo -e "${BLUE}📱 Android Package Complete!${NC}"
echo ""

