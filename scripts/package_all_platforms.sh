#!/bin/bash
#
# Package Goeckoh System for All Platforms
# Creates packages for Linux, Windows, macOS, Android, and iOS
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     Packaging Goeckoh System for All Platforms           ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check for icon
if [ ! -f "icons/goeckoh-icon.png" ]; then
    echo -e "${YELLOW}⚠️  Warning: Icon not found at icons/goeckoh-icon.png${NC}"
    echo "   Some packages may not have proper icons"
    echo ""
fi

# Linux packages
echo -e "${GREEN}🐧 Packaging for Linux...${NC}"
if ./package_deployment.sh; then
    echo -e "${GREEN}✅ Linux package created${NC}"
    if ./create_deb_package.sh; then
        echo -e "${GREEN}✅ Linux DEB package created${NC}"
    fi
else
    echo -e "${RED}❌ Linux packaging failed${NC}"
fi
echo ""

# Windows package
echo -e "${GREEN}🪟 Packaging for Windows...${NC}"
if ./package_windows.sh; then
    echo -e "${GREEN}✅ Windows package created${NC}"
else
    echo -e "${RED}❌ Windows packaging failed${NC}"
fi
echo ""

# macOS package (only on macOS)
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo -e "${GREEN}🍎 Packaging for macOS...${NC}"
    if ./package_macos.sh; then
        echo -e "${GREEN}✅ macOS package created${NC}"
    else
        echo -e "${RED}❌ macOS packaging failed${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  Skipping macOS package (requires macOS)${NC}"
fi
echo ""

# Android package
echo -e "${GREEN}🤖 Packaging for Android...${NC}"
read -p "Build Android APK? This takes 10-30 minutes (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if ./package_android.sh debug; then
        echo -e "${GREEN}✅ Android package created${NC}"
    else
        echo -e "${RED}❌ Android packaging failed${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  Skipping Android package${NC}"
fi
echo ""

# iOS package (only on macOS)
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo -e "${GREEN}📱 Packaging for iOS...${NC}"
    read -p "Build iOS package? Requires Xcode (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        if ./package_ios.sh; then
            echo -e "${GREEN}✅ iOS package created${NC}"
        else
            echo -e "${RED}❌ iOS packaging failed${NC}"
        fi
    else
        echo -e "${YELLOW}⚠️  Skipping iOS package${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  Skipping iOS package (requires macOS)${NC}"
fi

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║              Cross-Platform Packaging Complete!          ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}Packages created in dist/ directory:${NC}"
echo ""
ls -lh dist/*/ 2>/dev/null | grep -E "\.(tar\.gz|zip|deb|dmg|apk|ipa)" || echo "   (Run individual platform scripts to create packages)"
echo ""
echo -e "${BLUE}See CROSS_PLATFORM_PACKAGING.md for details${NC}"
echo ""

