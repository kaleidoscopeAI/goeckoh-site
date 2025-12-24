#!/bin/bash
#
# Setup Icon for Goeckoh System
# This script helps you set up the icon for the desktop launcher
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║          Goeckoh System Icon Setup                        ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Create icons directory
mkdir -p icons

ICON_PATH="icons/goeckoh-icon.png"
DESKTOP_FILE="Goeckoh_System.desktop"

echo -e "${YELLOW}📸 Icon Setup Instructions:${NC}"
echo ""
echo "1. Save your icon image file to:"
echo -e "   ${GREEN}$(pwd)/$ICON_PATH${NC}"
echo ""
echo "2. Supported formats: PNG (recommended), SVG, or ICO"
echo ""
echo "3. Recommended size: 256x256 pixels or larger"
echo ""

# Check if icon already exists
if [ -f "$ICON_PATH" ]; then
    echo -e "${GREEN}✅ Icon file found at: $ICON_PATH${NC}"
    
    # Get file info
    if command -v file &> /dev/null; then
        echo -e "${BLUE}   File type: $(file -b "$ICON_PATH")${NC}"
    fi
    
    # Update desktop file
    if [ -f "$DESKTOP_FILE" ]; then
        ABSOLUTE_ICON_PATH="$(pwd)/$ICON_PATH"
        
        # Update desktop file with absolute path
        python3 << PYEOF
import re
from pathlib import Path

desktop_file = Path("$DESKTOP_FILE")
if desktop_file.exists():
    content = desktop_file.read_text()
    
    # Replace Icon line with absolute path
    icon_pattern = r'^Icon=.*$'
    new_icon_line = f"Icon={Path('$ABSOLUTE_ICON_PATH').resolve()}"
    
    lines = content.split('\n')
    updated = False
    for i, line in enumerate(lines):
        if line.startswith('Icon='):
            lines[i] = new_icon_line
            updated = True
            break
    
    if updated:
        desktop_file.write_text('\n'.join(lines))
        print("✅ Updated desktop file with icon path")
    else:
        # Add Icon line if it doesn't exist
        if 'Icon=' not in content:
            # Add after Name line
            for i, line in enumerate(lines):
                if line.startswith('Name='):
                    lines.insert(i+1, new_icon_line)
                    desktop_file.write_text('\n'.join(lines))
                    print("✅ Added icon path to desktop file")
                    break
PYEOF
        
        echo -e "${GREEN}✅ Desktop file updated with icon path${NC}"
    else
        echo -e "${YELLOW}⚠️  Desktop file not found. Run install_launcher.sh first.${NC}"
    fi
else
    echo -e "${YELLOW}📋 To set up your icon:${NC}"
    echo ""
    echo "   Option 1: Copy your icon file:"
    echo -e "   ${BLUE}   cp /path/to/your/icon.png $ICON_PATH${NC}"
    echo ""
    echo "   Option 2: Download from URL:"
    echo -e "   ${BLUE}   wget -O $ICON_PATH <icon-url>${NC}"
    echo ""
    echo "   Option 3: Use an existing icon:"
    echo -e "   ${BLUE}   # Check for existing icons:${NC}"
    echo -e "   ${BLUE}   find . -name '*.png' -o -name '*.svg' | grep -i icon${NC}"
    echo ""
    echo "   Then run this script again to update the desktop file."
fi

echo ""
echo -e "${BLUE}💡 Tip: For best results, use PNG format at 256x256 or 512x512 pixels${NC}"

