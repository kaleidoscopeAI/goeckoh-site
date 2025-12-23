#!/bin/bash
#
# Install Desktop Launcher for Goeckoh System
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     Installing Goeckoh System Desktop Launcher            ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if we're in the right directory
if [ ! -f "launch_bubble_system.sh" ] || [ ! -f "Goeckoh_System.desktop" ]; then
    echo -e "${RED}❌ Required files not found. Please run from the bubble directory.${NC}"
    exit 1
fi

# Make launcher script executable
echo -e "${YELLOW}📝 Making launcher script executable...${NC}"
chmod +x launch_bubble_system.sh
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Launcher script is now executable${NC}"
else
    echo -e "${RED}❌ Failed to make launcher script executable${NC}"
    exit 1
fi

# Update desktop file with absolute paths
echo -e "${YELLOW}📝 Updating desktop file with absolute paths...${NC}"
python3 << 'PYEOF'
import os
from pathlib import Path

script_dir = Path.cwd().resolve()
launcher_script = script_dir / "launch_bubble_system.sh"
desktop_file = script_dir / "Goeckoh_System.desktop"

if desktop_file.exists():
    content = desktop_file.read_text()
    
    # Replace paths with absolute paths
    content = content.replace(
        "/home/jacob/bubble/launch_bubble_system.sh",
        str(launcher_script)
    )
    content = content.replace(
        "Path=/home/jacob/bubble",
        f"Path={script_dir}"
    )
    
    desktop_file.write_text(content)
    print(f"✅ Updated desktop file")
    print(f"   Script: {launcher_script}")
    print(f"   Path: {script_dir}")
else:
    print("❌ Desktop file not found")
    exit(1)
PYEOF

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Failed to update desktop file${NC}"
    exit 1
fi

# Make desktop file executable
chmod +x Goeckoh_System.desktop

# Determine installation location
APPS_DIR="$HOME/.local/share/applications"
DESKTOP_DIR="$HOME/Desktop"

# Create applications directory if it doesn't exist
mkdir -p "$APPS_DIR"

# Copy to applications directory
echo -e "${YELLOW}📦 Installing to applications directory...${NC}"
cp Goeckoh_System.desktop "$APPS_DIR/"
chmod +x "$APPS_DIR/Goeckoh_System.desktop"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Installed to $APPS_DIR/Goeckoh_System.desktop${NC}"
else
    echo -e "${RED}❌ Failed to install to applications directory${NC}"
    exit 1
fi

# Update desktop database
echo -e "${YELLOW}🔄 Updating desktop database...${NC}"
if command -v update-desktop-database &> /dev/null; then
    update-desktop-database "$APPS_DIR"
    echo -e "${GREEN}✅ Desktop database updated${NC}"
else
    echo -e "${YELLOW}⚠️  update-desktop-database not found (may not be needed)${NC}"
fi

# Optionally create desktop shortcut
read -p "Create desktop shortcut? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    cp Goeckoh_System.desktop "$DESKTOP_DIR/"
    chmod +x "$DESKTOP_DIR/Goeckoh_System.desktop"
    echo -e "${GREEN}✅ Desktop shortcut created at $DESKTOP_DIR/Goeckoh_System.desktop${NC}"
fi

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║              Installation Complete!                       ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}You can now launch the system by:${NC}"
echo -e "  • Searching for 'Goeckoh' in your application menu"
echo -e "  • Clicking the desktop shortcut (if created)"
echo -e "  • Running: $APPS_DIR/Goeckoh_System.desktop"
echo ""
echo -e "${YELLOW}To test the launcher, run:${NC}"
echo -e "  ./launch_bubble_system.sh gui"
echo ""

