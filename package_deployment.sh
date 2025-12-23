#!/bin/bash
#
# Package Goeckoh System for Deployment
# Creates a clean, distributable package
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

VERSION="1.0.0"
PACKAGE_NAME="goeckoh-system"
BUILD_DIR="build"
DIST_DIR="dist"
PACKAGE_DIR="$BUILD_DIR/$PACKAGE_NAME-$VERSION"

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     Packaging Goeckoh System for Deployment                ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Clean previous builds
echo -e "${YELLOW}🧹 Cleaning previous builds...${NC}"
rm -rf "$BUILD_DIR" "$DIST_DIR"
mkdir -p "$BUILD_DIR" "$DIST_DIR"

# Create package structure
echo -e "${YELLOW}📦 Creating package structure...${NC}"
PACKAGE_DIR="$BUILD_DIR/$PACKAGE_NAME-$VERSION"
mkdir -p "$PACKAGE_DIR"/{bin,lib,assets,config,docs,scripts,icons}

# Copy essential Python files
echo -e "${YELLOW}📝 Copying Python source files...${NC}"
rsync -av --exclude='__pycache__' --exclude='*.pyc' \
    --exclude='venv' --exclude='new_venv' \
    --exclude='GOECKOH' --exclude='rust_core' \
    --exclude='node_modules' --exclude='.git' \
    --exclude='build' --exclude='dist' \
    --exclude='*.log' --exclude='*.pyc' \
    --exclude='deployment_logs' \
    . "$PACKAGE_DIR/" \
    --exclude-from=<(cat <<EOF
venv/
new_venv/
GOECKOH/
rust_core/
node_modules/
.git/
build/
dist/
*.log
__pycache__/
*.pyc
deployment_logs/
legacy/
EOF
)

# Copy essential GOECKOH files (the canonical system)
echo -e "${YELLOW}📝 Copying GOECKOH core system...${NC}"
if [ -d "GOECKOH/goeckoh" ]; then
    mkdir -p "$PACKAGE_DIR/GOECKOH/goeckoh"
    rsync -av --exclude='__pycache__' --exclude='*.pyc' \
        GOECKOH/goeckoh/ "$PACKAGE_DIR/GOECKOH/goeckoh/"
fi

# Copy Rust library if exists
echo -e "${YELLOW}🔧 Copying Rust library...${NC}"
if [ -f "libbio_audio.so" ]; then
    cp libbio_audio.so "$PACKAGE_DIR/"
fi

# Copy assets (models)
echo -e "${YELLOW}📦 Copying assets and models...${NC}"
if [ -d "assets" ]; then
    rsync -av assets/ "$PACKAGE_DIR/assets/"
fi

# Copy icons
echo -e "${YELLOW}🎨 Copying icons...${NC}"
if [ -d "icons" ]; then
    rsync -av icons/ "$PACKAGE_DIR/icons/"
fi

# Copy launcher scripts
echo -e "${YELLOW}🚀 Copying launcher scripts...${NC}"
cp launch_bubble_system.sh "$PACKAGE_DIR/bin/"
cp Goeckoh_System.desktop "$PACKAGE_DIR/"
chmod +x "$PACKAGE_DIR/bin/launch_bubble_system.sh"

# Create installation script
echo -e "${YELLOW}📝 Creating installation script...${NC}"
cat > "$PACKAGE_DIR/install.sh" << 'INSTALL_EOF'
#!/bin/bash
#
# Goeckoh System Installation Script
#

set -e

INSTALL_DIR="${1:-/opt/goeckoh}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     Installing Goeckoh Neuro-Acoustic System               ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check for root if installing to /opt
if [[ "$INSTALL_DIR" == /opt/* ]] && [ "$EUID" -ne 0 ]; then
    echo -e "${YELLOW}⚠️  Installing to $INSTALL_DIR requires sudo${NC}"
    echo "Please run: sudo $0 $INSTALL_DIR"
    exit 1
fi

# Create installation directory
echo -e "${YELLOW}📁 Creating installation directory...${NC}"
mkdir -p "$INSTALL_DIR"

# Copy files
echo -e "${YELLOW}📦 Copying files...${NC}"
cp -r "$SCRIPT_DIR"/* "$INSTALL_DIR/"
chmod +x "$INSTALL_DIR/bin/launch_bubble_system.sh"
chmod +x "$INSTALL_DIR/install.sh"

# Create virtual environment
echo -e "${YELLOW}🐍 Setting up Python environment...${NC}"
cd "$INSTALL_DIR"
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Install dependencies
echo -e "${YELLOW}📚 Installing Python dependencies...${NC}"
"$INSTALL_DIR/venv/bin/pip" install --upgrade pip wheel setuptools
"$INSTALL_DIR/venv/bin/pip" install -r requirements.txt

# Install desktop launcher
echo -e "${YELLOW}🖥️  Installing desktop launcher...${NC}"
if [ -f "$INSTALL_DIR/Goeckoh_System.desktop" ]; then
    # Update paths in desktop file
    sed -i "s|/home/jacob/bubble|$INSTALL_DIR|g" "$INSTALL_DIR/Goeckoh_System.desktop"
    
    # Install to user applications
    mkdir -p ~/.local/share/applications
    cp "$INSTALL_DIR/Goeckoh_System.desktop" ~/.local/share/applications/
    chmod +x ~/.local/share/applications/Goeckoh_System.desktop
    
    if command -v update-desktop-database &> /dev/null; then
        update-desktop-database ~/.local/share/applications/
    fi
fi

# Create symlink in /usr/local/bin for command-line access
if [ "$EUID" -eq 0 ]; then
    echo -e "${YELLOW}🔗 Creating system-wide symlink...${NC}"
    ln -sf "$INSTALL_DIR/bin/launch_bubble_system.sh" /usr/local/bin/goeckoh
    chmod +x /usr/local/bin/goeckoh
fi

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║              Installation Complete!                       ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}Installation location: ${GREEN}$INSTALL_DIR${NC}"
echo ""
echo -e "${BLUE}To launch the system:${NC}"
echo -e "  • Search for 'Goeckoh' in your application menu"
echo -e "  • Or run: ${GREEN}goeckoh gui${NC} (if installed system-wide)"
echo -e "  • Or run: ${GREEN}$INSTALL_DIR/bin/launch_bubble_system.sh gui${NC}"
echo ""
INSTALL_EOF

chmod +x "$PACKAGE_DIR/install.sh"

# Create README for package
echo -e "${YELLOW}📖 Creating package README...${NC}"
cat > "$PACKAGE_DIR/README.md" << 'README_EOF'
# Goeckoh Neuro-Acoustic Exocortex System

## Installation

### Quick Install

```bash
./install.sh
```

This will install to `/opt/goeckoh` (requires sudo).

### Custom Install Location

```bash
./install.sh /path/to/install
```

### User Install (No sudo)

```bash
./install.sh ~/goeckoh
```

## Requirements

- Python 3.8 or higher
- pip
- Audio hardware (microphone and speakers)
- 2GB+ free disk space

## Launching

After installation:

1. **Application Menu**: Search for "Goeckoh" in your application launcher
2. **Command Line**: `goeckoh gui` (if installed system-wide)
3. **Direct**: `/opt/goeckoh/bin/launch_bubble_system.sh gui`

## Launch Modes

- `gui` - Main GUI interface (default)
- `universe` - With Cognitive Nebula visualization
- `child` - Child-friendly interface
- `clinician` - Clinician dashboard
- `api` - API server mode

## Documentation

See `docs/` directory for detailed documentation.

## Support

For issues and questions, see the main project documentation.
README_EOF

# Create requirements file for deployment
echo -e "${YELLOW}📋 Creating deployment requirements...${NC}"
cat > "$PACKAGE_DIR/requirements.txt" << 'REQ_EOF'
numpy>=1.20.0
scipy>=1.7.0
sounddevice>=0.4.0
sherpa-onnx>=1.0.0
kivy>=2.0.0
kivymd>=1.0.0
textual>=0.1.0
requests>=2.25.0
Cython>=0.29.0
librosa>=0.9.0
webrtcvad>=2.0.10
soundfile>=0.10.0
websockets>=10.0
REQ_EOF

# Create .gitignore for package
echo -e "${YELLOW}📝 Creating package .gitignore...${NC}"
cat > "$PACKAGE_DIR/.gitignore" << 'EOF'
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
venv/
*.log
*.so
.DS_Store
EOF

# Create package archive
echo -e "${YELLOW}📦 Creating distribution archive...${NC}"
cd "$SCRIPT_DIR/$BUILD_DIR"
tar -czf "$SCRIPT_DIR/$DIST_DIR/$PACKAGE_NAME-$VERSION.tar.gz" "$PACKAGE_NAME-$VERSION"

# Create zip archive as well
cd "$SCRIPT_DIR/$BUILD_DIR"
zip -r "$SCRIPT_DIR/$DIST_DIR/$PACKAGE_NAME-$VERSION.zip" "$PACKAGE_NAME-$VERSION" > /dev/null

# Calculate sizes
PACKAGE_SIZE=$(du -sh "$SCRIPT_DIR/$PACKAGE_DIR" | cut -f1)
ARCHIVE_SIZE=$(du -sh "$SCRIPT_DIR/$DIST_DIR/$PACKAGE_NAME-$VERSION.tar.gz" | cut -f1)

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║              Packaging Complete!                          ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}Package created:${NC}"
echo -e "  📦 Package directory: ${GREEN}$SCRIPT_DIR/$PACKAGE_DIR${NC} (${PACKAGE_SIZE})"
echo -e "  📦 Tar archive: ${GREEN}$SCRIPT_DIR/$DIST_DIR/$PACKAGE_NAME-$VERSION.tar.gz${NC} (${ARCHIVE_SIZE})"
echo -e "  📦 Zip archive: ${GREEN}$SCRIPT_DIR/$DIST_DIR/$PACKAGE_NAME-$VERSION.zip${NC}"
echo ""
echo -e "${BLUE}To test the package:${NC}"
echo -e "  ${YELLOW}cd $SCRIPT_DIR/$PACKAGE_DIR${NC}"
echo -e "  ${YELLOW}./install.sh ~/test-goeckoh${NC}"
echo ""
echo -e "${BLUE}To distribute:${NC}"
echo -e "  ${YELLOW}Share: $SCRIPT_DIR/$DIST_DIR/$PACKAGE_NAME-$VERSION.tar.gz${NC}"
echo ""

