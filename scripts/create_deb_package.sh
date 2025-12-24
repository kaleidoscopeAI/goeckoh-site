#!/bin/bash
#
# Create Debian Package (.deb) for Goeckoh System
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VERSION="1.0.0"
PACKAGE_NAME="goeckoh-system"
DEB_NAME="goeckoh-system_${VERSION}_all"
BUILD_DIR="build/deb"
DEB_DIR="$BUILD_DIR/$DEB_NAME"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     Creating Debian Package                               ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Clean previous build
rm -rf "$BUILD_DIR"
mkdir -p "$DEB_DIR"/{DEBIAN,opt/goeckoh,usr/local/bin,usr/share/applications,usr/share/pixmaps}

# First create the regular package
echo -e "${YELLOW}📦 Creating base package...${NC}"
./package_deployment.sh

# Copy from regular package
PACKAGE_DIR="build/$PACKAGE_NAME-$VERSION"
if [ -d "$PACKAGE_DIR" ]; then
    echo -e "${YELLOW}📋 Copying files to DEB structure...${NC}"
    mkdir -p "$DEB_DIR/opt/goeckoh"
    cp -r "$PACKAGE_DIR"/* "$DEB_DIR/opt/goeckoh/"
fi

# Create DEBIAN control file
echo -e "${YELLOW}📝 Creating DEBIAN control file...${NC}"
cat > "$DEB_DIR/DEBIAN/control" << CONTROL_EOF
Package: goeckoh-system
Version: ${VERSION}
Section: education
Priority: optional
Architecture: all
Depends: python3 (>= 3.8), python3-pip, python3-venv, libasound2, portaudio19-dev
Maintainer: Goeckoh Team <support@goeckoh.com>
Description: Goeckoh Neuro-Acoustic Exocortex - Therapeutic Voice Therapy System
 A comprehensive therapeutic voice therapy and AGI system designed for
 Auditory Prediction Error (APE) therapy, optimized for autistic individuals.
 Features real-time voice cloning, speech correction, emotional regulation,
 and therapeutic interventions.
 .
 Features:
  - Real-time voice processing and cloning
  - Speech-to-text and text-to-speech
  - Autism-optimized voice activity detection
  - ABA therapeutics integration
  - Emotional regulation system
  - 3D Cognitive Nebula visualization
  - Multiple launch modes (GUI, Child, Clinician, API)
Homepage: https://goeckoh.com
Keywords: speech, therapy, autism, voice, correction, ai, therapeutic
CONTROL_EOF

# Create postinst script
cat > "$DEB_DIR/DEBIAN/postinst" << POSTINST_EOF
#!/bin/bash
set -e

# Create virtual environment and install dependencies
if [ -d /opt/goeckoh ] && [ ! -d /opt/goeckoh/venv ]; then
    python3 -m venv /opt/goeckoh/venv
    /opt/goeckoh/venv/bin/pip install --upgrade pip wheel setuptools
    /opt/goeckoh/venv/bin/pip install -r /opt/goeckoh/requirements.txt
fi

# Update desktop database
if command -v update-desktop-database &> /dev/null; then
    update-desktop-database /usr/share/applications/ 2>/dev/null || true
fi

# Create symlink
ln -sf /opt/goeckoh/bin/launch_bubble_system.sh /usr/local/bin/goeckoh 2>/dev/null || true

exit 0
POSTINST_EOF

chmod +x "$DEB_DIR/DEBIAN/postinst"

# Create prerm script
cat > "$DEB_DIR/DEBIAN/prerm" << PRERM_EOF
#!/bin/bash
set -e

# Remove symlink
rm -f /usr/local/bin/goeckoh

exit 0
PRERM_EOF

chmod +x "$DEB_DIR/DEBIAN/prerm"

# Copy desktop file
if [ -f "$PACKAGE_DIR/Goeckoh_System.desktop" ]; then
    cp "$PACKAGE_DIR/Goeckoh_System.desktop" "$DEB_DIR/usr/share/applications/"
    # Update paths in desktop file
    sed -i "s|/home/jacob/bubble|/opt/goeckoh|g" "$DEB_DIR/usr/share/applications/Goeckoh_System.desktop"
fi

# Copy icon if exists
if [ -f "$PACKAGE_DIR/icons/goeckoh-icon.png" ]; then
    cp "$PACKAGE_DIR/icons/goeckoh-icon.png" "$DEB_DIR/usr/share/pixmaps/goeckoh-icon.png"
    # Update desktop file to use system icon
    sed -i "s|Icon=.*|Icon=goeckoh-icon|g" "$DEB_DIR/usr/share/applications/Goeckoh_System.desktop"
fi

# Build DEB package
echo -e "${YELLOW}🔨 Building DEB package...${NC}"
cd "$BUILD_DIR"
dpkg-deb --build "$DEB_NAME" "../../dist/${DEB_NAME}.deb"

DEB_SIZE=$(du -sh "../../dist/${DEB_NAME}.deb" | cut -f1)

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║              DEB Package Created!                         ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}Package: ${GREEN}dist/${DEB_NAME}.deb${NC} (${DEB_SIZE})"
echo ""
echo -e "${BLUE}To install:${NC}"
echo -e "  ${YELLOW}sudo dpkg -i dist/${DEB_NAME}.deb${NC}"
echo -e "  ${YELLOW}sudo apt-get install -f${NC}  # Fix any missing dependencies"
echo ""

