#!/bin/bash
#
# Package Goeckoh System for macOS
# Creates .app bundle and .dmg installer
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VERSION="1.0.0"
PACKAGE_NAME="goeckoh-system"
BUILD_DIR="build/macos"
DIST_DIR="dist/macos"
APP_NAME="Goeckoh System.app"
APP_DIR="$BUILD_DIR/$APP_NAME"
APP_CONTENTS="$APP_DIR/Contents"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     Packaging Goeckoh System for macOS                   ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo -e "${YELLOW}⚠️  Warning: Not running on macOS. Some features may not work.${NC}"
fi

# Clean previous builds
rm -rf "$BUILD_DIR" "$DIST_DIR"
mkdir -p "$BUILD_DIR" "$DIST_DIR"

# Create .app bundle structure
echo -e "${YELLOW}📦 Creating macOS .app bundle...${NC}"
mkdir -p "$APP_CONTENTS"/{MacOS,Resources,Frameworks}

# Copy files
PACKAGE_DIR="$BUILD_DIR/package"
mkdir -p "$PACKAGE_DIR"

rsync -av --exclude='__pycache__' --exclude='*.pyc' \
    --exclude='venv' --exclude='new_venv' \
    --exclude='GOECKOH' --exclude='rust_core' \
    --exclude='node_modules' --exclude='.git' \
    --exclude='build' --exclude='dist' \
    --exclude='*.log' \
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

# Copy GOECKOH core
if [ -d "GOECKOH/goeckoh" ]; then
    mkdir -p "$PACKAGE_DIR/GOECKOH/goeckoh"
    rsync -av --exclude='__pycache__' GOECKOH/goeckoh/ "$PACKAGE_DIR/GOECKOH/goeckoh/"
fi

# Copy to app bundle
cp -r "$PACKAGE_DIR"/* "$APP_CONTENTS/Resources/"

# Copy icon
if [ -f "icons/goeckoh-icon.png" ]; then
    cp icons/goeckoh-icon.png "$APP_CONTENTS/Resources/goeckoh-icon.png"
    # Convert to .icns if sips is available (macOS)
    if command -v sips &> /dev/null; then
        mkdir -p "$APP_CONTENTS/Resources/icon.iconset"
        sips -z 16 16 icons/goeckoh-icon.png --out "$APP_CONTENTS/Resources/icon.iconset/icon_16x16.png" 2>/dev/null || true
        sips -z 32 32 icons/goeckoh-icon.png --out "$APP_CONTENTS/Resources/icon.iconset/icon_16x16@2x.png" 2>/dev/null || true
        sips -z 32 32 icons/goeckoh-icon.png --out "$APP_CONTENTS/Resources/icon.iconset/icon_32x32.png" 2>/dev/null || true
        sips -z 64 64 icons/goeckoh-icon.png --out "$APP_CONTENTS/Resources/icon.iconset/icon_32x32@2x.png" 2>/dev/null || true
        sips -z 128 128 icons/goeckoh-icon.png --out "$APP_CONTENTS/Resources/icon.iconset/icon_128x128.png" 2>/dev/null || true
        sips -z 256 256 icons/goeckoh-icon.png --out "$APP_CONTENTS/Resources/icon.iconset/icon_128x128@2x.png" 2>/dev/null || true
        sips -z 256 256 icons/goeckoh-icon.png --out "$APP_CONTENTS/Resources/icon.iconset/icon_256x256.png" 2>/dev/null || true
        sips -z 512 512 icons/goeckoh-icon.png --out "$APP_CONTENTS/Resources/icon.iconset/icon_256x256@2x.png" 2>/dev/null || true
        sips -z 512 512 icons/goeckoh-icon.png --out "$APP_CONTENTS/Resources/icon.iconset/icon_512x512.png" 2>/dev/null || true
        if command -v iconutil &> /dev/null; then
            iconutil -c icns "$APP_CONTENTS/Resources/icon.iconset" -o "$APP_CONTENTS/Resources/goeckoh.icns" 2>/dev/null || true
        fi
    fi
fi

# Create Info.plist
echo -e "${YELLOW}📝 Creating Info.plist...${NC}"
cat > "$APP_CONTENTS/Info.plist" << 'PLIST_EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>CFBundleExecutable</key>
    <string>goeckoh</string>
    <key>CFBundleIconFile</key>
    <string>goeckoh.icns</string>
    <key>CFBundleIdentifier</key>
    <string>com.goeckoh.system</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>Goeckoh Neuro-Acoustic System</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0.0</string>
    <key>CFBundleVersion</key>
    <string>1.0.0</string>
    <key>LSMinimumSystemVersion</key>
    <string>10.13</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSMicrophoneUsageDescription</key>
    <string>Goeckoh needs microphone access for voice therapy and speech processing.</string>
    <key>NSCameraUsageDescription</key>
    <string>Goeckoh may use camera for visual feedback (optional).</string>
</dict>
</plist>
PLIST_EOF

# Create launcher script
echo -e "${YELLOW}📝 Creating macOS launcher...${NC}"
cat > "$APP_CONTENTS/MacOS/goeckoh" << 'LAUNCHER_EOF'
#!/bin/bash
# Goeckoh System macOS Launcher

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESOURCES_DIR="$APP_DIR/Resources"
cd "$RESOURCES_DIR"

# Check for Python
if ! command -v python3 &> /dev/null; then
    osascript -e 'display dialog "Python 3 is required. Please install from python.org" buttons {"OK"} default button "OK"'
    exit 1
fi

# Create virtual environment if needed
if [ ! -d "venv" ]; then
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip wheel setuptools
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# Launch based on mode
MODE="${1:-gui}"

case "$MODE" in
    gui)
        # Try PySide6 first
        python3 -c "import PySide6" 2>/dev/null
        if [ $? -eq 0 ]; then
            python3 -m apps.run_gui
        else
            python3 -m apps.gui_main
        fi
        ;;
    universe)
        python3 system_launcher.py --mode universe
        ;;
    api)
        python3 -m apps.real_unified_system --api
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Available modes: gui, universe, api"
        exit 1
        ;;
esac
LAUNCHER_EOF

chmod +x "$APP_CONTENTS/MacOS/goeckoh"

# Create .dmg
echo -e "${YELLOW}📦 Creating DMG package...${NC}"
DMG_NAME="${PACKAGE_NAME}-${VERSION}-macos.dmg"
DMG_DIR="$BUILD_DIR/dmg"
rm -rf "$DMG_DIR"
mkdir -p "$DMG_DIR"

# Copy app to DMG
cp -r "$APP_DIR" "$DMG_DIR/"

# Create Applications symlink
ln -s /Applications "$DMG_DIR/Applications"

# Create README
cat > "$DMG_DIR/README.txt" << 'DMG_README'
Goeckoh Neuro-Acoustic System - macOS Installation
==================================================

INSTALLATION:
1. Drag "Goeckoh System.app" to Applications folder
2. Open Applications and double-click Goeckoh System
3. First launch will set up Python environment

REQUIREMENTS:
- macOS 10.13 (High Sierra) or later
- Python 3.8 or higher (install via Homebrew: brew install python3)
- 2GB free disk space
- Microphone and speakers

PERMISSIONS:
- Microphone access will be requested on first launch
- Grant permissions for full functionality

LAUNCHING:
- Double-click the app in Applications
- Or use Spotlight: Cmd+Space, type "Goeckoh"

For command-line access, see Terminal usage in docs.
DMG_README

# Create DMG (requires hdiutil on macOS)
if command -v hdiutil &> /dev/null; then
    hdiutil create -volname "Goeckoh System" \
        -srcfolder "$DMG_DIR" \
        -ov -format UDZO \
        "$DIST_DIR/$DMG_NAME"
    echo -e "${GREEN}✅ DMG created: ${DIST_DIR}/${DMG_NAME}${NC}"
else
    echo -e "${YELLOW}⚠️  hdiutil not found. To create DMG on macOS:${NC}"
    echo -e "   hdiutil create -volname 'Goeckoh System' -srcfolder '$DMG_DIR' -ov -format UDZO '$DIST_DIR/$DMG_NAME'"
fi

# Create zip as alternative
cd "$BUILD_DIR"
zip -r "../$DIST_DIR/${PACKAGE_NAME}-${VERSION}-macos.zip" "$APP_NAME" > /dev/null

echo ""
echo -e "${GREEN}✅ macOS package created!${NC}"
echo -e "   📦 App bundle: ${APP_DIR}"
echo -e "   📦 ZIP: ${DIST_DIR}/${PACKAGE_NAME}-${VERSION}-macos.zip"
if [ -f "$DIST_DIR/$DMG_NAME" ]; then
    echo -e "   📦 DMG: ${DIST_DIR}/${DMG_NAME}"
fi
echo ""

