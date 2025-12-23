#!/bin/bash
#
# Package Goeckoh System for Windows
# Creates Windows installer and portable package
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VERSION="1.0.0"
PACKAGE_NAME="goeckoh-system"
BUILD_DIR="build/windows"
DIST_DIR="dist/windows"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     Packaging Goeckoh System for Windows                 ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Clean previous builds
rm -rf "$BUILD_DIR" "$DIST_DIR"
mkdir -p "$BUILD_DIR" "$DIST_DIR"

# Create package structure
PACKAGE_DIR="$BUILD_DIR/$PACKAGE_NAME-$VERSION"
mkdir -p "$PACKAGE_DIR"/{bin,lib,assets,config,docs,scripts,icons}

echo -e "${YELLOW}📦 Creating Windows package...${NC}"

# Copy files (same as Linux package)
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

# Copy Windows-specific files
if [ -f "libbio_audio.dll" ] || [ -f "libbio_audio.pyd" ]; then
    cp libbio_audio.* "$PACKAGE_DIR/" 2>/dev/null || true
fi

# Copy assets
if [ -d "assets" ]; then
    rsync -av assets/ "$PACKAGE_DIR/assets/"
fi

# Copy icons
if [ -d "icons" ]; then
    rsync -av icons/ "$PACKAGE_DIR/icons/"
fi

# Create Windows launcher script
echo -e "${YELLOW}📝 Creating Windows launcher...${NC}"
cat > "$PACKAGE_DIR/goeckoh.bat" << 'BAT_EOF'
@echo off
REM Goeckoh System Launcher for Windows

setlocal
set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

REM Check for Python
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from python.org
    pause
    exit /b 1
)

REM Check for virtual environment
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    call venv\Scripts\activate.bat
    python -m pip install --upgrade pip wheel setuptools
    python -m pip install -r requirements.txt
) else (
    call venv\Scripts\activate.bat
)

REM Launch based on argument
set MODE=%~1
if "%MODE%"=="" set MODE=gui

if "%MODE%"=="gui" (
    REM Try PySide6 first
    python -c "import PySide6" 2>nul
    if not errorlevel 1 (
        python -m apps.run_gui
    ) else (
        REM Fallback to Kivy
        python -m apps.gui_main
    )
) else if "%MODE%"=="universe" (
    python system_launcher.py --mode universe
) else if "%MODE%"=="api" (
    python -m apps.real_unified_system --api
) else (
    echo Unknown mode: %MODE%
    echo Available modes: gui, universe, api
    pause
    exit /b 1
)

endlocal
BAT_EOF

# Create Windows installer script (NSIS template)
echo -e "${YELLOW}📝 Creating Windows installer script...${NC}"
cat > "$BUILD_DIR/goeckoh-installer.nsi" << 'NSIS_EOF'
; Goeckoh System Windows Installer
; NSIS Installer Script

!define PRODUCT_NAME "Goeckoh Neuro-Acoustic System"
!define PRODUCT_VERSION "1.0.0"
!define PRODUCT_PUBLISHER "Goeckoh"
!define PRODUCT_WEB_SITE "https://goeckoh.com"
!define PRODUCT_DIR_REGKEY "Software\Microsoft\Windows\CurrentVersion\App Paths\goeckoh.exe"
!define PRODUCT_UNINST_KEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\${PRODUCT_NAME}"
!define PRODUCT_UNINST_ROOT_KEY "HKLM"

; Installer
Name "${PRODUCT_NAME}"
OutFile "goeckoh-system-${PRODUCT_VERSION}-setup.exe"
InstallDir "$PROGRAMFILES\Goeckoh"
InstallDirRegKey HKLM "${PRODUCT_DIR_REGKEY}" ""
RequestExecutionLevel admin
ShowInstDetails show
ShowUnInstDetails show

; Pages
Page directory
Page instfiles

; Installer sections
Section "MainSection" SEC01
    SetOutPath "$INSTDIR"
    File /r /x "__pycache__" /x "*.pyc" /x "venv" "*.*"
    
    ; Create virtual environment
    ExecWait 'python -m venv "$INSTDIR\venv"'
    ExecWait '"$INSTDIR\venv\Scripts\pip.exe" install --upgrade pip wheel setuptools'
    ExecWait '"$INSTDIR\venv\Scripts\pip.exe" install -r "$INSTDIR\requirements.txt"'
    
    ; Create shortcuts
    CreateDirectory "$SMPROGRAMS\Goeckoh"
    CreateShortCut "$SMPROGRAMS\Goeckoh\Goeckoh System.lnk" "$INSTDIR\goeckoh.bat" "" "$INSTDIR\icons\goeckoh-icon.ico"
    CreateShortCut "$SMPROGRAMS\Goeckoh\Uninstall.lnk" "$INSTDIR\uninstall.exe"
    CreateShortCut "$DESKTOP\Goeckoh System.lnk" "$INSTDIR\goeckoh.bat" "" "$INSTDIR\icons\goeckoh-icon.ico"
    
    ; Registry
    WriteRegStr HKLM "${PRODUCT_DIR_REGKEY}" "" "$INSTDIR\goeckoh.bat"
    WriteRegStr ${PRODUCT_UNINST_ROOT_KEY} "${PRODUCT_UNINST_KEY}" "DisplayName" "$(^Name)"
    WriteRegStr ${PRODUCT_UNINST_ROOT_KEY} "${PRODUCT_UNINST_KEY}" "UninstallString" "$INSTDIR\uninstall.exe"
    WriteRegStr ${PRODUCT_UNINST_ROOT_KEY} "${PRODUCT_UNINST_KEY}" "DisplayIcon" "$INSTDIR\icons\goeckoh-icon.ico"
    WriteRegStr ${PRODUCT_UNINST_ROOT_KEY} "${PRODUCT_UNINST_KEY}" "DisplayVersion" "${PRODUCT_VERSION}"
    WriteRegStr ${PRODUCT_UNINST_ROOT_KEY} "${PRODUCT_UNINST_KEY}" "URLInfoAbout" "${PRODUCT_WEB_SITE}"
SectionEnd

; Uninstaller
Section Uninstall
    Delete "$INSTDIR\uninstall.exe"
    RMDir /r "$INSTDIR"
    RMDir /r "$SMPROGRAMS\Goeckoh"
    Delete "$DESKTOP\Goeckoh System.lnk"
    DeleteRegKey ${PRODUCT_UNINST_ROOT_KEY} "${PRODUCT_UNINST_KEY}"
    DeleteRegKey HKLM "${PRODUCT_DIR_REGKEY}"
SectionEnd
NSIS_EOF

# Create portable package
echo -e "${YELLOW}📦 Creating portable package...${NC}"
cd "$BUILD_DIR"
zip -r "../$DIST_DIR/${PACKAGE_NAME}-${VERSION}-windows-portable.zip" "$PACKAGE_NAME-$VERSION" > /dev/null

# Create README for Windows
cat > "$PACKAGE_DIR/README_WINDOWS.txt" << 'WIN_README'
Goeckoh Neuro-Acoustic System - Windows Installation
====================================================

QUICK START:
1. Extract this folder to your desired location (e.g., C:\Program Files\Goeckoh)
2. Double-click goeckoh.bat to launch
3. First run will set up Python environment (may take a few minutes)

REQUIREMENTS:
- Windows 10 or later
- Python 3.8 or higher (download from python.org if not installed)
- 2GB free disk space
- Microphone and speakers

INSTALLATION OPTIONS:

Option 1: Portable (No Install)
- Extract zip file anywhere
- Run goeckoh.bat
- No admin rights needed

Option 2: Installer (Recommended)
- Run goeckoh-system-1.0.0-setup.exe
- Follow installation wizard
- Creates Start Menu shortcuts

LAUNCH MODES:
- Double-click goeckoh.bat (GUI mode)
- Right-click > Edit goeckoh.bat to change mode
- Modes: gui, universe, api

TROUBLESHOOTING:
- If Python not found: Install from python.org
- If audio doesn't work: Check Windows audio settings
- If errors occur: Check that all files extracted correctly

For more help, see docs/ directory.
WIN_README

echo ""
echo -e "${GREEN}✅ Windows package created!${NC}"
echo -e "   📦 Portable: ${DIST_DIR}/${PACKAGE_NAME}-${VERSION}-windows-portable.zip"
echo -e "   📝 Installer script: ${BUILD_DIR}/goeckoh-installer.nsi"
echo ""
echo -e "${YELLOW}To create installer:${NC}"
echo -e "   1. Install NSIS: https://nsis.sourceforge.io/"
echo -e "   2. Compile: makensis ${BUILD_DIR}/goeckoh-installer.nsi"
echo ""

