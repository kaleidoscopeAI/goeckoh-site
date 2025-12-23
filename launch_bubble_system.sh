#!/bin/bash
#
# Bubble System Unified Launcher
# Launches the complete Goeckoh Neuro-Acoustic Exocortex system
#

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Goeckoh Neuro-Acoustic Exocortex - System Launcher     ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 is not installed or not in PATH${NC}"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "system_launcher.py" ]; then
    echo -e "${RED}❌ system_launcher.py not found. Please run from the bubble directory.${NC}"
    exit 1
fi

# Check for virtual environment
if [ -d "venv" ] && [ -f "venv/bin/activate" ]; then
    echo -e "${YELLOW}📦 Activating virtual environment...${NC}"
    source venv/bin/activate
fi

# Determine launch mode (default to GUI if no argument)
MODE="${1:-gui}"

case "$MODE" in
    gui|GUI)
        echo -e "${GREEN}🚀 Launching GUI mode...${NC}"
        echo ""
        
        # Activate venv if available
        if [ -d "venv" ] && [ -f "venv/bin/activate" ]; then
            source venv/bin/activate
        fi
        
        # Try PySide6 GUI first (preferred - functional GUI)
        if python3 -c "import PySide6" 2>/dev/null; then
            echo -e "${BLUE}→ Starting Functional PySide6 GUI (with real system integration)...${NC}"
            python3 -m apps.run_gui
        # Fallback to Kivy GUI
        elif python3 -c "import kivy" 2>/dev/null; then
            echo -e "${BLUE}→ Starting Kivy GUI...${NC}"
            python3 -m apps.gui_main
        # Fallback to system launcher
        else
            echo -e "${YELLOW}⚠ No GUI framework found. Installing PySide6...${NC}"
            pip install PySide6 2>&1 | tail -5
            if python3 -c "import PySide6" 2>/dev/null; then
                echo -e "${BLUE}→ Starting PySide6 GUI...${NC}"
                python3 -m apps.run_gui
            else
                echo -e "${RED}❌ Failed to install GUI. Starting system launcher (universe mode)...${NC}"
                python3 system_launcher.py --mode universe
            fi
        fi
        ;;
    
    universe|UNIVERSE)
        echo -e "${GREEN}🌌 Launching Universe mode (with Cognitive Nebula)...${NC}"
        echo ""
        python3 system_launcher.py --mode universe
        ;;
    
    child|CHILD)
        echo -e "${GREEN}👶 Launching Child mode...${NC}"
        echo ""
        python3 main_app.py --mode child
        ;;
    
    clinician|CLINICIAN)
        echo -e "${GREEN}👨‍⚕️ Launching Clinician mode...${NC}"
        echo ""
        python3 main_app.py --mode clinician
        ;;
    
    api|API)
        echo -e "${GREEN}🌐 Launching API mode...${NC}"
        echo ""
        python3 -m apps.real_unified_system --api
        ;;
    
    nebula|NEBULA)
        echo -e "${GREEN}✨ Launching Cognitive Nebula only...${NC}"
        echo ""
        if [ -d "cognitive-nebula" ]; then
            cd cognitive-nebula
            if [ -f "package.json" ]; then
                npm run dev
            else
                echo -e "${RED}❌ Cognitive Nebula not set up. Run 'npm install' in cognitive-nebula/ first.${NC}"
                exit 1
            fi
        else
            echo -e "${RED}❌ cognitive-nebula directory not found${NC}"
            exit 1
        fi
        ;;
    
    help|--help|-h)
        echo "Usage: $0 [MODE]"
        echo ""
        echo "Modes:"
        echo "  gui         - Launch GUI (default, tries PySide6 then Kivy)"
        echo "  universe    - Launch with Cognitive Nebula visualization"
        echo "  child       - Launch child-friendly interface"
        echo "  clinician   - Launch clinician dashboard"
        echo "  api         - Launch API server mode"
        echo "  nebula      - Launch Cognitive Nebula frontend only"
        echo "  help        - Show this help message"
        echo ""
        exit 0
        ;;
    
    *)
        echo -e "${RED}❌ Unknown mode: $MODE${NC}"
        echo "Run '$0 help' for available modes"
        exit 1
        ;;
esac

