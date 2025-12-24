# Icon Setup Guide

## Quick Setup

1. **Save your icon image** to:
   ```
   /home/jacob/bubble/icons/goeckoh-icon.png
   ```

2. **Run the setup script:**
   ```bash
   ./setup_icon.sh
   ```

3. **Install the launcher:**
   ```bash
   ./install_launcher.sh
   ```

## Icon Requirements

- **Format**: PNG (recommended), SVG, or ICO
- **Size**: 256x256 pixels or larger (512x512 ideal)
- **Location**: `icons/goeckoh-icon.png`

## Manual Setup

If you prefer to set it up manually:

1. **Create icons directory:**
   ```bash
   mkdir -p icons
   ```

2. **Copy your icon:**
   ```bash
   cp /path/to/your/icon.png icons/goeckoh-icon.png
   ```

3. **Update desktop file:**
   The desktop file (`Goeckoh_System.desktop`) is already configured to use:
   ```
   Icon=/home/jacob/bubble/icons/goeckoh-icon.png
   ```

4. **Verify the icon path:**
   ```bash
   grep Icon= Goeckoh_System.desktop
   ```

## Alternative Icon Locations

The desktop file will automatically use the icon if it's placed at:
- `icons/goeckoh-icon.png` (primary)
- `icons/goeckoh-icon.svg` (SVG format)
- `icons/goeckoh.png` (alternative name)

## Testing the Icon

After setting up:

1. **Install the launcher:**
   ```bash
   ./install_launcher.sh
   ```

2. **Check in application menu:**
   - Search for "Goeckoh" in your application launcher
   - The icon should appear next to the application name

3. **If icon doesn't show:**
   - Verify the file exists: `ls -lh icons/goeckoh-icon.png`
   - Check file permissions: `chmod 644 icons/goeckoh-icon.png`
   - Update desktop database: `update-desktop-database ~/.local/share/applications/`

## Icon Description

Based on your icon:
- **Design**: Ear shape with integrated "G" letter
- **Style**: Metallic silver with circuit board patterns
- **Theme**: Technology, sound, and communication
- **Colors**: Cyan-blue and lime-green accent rings

This icon perfectly represents the Goeckoh Neuro-Acoustic Exocortex system!

