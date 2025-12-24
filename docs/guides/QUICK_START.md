# Quick Start - Launch Goeckoh System with Icon Click

## 🚀 Complete Setup (3 Steps)

### Step 1: Save Your Icon
Save your icon image to:
```
/home/jacob/bubble/icons/goeckoh-icon.png
```

**Recommended formats:**
- PNG (256x256 or 512x512 pixels) - Best compatibility
- SVG - Best scaling, works on modern systems

### Step 2: Make Scripts Executable
```bash
chmod +x launch_bubble_system.sh
chmod +x install_launcher.sh
chmod +x setup_icon.sh
```

### Step 3: Install the Launcher
```bash
./install_launcher.sh
```

This will:
- ✅ Install the desktop launcher to your applications menu
- ✅ Optionally create a desktop shortcut
- ✅ Update the desktop database

## 🎯 Launching the System

After installation, you can launch the system by:

1. **Application Menu**: Search for "Goeckoh" in your application launcher
2. **Desktop Shortcut**: Click the icon on your desktop (if created)
3. **Command Line**: `./launch_bubble_system.sh gui`

## 🎮 Launch Modes

Right-click the launcher icon to access different modes:

- **Default (GUI)**: Main graphical interface
- **Universe Mode**: With Cognitive Nebula visualization
- **Child Mode**: Child-friendly interface
- **Clinician Mode**: Clinician dashboard
- **API Server**: Start API server mode

## 📋 Files Created

- `launch_bubble_system.sh` - Main launcher script
- `Goeckoh_System.desktop` - Desktop entry file
- `install_launcher.sh` - Installation script
- `setup_icon.sh` - Icon setup helper
- `icons/` - Directory for icon files

## 🔧 Troubleshooting

### Icon not showing?
1. Verify icon exists: `ls -lh icons/goeckoh-icon.png`
2. Check permissions: `chmod 644 icons/goeckoh-icon.png`
3. Run: `./setup_icon.sh`

### Launcher not working?
1. Check script is executable: `chmod +x launch_bubble_system.sh`
2. Test manually: `./launch_bubble_system.sh gui`
3. Check Python: `python3 --version`

### Desktop file not appearing?
1. Reinstall: `./install_launcher.sh`
2. Update database: `update-desktop-database ~/.local/share/applications/`
3. Log out and log back in

## 📝 Manual Installation

If the install script doesn't work:

```bash
# Copy to applications
cp Goeckoh_System.desktop ~/.local/share/applications/
chmod +x ~/.local/share/applications/Goeckoh_System.desktop

# Update database
update-desktop-database ~/.local/share/applications/

# Create desktop shortcut (optional)
cp Goeckoh_System.desktop ~/Desktop/
chmod +x ~/Desktop/Goeckoh_System.desktop
```

## ✅ Verification

After installation, verify:

1. **Icon file exists:**
   ```bash
   ls -lh icons/goeckoh-icon.png
   ```

2. **Desktop file is correct:**
   ```bash
   cat Goeckoh_System.desktop | grep Icon
   ```

3. **Launcher works:**
   ```bash
   ./launch_bubble_system.sh help
   ```

4. **Application appears in menu:**
   - Search for "Goeckoh" in your application launcher
   - The icon should appear next to the name

---

**That's it!** Once set up, you can launch the entire Goeckoh Neuro-Acoustic Exocortex system with a single icon click! 🎉

