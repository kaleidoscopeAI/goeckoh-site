# Installing the Desktop Launcher

## Quick Install

To make the system launchable by clicking an icon on your desktop:

### Option 1: Automatic Install (Recommended)

```bash
# Make the install script executable and run it
chmod +x install_launcher.sh
./install_launcher.sh
```

### Option 2: Manual Install

1. **Make the launcher script executable:**
   ```bash
   chmod +x launch_bubble_system.sh
   ```

2. **Copy the desktop file to your applications directory:**
   ```bash
   # For current user only
   cp Goeckoh_System.desktop ~/.local/share/applications/
   
   # Or for all users (requires sudo)
   sudo cp Goeckoh_System.desktop /usr/share/applications/
   ```

3. **Update desktop database:**
   ```bash
   update-desktop-database ~/.local/share/applications/
   ```

4. **Make desktop file executable:**
   ```bash
   chmod +x ~/.local/share/applications/Goeckoh_System.desktop
   ```

5. **Test the launcher:**
   ```bash
   ~/.local/share/applications/Goeckoh_System.desktop
   ```

## Creating a Desktop Shortcut

You can also create a desktop shortcut:

```bash
# Copy to desktop
cp Goeckoh_System.desktop ~/Desktop/
chmod +x ~/Desktop/Goeckoh_System.desktop
```

## Launch Modes

The launcher supports multiple modes accessible via right-click menu:

- **Default (GUI)**: Launches the main GUI interface
- **Universe Mode**: Launches with Cognitive Nebula visualization
- **Child Mode**: Child-friendly interface
- **Clinician Mode**: Clinician dashboard
- **API Server**: Starts the API server

## Troubleshooting

### Icon not showing
- The desktop file uses a generic icon. To use a custom icon:
  1. Place an icon file (PNG, SVG, or ICO) in the project directory
  2. Update the `Icon=` line in `Goeckoh_System.desktop` to point to the icon file

### Launcher doesn't work
- Check that `launch_bubble_system.sh` is executable: `chmod +x launch_bubble_system.sh`
- Check that Python 3 is installed: `python3 --version`
- Check the launcher script path in the desktop file is correct

### Permission errors
- Make sure the desktop file is executable: `chmod +x Goeckoh_System.desktop`
- Check file permissions on the launcher script

## Uninstalling

To remove the launcher:

```bash
rm ~/.local/share/applications/Goeckoh_System.desktop
update-desktop-database ~/.local/share/applications/
```

