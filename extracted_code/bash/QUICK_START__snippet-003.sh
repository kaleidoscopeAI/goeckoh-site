# Copy to applications
cp Goeckoh_System.desktop ~/.local/share/applications/
chmod +x ~/.local/share/applications/Goeckoh_System.desktop

# Update database
update-desktop-database ~/.local/share/applications/

# Create desktop shortcut (optional)
cp Goeckoh_System.desktop ~/Desktop/
chmod +x ~/Desktop/Goeckoh_System.desktop
