• That error is coming from a third‑party plugin (“Gradle Metrics”: com.drakklord.gradle.metric.core). Remove/disable it and
  Android Studio will start cleanly. Quick fixes:

  Linux/macOS:

  - Close Android Studio.
  - Delete the plugin folder (adjust the version suffix if different):

    rm -rf ~/.local/share/Google/AndroidStudio*/plugins/gradle-metric*
    rm -rf ~/.local/share/Google/AndroidStudio*/plugins/com.drakklord.gradle.metric.core*
    (On macOS, the path is ~/Library/Application Support/Google/AndroidStudio*/plugins/…)

  Windows:

  - Close Android Studio.
  - Delete C:\Users\<you>\AppData\Roaming\Google\AndroidStudio*\plugins\gradle-metric* (or com.drakklord.gradle.metric.core*).

  Then restart Android Studio. If it still complains, start once with all plugins disabled and re-enable only the defaults:

  - Run studio.sh/studio64.exe with -disable-plugins com.drakklord.gradle.metric.core (or use the “Disable plugin and restart”
    button if the crash dialog offers it).

  No need to reinstall; removing that plugin is usually sufficient.

