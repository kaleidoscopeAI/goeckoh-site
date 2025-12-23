*   **Python (Kivy/Pyjnius):** Accessing the microphone and speaker on Android/iOS via Python requires bridges like `Pyjnius` or `Pyobjus`. These add complexity and a "foreign function interface" (FFI) overhead every time you touch the hardware.
*   **Rust:** Rust has top-tier support for compiling to **Android (NDK)** and **iOS**. You can write a single, high-performance "audio engine" library in Rust and call it from a thin Kotlin (Android) or Swift (iOS) UI layer. This is the industry standard for high-performance mobile audio (e.g., Spotify, Discord use similar patterns).

