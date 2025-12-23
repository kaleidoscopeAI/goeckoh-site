      - TarsosDSP JAR (e.g., tarsosdsp-2.5.jar from Maven Central)
      - Sherpa-onnx Android AAR (1.10.12 from k2-fsa repo)
      - Lingua JAR (1.1.0)
        Then switch the Gradle deps to implementation(files("libs/<name>.jar")) (AARs also supported).
  2. Add the model files above to app/src/main/assets/.
  3. Build: ./gradlew assembleDebug and install the APK.
