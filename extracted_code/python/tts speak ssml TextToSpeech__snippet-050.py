  - Model assets missing from app/src/main/assets/:
      - vosk-model-small-en-us-0.15.zip
      - vosk-model-small-es-0.42.zip
      - vosk-model-small-fr-0.22.zip
      - vits-piper-en_US-amy-medium.onnx
      - vits-piper-es_ES-mls_9972-medium.onnx
      - vits-piper-fr_FR-upmc-medium.onnx
      - gemma-1.1-2b-it-q4f16.task

  How to finish

  1. Ensure Maven access, or manually download the missing artifacts and place them in app/libs/:
      - TarsosDSP JAR (e.g., tarsosdsp-2.5.jar from Maven Central)
      - Sherpa-onnx Android AAR (1.10.12 from k2-fsa repo)
      - Lingua JAR (1.1.0)
        Then switch the Gradle deps to implementation(files("libs/<name>.jar")) (AARs also supported).
  2. Add the model files above to app/src/main/assets/.
  3. Build: ./gradlew assembleDebug and install the APK.

  If you want, I can flip the Gradle dependencies to local app/libs references (assuming you’ll drop the JAR/AARs in) and re-run
  the build once the files are there.

 
