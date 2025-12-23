  - Asset installer to copy/unzip models from assets into filesDir.
  - Layout and manifest complete.

  What you need to do:

  1. Place the required models into app/src/main/assets/:
      - vosk-model-small-en-us-0.15.zip
      - vosk-model-small-es-0.42.zip
      - vosk-model-small-fr-0.22.zip
      - vits-piper-en_US-amy-medium.onnx
      - vits-piper-es_ES-mls_9972-medium.onnx
      - vits-piper-fr_FR-upmc-medium.onnx
      - gemma-1.1-2b-it-q4f16.task
  2. Add a Gradle wrapper (on a machine with Gradle installed): from repo root run gradle wrapper --gradle-version 8.3 (or
     similar) to generate gradlew, gradlew.bat, gradle/wrapper/*.
  3. Build/install: ./gradlew assembleDebug then adb install app/build/outputs/apk/debug/app-debug.apk.

  If you provide a Gradle wrapper or allow me to drop in the generated wrapper files, I’ll add them and verify the build.


