    Piper TTS pitch/volume transfer, metrics broadcast, and adaptive VAD from GCL.
  - Added asset installer and repo scaffolding (Gradle, manifest, layout).
  - Removed Whisper/RNNoise stubs; using Vosk + WebRTC VAD and optional system NoiseSuppressor.

  Place assets (required to run):

  - Put these into app/src/main/assets/:
      - vosk-model-small-en-us-0.15.zip
      - vosk-model-small-es-0.42.zip
      - vosk-model-small-fr-0.22.zip
      - vits-piper-en_US-amy-medium.onnx
      - vits-piper-es_ES-mls_9972-medium.onnx
      - vits-piper-fr_FR-upmc-medium.onnx
      - gemma-1.1-2b-it-q4f16.task

  Files to note:

  - Gradle: build.gradle, app/build.gradle
  - Core: app/src/main/java/com/exocortex/neuroacoustic/
    {MainActivity.kt,ExocortexService.kt,NeuroAcousticMirror.kt,CrystallineHeart.kt,GatedAGI.kt,Prosody.kt,AssetInstaller.kt}
  - Layout: app/src/main/res/layout/activity_main.xml

  After dropping assets:

  - (If you add Gradle wrapper) run ./gradlew assembleDebug, then install the APK.
  - Launch, grant mic permission, adjust VAD controls, watch metrics, and the loop will transcribe → correct to first-person →
    synthesize with your prosody and gate AGI actions by GCL.

  If you want me to add the Gradle wrapper or adjust model names/paths, say the word.


