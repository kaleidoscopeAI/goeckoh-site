#!/usr/bin/env bash
# Safety: Exit immediately on error (-e), treat unset variables as an error (-u),
# and pipefail means a pipe's return status is the rightmost command's non-zero status.
set -euo pipefail

# ------------------------------------
# 0) Setup & Reminders
# ------------------------------------
echo "✅ Creating required directories..."
mkdir -p app/libs app/src/main/assets /tmp/piper

echo "⚠️ IMPORTANT: For the Gemma model (Section 4), you must first install"
echo "   'huggingface_hub' and run 'huggingface-cli login' to accept the license."
echo "   (pip install huggingface_hub)"
echo ""
echo "--- Starting Downloads ---"

############################################
# 1) Core JAR/AAR libs (local-only build)
############################################

# TarsosDSP (2.4 is the latest on Maven Central)
echo "⬇️ Downloading TarsosDSP JAR..."
wget -O app/libs/tarsosdsp-2.4.jar \
  https://repo1.maven.org/maven2/be/tarsos/tarsosdsp/2.4/tarsosdsp-2.4.jar

# Sherpa ONNX AAR (Using stable recommended version 1.10.16)
# Fallback version is 1.10.45 if 1.10.16 is deprecated/removed.
echo "⬇️ Downloading Sherpa ONNX AAR (1.10.16)..."
wget -O /tmp/sherpa-onnx-android-aar-1.10.16.tar.bz2 \
  https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.10.16/sherpa-onnx-android-aar-1.10.16.tar.bz2
tar -xjf /tmp/sherpa-onnx-android-aar-1.10.16.tar.bz2 -C /tmp
cp /tmp/sherpa-onnx-android-aar-1.10.16/sherpa-onnx-android-1.10.16.aar app/libs/

# WebRTC VAD AAR
echo "⬇️ Downloading WebRTC VAD AAR..."
wget -O app/libs/webrtc-vad-2.0.10.aar \
  https://jitpack.io/com/github/gkonovalov/android-vad/webrtc/2.0.10/webrtc-2.0.10.aar

# (Optional dependencies; uncomment if you need local copies of Kotlin/Lingua JARs)
# wget -O app/libs/lingua-1.1.0.jar https://repo1.maven.org/maven2/com/github/pemistahl/lingua/1.1.0/lingua-1.1.0.jar
# wget -O app/libs/kotlin-stdlib-1.8.20.jar https://repo1.maven.org/maven2/org/jetbrains/kotlin/kotlin-stdlib/1.8.20/kotlin-stdlib-1.8.20.jar
# wget -O app/libs/kotlin-reflect-1.8.20.jar https://repo1.maven.org/maven2/org/jetbrains/kotlin/kotlin-reflect/1.8.20/kotlin-reflect-1.8.20.jar

############################################
# 2) Vosk STT models (small, multi-lang)
############################################
echo "⬇️ Downloading Vosk STT models (EN, ES, FR)..."
wget -O app/src/main/assets/vosk-model-small-en-us-0.15.zip \
  https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
wget -O app/src/main/assets/vosk-model-small-es-0.42.zip \
  https://alphacephei.com/vosk/models/vosk-model-small-es-0.42.zip
wget -O app/src/main/assets/vosk-model-small-fr-0.22.zip \
  https://alphacephei.com/vosk/models/vosk-model-small-fr-0.22.zip

############################################
# 3) Piper TTS ONNX (real models)
############################################
# Note on "?download=1" suffix: The -O flag is used to explicitly name the local file,
# which correctly bypasses the issue of the download utility naming the file incorrectly.
echo "⬇️ Downloading Piper TTS ONNX models (EN, ES, FR)..."

wget -O app/src/main/assets/vits-piper-en_US-amy-medium.onnx \
  "https://huggingface.co/csukuangfj/vits-piper-en_US-amy-medium/resolve/main/model.onnx?download=1"

wget -O app/src/main/assets/vits-piper-es_ES-mls_9972-medium.onnx \
  "https://huggingface.co/csukuangfj/vits-piper-es_ES-mls_9972-medium/resolve/main/model.onnx?download=1"

wget -O app/src/main/assets/vits-piper-fr_FR-upmc-medium.onnx \
  "https://huggingface.co/csukuangfj/vits-piper-fr_FR-upmc-medium/resolve/main/model.onnx?download=1"


############################################
# 4) Gemma LLM task file (needs HF auth)
############################################
echo ""
echo "--------------------------------------------------------------------------------"
echo "⬇️ GEMMA MODEL DOWNLOAD (Manual/HF-CLI Required Step)"
echo "--------------------------------------------------------------------------------"
echo "To download the **Gemma LLM** model asset, run the following command **manually**"
echo "after verifying you have run 'huggingface-cli login' and accepted the license:"
echo ""
echo "huggingface-cli download google/gemma-1.1-2b-it gemma-1.1-2b-it-q4f16.task \\"
echo "  --local-dir app/src/main/assets --local-dir-use-symlinks False"
echo ""
echo "If you cannot use huggingface-cli, manually copy the file 'gemma-1.1-2b-it-q4f16.task'"
echo "into the 'app/src/main/assets/' directory."
echo "--------------------------------------------------------------------------------"

echo ""
echo "✅ All automated downloads complete. Files are now local."

echo "--- Next Step ---"
echo "After manually downloading the Gemma asset, you can build offline:"
echo ""
echo "./gradlew assembleDebug"
