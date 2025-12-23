  - TarsosDSP JAR (e.g., tarsosdsp-2.5.jar from Maven Central)
  - Sherpa-onnx Android AAR (1.10.12 from k2-fsa repo)
  - Lingua JAR (1.1.0)
    Then switch the Gradle deps to implementation(files("libs/<name>.jar")) (AARs also supported).
