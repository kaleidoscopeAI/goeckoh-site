try {
    auto* ns = new webrtc::NoiseSuppressor();
    // Configure NS parameters
    webrtc::NoiseSuppressor::Config config;
    config.target_level = webrtc::NoiseSuppressor::NoiseReductionLevel::kHigh;

    if (ns->Initialize(1, 16000) != 0) { // 1 channel, 16kHz
        delete ns;
        return 0;
    }

    return reinterpret_cast<jlong>(ns);
} catch (const std::exception& e) {
    LOGE("Exception in nativeCreate: %s", e.what());
    return 0;
}
