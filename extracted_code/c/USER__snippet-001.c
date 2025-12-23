#include <jni.h>
#include <android/log.h>
#include "modules/audio_processing/ns/noise_suppressor.h"

#define LOG_TAG "WebRTC_NS"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

extern "C" {

JNIEXPORT jlong JNICALL
Java_com_exocortex_neuroacoustic_NsWrapper_nativeCreate(JNIEnv *env, jobject thiz) {
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
}

JNIEXPORT void JNICALL
Java_com_exocortex_neuroacoustic_NsWrapper_nativeFree(JNIEnv *env, jobject thiz, jlong handle) {
    if (handle != 0) {
        auto* ns = reinterpret_cast<webrtc::NoiseSuppressor*>(handle);
        delete ns;
    }
}

JNIEXPORT jint JNICALL
Java_com_exocortex_neuroacoustic_NsWrapper_nativeProcess(
    JNIEnv *env, jobject thiz, jlong handle, 
    jshortArray in_l, jshortArray in_h, 
    jshortArray out_l, jshortArray out_h) {
    
    if (handle == 0) return -1;
    
    auto* ns = reinterpret_cast<webrtc::NoiseSuppressor*>(handle);
    
    jshort* in_l_ptr = env->GetShortArrayElements(in_l, nullptr);
    jsize length = env->GetArrayLength(in_l);
    
    // Process audio (simplified - real implementation would use WebRTC AudioBuffer)
    webrtc::AudioBuffer audio_buffer(16000, 1, 16000, 1, 16000, 1);
    // Fill audio_buffer with input data...
    
    ns->AnalyzeCaptureAudio(&audio_buffer);
    ns->ProcessCaptureAudio(&audio_buffer);
    
    // Copy processed data to output...
    jshort* out_l_ptr = env->GetShortArrayElements(out_l, nullptr);
    // Copy processed samples to out_l_ptr...
    
    env->ReleaseShortArrayElements(in_l, in_l_ptr, JNI_ABORT);
    env->ReleaseShortArrayElements(out_l, out_l_ptr, 0);
    
    return 0;
}

} // extern "C"
