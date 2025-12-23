[app]
title = Goeckoh System
package.name = goeckoh
package.domain = com.goeckoh
source.dir = .
source.include_exts = py,png,jpg,kv,atlas,onnx,json,txt,wav
source.include_patterns = assets/*,GOECKOH/goeckoh/systems/*,icons/*
version = 1.0.0
requirements = python3,kivy,kivymd,numpy,scipy,sherpa-onnx,jnius,android,requests,librosa,soundfile,webrtcvad
android.permissions = INTERNET,RECORD_AUDIO,MODIFY_AUDIO_SETTINGS,WRITE_EXTERNAL_STORAGE,READ_EXTERNAL_STORAGE
android.api = 33
android.minapi = 21
android.archs = arm64-v8a,armeabi-v7a
p4a.branch = master

[buildozer]
log_level = 2
