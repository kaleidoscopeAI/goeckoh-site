from src.platform_utils import is_android

def get_audio_driver(mic_q, spk_q, sr):
    if is_android():
        from src.audio_mobile import AndroidAudioBridge
        return AndroidAudioBridge(mic_q, spk_q, sr)
    else:
        from src.audio_desktop import DesktopAudioBridge
        return DesktopAudioBridge(mic_q, spk_q, sr)
