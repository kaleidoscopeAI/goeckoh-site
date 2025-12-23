var nativeHandle: Long = 0
    private set

init {
    nativeHandle = try {
        nativeCreate()
    } catch (e: UnsatisfiedLinkError) {
        Log.e("NsWrapper", "JNI library not loaded: ${e.message}")
        0L
    } catch (e: Exception) {
        Log.e("NsWrapper", "Failed to create native instance: ${e.message}")
        0L
    }
}

external fun nativeCreate(): Long

external fun nativeFree()

external fun nativeProcess(inL: ShortArray, inH: ShortArray?, outL: ShortArray, outH: ShortArray?): Int

fun process(audioData: ShortArray): ShortArray {
    if (nativeHandle == 0L) {
        throw IllegalStateException("Native instance not initialized")
    }

    val output = ShortArray(audioData.size)
    val result = nativeProcess(audioData, null, output, null)

    if (result != 0) {
        throw RuntimeException("Noise suppression processing failed with code: $result")
    }

    return output
}

protected fun finalize() {
    if (nativeHandle != 0L) {
        try {
            nativeFree()
        } catch (e: Exception) {
            Log.e("NsWrapper", "Error in finalizer: ${e.message}")
        }
    }
}
