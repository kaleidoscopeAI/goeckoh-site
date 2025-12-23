     1 -package com.exocortex.neuroacoustic
     2 -
     3 -import android.util.Log
     4 -
     5 -class NsWrapper {
     6 -    var nativeHandle: Long = 0
     7 -        private set
     8 -
     9 -    init {
    10 -        nativeHandle = try {
    11 -            nativeCreate()
    12 -        } catch (e: UnsatisfiedLinkError) {
    13 -            Log.e("NsWrapper", "JNI library not loaded: ${e.message}")
    14 -            0L
    15 -        } catch (e: Exception) {
    16 -            Log.e("NsWrapper", "Failed to create native instance: ${e.me
        ssage}")
    17 -            0L
    18 -        }
    19 -    }
    20 -
    21 -    external fun nativeCreate(): Long
    22 -    external fun nativeFree()
    23 -    external fun nativeProcess(inL: ShortArray, inH: ShortArray?, outL:
        ShortArray, outH: ShortArray?): Int
    24 -
    25 -    fun process(audioData: ShortArray): ShortArray {
    26 -        if (nativeHandle == 0L) return audioData
    27 -        val output = ShortArray(audioData.size)
    28 -        val result = nativeProcess(audioData, null, output, null)
    29 -        if (result != 0) {
    30 -            Log.e("NsWrapper", "Noise suppression failed code=$result")
    31 -            return audioData
    32 -        }
    33 -        return output
    34 -    }
    35 -
    36 -    fun nativeFreeSafe() {
    37 -        try {
    38 -            if (nativeHandle != 0L) nativeFree()
    39 -        } catch (_: Exception) {
    40 -        }
    41 -    }
    42 -}

