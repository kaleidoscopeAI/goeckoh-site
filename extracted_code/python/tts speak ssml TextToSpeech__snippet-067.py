 5  import android.content.SharedPreferences
 6 -import android.media.AudioAttributes
 7 -import android.media.AudioFormat
 8 -import android.media.AudioTrack
 6 +import android.content.Context
 7 +import android.content.Intent
 8 +import android.content.SharedPreferences
 9 +import android.media.*
10  import android.util.Log
   ⋮
24  import com.k2fsa.sherpa.onnx.OfflineTtsConfig
24 -import org.silero.vad.SileroVad
25 +import org.gkonovalov.android.vad.FrameSize
26 +import org.gkonovalov.android.vad.Mode
27 +import org.gkonovalov.android.vad.SampleRate
28 +import org.gkonovalov.android.vad.VadWebRTC
29 +import org.json.JSONObject
30 +import org.kaldi.Model
31 +import org.kaldi.Recognizer
32  import java.io.ByteArrayOutputStream

