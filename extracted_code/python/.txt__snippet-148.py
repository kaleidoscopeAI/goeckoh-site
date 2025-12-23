import android.webkit.WebViewClient
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import okhttp3.*
import java.io.File
import java.io.IOException
class MainActivity : ComponentActivity() {
private lateinit var webView: WebView
private var recorder: MediaRecorder? = null
private var recordingFile: File? = null
private val client = OkHttpClient()
private val serverHost = "10.0.2.2"
private val serverPort = 5000
