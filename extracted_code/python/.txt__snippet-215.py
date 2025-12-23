MAIN_ACTIVITY_KT = r"""package com.echo.mobile
import android.Manifest
import android.content.pm.PackageManager
import android.media.MediaRecorder
import android.os.Bundle
import android.util.Base64
import android.util.Log
import android.view.ViewGroup
import android.webkit.WebChromeClient
import android.webkit.WebView
import android.webkit.WebViewClient
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import okhttp3.*
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileInputStream
import java.io.IOException
class MainActivity : ComponentActivity() {
private lateinit var webView: WebView
private var recorder: MediaRecorder? = null
private var recordingFile: File? = null
private val client = OkHttpClient()
private val serverHost = "10.0.2.2"
private val serverPort = 5000
