ANDROID_MAIN_ACTIVITY = r"""package com.echo.mobile
import android.Manifest
import android.content.pm.PackageManager
import android.media.MediaRecorder
import android.os.Bundle
import android.webkit.WebChromeClient
import android.webkit.WebView
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
class MainActivity : ComponentActivity() {
private lateinit var webView: WebView
private val micPermissionLauncher = registerForActivityResult(
