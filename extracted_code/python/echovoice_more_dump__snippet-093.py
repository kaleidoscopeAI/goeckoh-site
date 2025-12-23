package com.kaleidoscope.body
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Intent
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import android.util.Log
import kotlinx.coroutines.*
class MainService : Service() {
private val TAG = "MainService"
private val scope = CoroutineScope(Dispatchers.Default + SupervisorJob())
private lateinit var hal: DeviceHAL
private lateinit var mapper: ControlMapper
