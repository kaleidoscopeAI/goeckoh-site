package com.kaleidoscope.body
import android.app.job.JobInfo
import android.app.job.JobScheduler
import android.content.ComponentName
import android.content.Context
import android.net.ConnectivityManager
import android.net.NetworkRequest
import android.os.Build
import android.provider.Settings
import android.util.Log
import androidx.work.*
import java.util.concurrent.TimeUnit
class DeviceHAL(private val ctx: Context) {
private val TAG = "DeviceHAL"
