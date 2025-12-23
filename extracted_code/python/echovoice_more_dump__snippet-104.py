package com.kaleidoscope.body
import android.content.Context
import android.util.Log
import androidx.work.CoroutineWorker
import androidx.work.WorkerParameters
class SyncWorker(appContext: Context, params: WorkerParameters) : CoroutineWorker(appContext, params) {
private val TAG = "SyncWorker"
