package com.kaleidoscope.body
import android.content.Context
import android.opengl.GLES20
import android.opengl.GLSurfaceView
import android.util.AttributeSet
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10
import kotlin.math.sin
class GLCubeView(context: Context, attrs: AttributeSet? = null) : GLSurfaceView(context, attrs) {
private val renderer: CubeRenderer
