 92          voskDirs.forEach { (lang, dir) ->
 93 -            val path = File(context.filesDir, dir)
 94 -            if (path.exists()) {
 95 -                models[lang] = Model(path.absolutePath)
 93 +            val base = File(context.filesDir, dir)
 94 +            val modelDir = resolveModelDir(base)
 95 +            if (modelDir != null && modelDir.exists()) {
 96 +                models[lang] = Model(modelDir.absolutePath)
 97              } else {
 97 -                Log.w("Mirror", "Vosk model missing for $lang at ${path
     .absolutePath}")
 98 +                Log.w("Mirror", "Vosk model missing for $lang at ${base
     .absolutePath}")
 99              }
    ⋮
102
103 +    private fun resolveModelDir(base: File): File? {
104 +        if (!base.exists()) return null

