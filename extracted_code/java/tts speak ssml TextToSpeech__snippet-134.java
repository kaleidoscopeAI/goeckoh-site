283
284 +    private fun calculateRms(frame: ShortArray): Double {
285 +        var sum = 0.0
286 +        frame.forEach { sum += (it / 32768.0).pow(2) }
287 +        return if (frame.isNotEmpty()) sqrt(sum / frame.size) else 0.0
288 +    }
289 +
290      private fun correctToFirstPerson(text: String): String {

