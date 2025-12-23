                      // Get results from all
                      val results = recognizers.map { (lang, rec) ->
                          val json = rec.result
                          lang to parseResultWithConfidence(json)
                      }
                      // Select best
                      val best = results.maxByOrNull
