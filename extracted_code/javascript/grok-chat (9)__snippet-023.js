    120 +    _mirrorPoller = Timer.periodic(const Duration(seconds: 2), (_) async {
    121 +      try {
    122 +        final status = await _apiService.getSystemStatus();
    123 +        final mirror = await _apiService.pollMirror();
    124 +        _mirrorController.add({
    125 +          "gcl": status["gcl"],
    126 +          "mode": status["system_mode"],
    127 +          "latency_ms": mirror["latency_ms"],
    128 +          "heart": mirror["heart_rust"] ?? mirror["heart"] ?? {},
    129 +        });
    130 +      } catch (_) {
    131 +        // swallow to keep UI responsive
    132 +      }
    133 +    });
    134 +  }
    135  }

