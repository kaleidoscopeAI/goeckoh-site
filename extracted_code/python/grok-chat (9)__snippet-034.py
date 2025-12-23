    2557 +            from goeckoh_ctypes_wrapper import CrystallineHeartEngine
           # type: ignore
    2558 +            heart = CrystallineHeartEngine()
    2559 +            val, ar, coh = heart.get_affective_state()
    2560 +            heart_rust = {
    2561 +                "valence": float(val),
    2562 +                "arousal": float(ar),
    2563 +                "coherence": float(coh),
    2564 +            }
    2565 +        except Exception:
    2566 +            heart_rust = {}
    2567

