2541              hamiltonian = 0.0
2542 +        heart_name = "Neurocoherence Lattice"
2543 +        heart_sample = []
2544 +        heart_rust = {}
2545 +        try:
2546 +            # Python lattice sample
2547 +            nodes = np.asarray(self.crystalline_heart.nodes, dtype=flo
      at)
2548 +            if nodes.size >= 32:
2549 +                bins = nodes.reshape(32, -1).mean(axis=1)
2550 +                heart_sample = bins.tolist()
2551 +            else:
2552 +                heart_sample = nodes.tolist()
2553 +        except Exception:
2554 +            heart_sample = []
2555 +        # Optional: expose Rust heart snapshot if available
2556 +        try:
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

