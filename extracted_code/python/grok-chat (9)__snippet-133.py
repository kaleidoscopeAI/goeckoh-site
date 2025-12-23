 991      def _default_sherpa_assets(self) -> Path:
 992 -        # mirrors speech_mirror default
 992 +        # Prefer echovoice models under backend/assets/echovoice_models if present
 993          backend_dir = Path(__file__).resolve().parents[2]
 994 -        return backend_dir / "echovoice_integration" / "assets"
 994 +        assets_candidates = [
 995 +            backend_dir / "assets" / "echovoice_models",
 996 +            backend_dir / "echovoice_integration" / "assets",
 997 +        ]
 998 +        for cand in assets_candidates:
 999 +            if cand.exists():
1000 +                return cand
1001 +        return assets_candidates[0]
1002

