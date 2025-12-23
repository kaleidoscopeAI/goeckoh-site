16  try:
17 -    from goeckoh_ctypes_wrapper import CrystallineHeartEngine  # type: i
    gnore
17 +    from goeckoh_ctypes_wrapper import CrystallineHeartEngine, VoiceSynt
    hesizerEngine  # type: ignore
18      HEART_AVAILABLE = True
19 +    RUST_VC_AVAILABLE = True
20  except Exception:
21      CrystallineHeartEngine = None  # type: ignore
22 +    VoiceSynthesizerEngine = None  # type: ignore
23      HEART_AVAILABLE = False
24 +    RUST_VC_AVAILABLE = False
25

