def __init__(self, profile_name: str):
    print("✅ Initializing Goeckoh Real-Time Pipeline...")
    self.profile_name = profile_name
    self.stop_event = threading.Event()

    # Load Voice Profile
    self.profile_path = get_profile_path(self.profile_name)
    if not self.profile_path.exists():
        raise FileNotFoundError(f
