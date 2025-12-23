"""Manage user profiles and personalization"""

def __init__(self, config: SystemConfig, logger: ProductionLogger):
    self.config = config
    self.logger = logger
    self.profiles_dir = Path("profiles")
    self.profiles_dir.mkdir(exist_ok=True)
    self.active_profiles = {}

def create_profile(self, name: str, preferences: Dict = None) -> UserProfile:
    """Create new user profile"""
    user_id = str(uuid.uuid4())
    profile = UserProfile(
        user_id=user_id,
        name=name,
        preferences=preferences or {}
    )

    self.active_profiles[user_id] = profile
    self.save_profile(profile)
    self.logger.log_event("PROFILE_CREATED", f"Created profile for {name}")

    return profile

def load_profile(self, user_id: str) -> Optional[UserProfile]:
    """Load user profile from disk"""
    profile_file = self.profiles_dir / f"{user_id}.json"

    if not profile_file.exists():
        return None

    try:
        with open(profile_file, 'r') as f:
            data = json.load(f)

        profile = UserProfile(
            user_id=data['user_id'],
            name=data['name'],
            preferences=data['preferences'],
            session_history=data['session_history'],
            emotional_baseline=EmotionalState(**data['emotional_baseline']),
            skill_levels=data['skill_levels'],
            created_at=data['created_at'],
            last_active=data['last_active']
        )

        self.active_profiles[user_id] = profile
        return profile

    except Exception as e:
        self.logger.log_event("PROFILE_LOAD_ERROR", str(e), 'ERROR')
        return None

def save_profile(self, profile: UserProfile):
    """Save user profile to disk"""
    profile_file = self.profiles_dir / f"{profile.user_id}.json"

    try:
        data = {
            'user_id': profile.user_id,
            'name': profile.name,
            'preferences': profile.preferences,
            'session_history': profile.session_history,
            'emotional_baseline': profile.emotional_baseline.__dict__,
            'skill_levels': profile.skill_levels,
            'created_at': profile.created_at,
            'last_active': profile.last_active
        }

        with open(profile_file, 'w') as f:
            json.dump(data, f, indent=2)

    except Exception as e:
        self.logger.log_event("PROFILE_SAVE_ERROR", str(e), 'ERROR')

def update_profile_activity(self, user_id: str):
    """Update profile last active timestamp"""
    if user_id in self.active_profiles:
        self.active_profiles[user_id].last_active = time.time()
        self.save_profile(self.active_profiles[user_id])

