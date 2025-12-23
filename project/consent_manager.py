from __future__ import annotations
class ConsentManager:
    """Manages user consent and data privacy boundaries."""
    def check_consent(self, action, user_preferences=None):
        # In a real system, this would check a user's privacy settings.
        # For now, we'll assume consent is given for most actions.
        if user_preferences and user_preferences.get("block_all"): # type: ignore
            return False
        return True