"""
Processes text for normalization and grammar correction.
"""

def __init__(self, settings: SpeechSettings):
    self.settings = settings
    self.tool = None
    if self.settings.language_tool_server:
        try:
            self.tool = language_tool_python.LanguageTool(
                self.settings.language_tool_server
            )
        except Exception as e:
            print(f"Could not connect to LanguageTool server: {e}")
            self.tool = None
    else:
        try:
            self.tool = language_tool_python.LanguageTool(
                self.settings.normalization_locale
            )
        except Exception as e:
            print(f"Could not initialize LanguageTool: {e}")
            self.tool = None


def normalize(self, text: str) -> str:
    """
    Normalizes and corrects the text.
    """
    if self.tool:
        return self.tool.correct(text)
    return text
