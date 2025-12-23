# Inside echo_prime.py -> EchoSystem class

    def set_gui_callback(self, callback_func):
        """Allows the GUI to hook into the system logs."""
        self.gui_callback = callback_func

    def log(self, message_type, content):
        """Sends data to GUI if attached, else prints."""
        if hasattr(self, 'gui_callback') and self.gui_callback:
            self.gui_callback(message_type, content)
        else:
            print(f"[{message_type}] {content}")

    # Then, replace your print() statements with self.log()
    # Example:
    # self.log("TRANSCRIPT", f"User: {text}")
    # self.log("AROUSAL", arousal_value)
