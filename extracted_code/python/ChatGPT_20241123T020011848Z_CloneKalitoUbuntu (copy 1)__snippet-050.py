from cryptography.fernet import Fernet

class DataSecurity:
    def __init__(self):
        self.key = Fernet.generate_key()
        self.cipher = Fernet(self.key)

    def encrypt_message(self, message):
        """Encrypt a message."""
        return self.cipher.encrypt(message.encode())

    def decrypt_message(self, encrypted_message):
        """Decrypt a message."""
        return self.cipher.decrypt(encrypted_message).decode()

