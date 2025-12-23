# Don't close underlying buffer on destruction.
def close(self):
    self.flush()


