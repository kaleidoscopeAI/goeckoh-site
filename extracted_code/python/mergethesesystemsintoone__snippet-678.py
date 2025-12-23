"""
Represents a memory bank for a node, storing data with a limited capacity.
"""
def __init__(self, capacity: int = 100):
    self.capacity = capacity
    self.memory = deque(maxlen=capacity)

def add_data(self, data: Any):
    """
    Adds data to the memory bank.
    """
    self.memory.append(data)

def retrieve_data(self, num_items: int) -> list:
    """
    Retrieves a specified number of items from the memory, prioritizing recent data.
    """
    return list(self.memory)[-num_items:]

def get_size(self):
    """
    Returns the current size of the memory bank.
    """
    return len(self.memory)

def clear(self):
    """
    Clears all data from the memory bank.
    """
    self.memory.clear()

