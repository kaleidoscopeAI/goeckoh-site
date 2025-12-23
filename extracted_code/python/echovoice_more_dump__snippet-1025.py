"""Unified Abstract Syntax Tree"""

def __init__(self, root: Optional[UnifiedNode] = None):
    """
    Initialize the AST

    Args:
        root: Root node
    """
    self.root = root or UnifiedNode(type=NodeType.ROOT, name="root")

def find_by_type(self, node_type: NodeType) -> List[UnifiedNode]:
    """
    Find all nodes of a specific type

    Args:
        node_type: Node type to find

    Returns:
        List of matching nodes
    """
    return self.root.find_by_type(node_type)

def find_by_name(self, name: str) -> List[UnifiedNode]:
    """
    Find all nodes with a specific name

    Args:
        name: Node name to find

    Returns:
        List of matching nodes
    """
    return self.root.find_by_name(name)

def find_by_path(self, path: List[str]) -> Optional[UnifiedNode]:
    """
    Find a node by path

    Args:
        path: Path components

    Returns:
        Matching node or None
    """
    return self.root.find_by_path(path)

def to_dict(self) -> Dict[str, Any]:
    """
    Convert to dictionary

    Returns:
        Dictionary representation
    """
    return self.root.to_dict()

@classmethod
def from_dict(cls, data: Dict[str, Any]) -> 'UnifiedAST':
    """
    Create from dictionary

    Args:
        data: Dictionary representation

    Returns:
        Created AST
    """
    root = UnifiedNode.from_dict(data)
    return cls(root)

def to_json(self) -> str:
    """
    Convert to JSON

    Returns:
        JSON representation
    """
    return self.root.to_json()

@classmethod
def from_json(cls, json_str: str) -> 'UnifiedAST':
    """
    Create from JSON

    Args:
        json_str: JSON representation

    Returns:
        Created AST
    """
    root = UnifiedNode.from_json(json_str)
    return cls(root)

def merge(self, other: 'UnifiedAST') -> 'UnifiedAST':
    """
    Merge with another AST

    Args:
        other: Other AST

    Returns:
        Merged AST
    """
    # Create a new AST
    merged = UnifiedAST()

    # Clone this AST's nodes
    for child in self.root.children:
        merged.root.add_child(_clone_node(child))

    # Add other AST's nodes
    for child in other.root.children:
        merged.root.add_child(_clone_node(child))

    return merged

