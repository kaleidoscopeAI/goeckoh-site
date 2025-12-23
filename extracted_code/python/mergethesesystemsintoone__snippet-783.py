"""Base class for code transformers"""

@abstractmethod
def can_transform(self, code_file: CodeFile) -> bool:
    """Check if this transformer can handle the given file"""
    pass

@abstractmethod
def transform(self, code_file: CodeFile, system_info: SystemInfo) -> Tuple[str, List[str]]:
    """
    Transform the code

    Args:
        code_file: Code file to transform
        system_info: System information

    Returns:
        Tuple of (transformed code, list of applied transformations)
    """
    pass

