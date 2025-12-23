"""Base class for language adapters"""

@property
@abstractmethod
def language(self) -> str:
    """Language identifier"""
    pass

@property
@abstractmethod
def file_extensions(self) -> List[str]:
    """File extensions handled by this adapter"""
    pass

@abstractmethod
async def parse(self, code: str, file

