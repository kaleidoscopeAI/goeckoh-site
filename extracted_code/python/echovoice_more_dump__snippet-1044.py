def __init__(self) -> None:
    self.objects: set = set()
    self.morphisms: Dict[Tuple[Any, Any], List[Callable]] = {}

def add_object(self, obj: Any) -> None:
    self.objects.add(obj)

def add_morphism(self, a: Any, b: Any, morphism: Callable) -> None:
    self.morphisms.setdefault((a, b), []).append(morphism)

