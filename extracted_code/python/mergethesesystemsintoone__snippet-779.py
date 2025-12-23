"""Results from analyzing a software artifact"""
software_id: str
file_path: str
file_type: FileType
status: str
decompiled_files: List[str] = field(default_factory=list)
spec_files: List[str] = field(default_factory=list)
reconstructed_files: List[str] = field(default_factory=list)
functions: List[Dict[str, Any]] = field(default_factory=list)
classes: List[Dict[str, Any]] = field(default_factory=list)
dependencies: List[Dict[str, Any]] = field(default_factory=list)
metrics: Dict[str, Any] = field(default_factory=dict)
graph: Optional[nx.DiGraph] = None
error: Optional[str] = None

def to_dict(self) -> Dict[str, Any]:
    """Convert to dictionary for serialization"""
    result = asdict(self)
    # Convert non-serializable types
    result["file_type"] = self.file_type.value
    if self.graph:
        # Convert graph to adjacency list
        result["graph"] = nx.node_link_data(self.graph)
    return result

@classmethod
def from_dict(cls, data: Dict[str, Any]) -> 'AnalysisResult':
    """Create from dictionary"""
    # Convert string enum values to actual enums
    if "file_type" in data:
        data["file_type"] = FileType(data["file_type"])
    # Convert adjacency list to graph
    if "graph" in data and data["graph"]:
        graph_data = data.pop("graph")
        graph = nx.node_link_graph(graph_data)
        return cls(**data, graph=graph)
    return cls(**data)

