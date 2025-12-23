class SystemInfo:
    """Information about the system to upgrade"""
    root_path: str
    system_type: SystemType
    primary_language: LanguageType
    other_languages: List[LanguageType] = field.default_factory.list)
    files: Dict[str, CodeFile] = field.default_factory.dict)
    dependencies: Dict[str, DependencyInfo] = field.default_factory.dict)
    entry_points: List[str] = field.default_factory.list)
    config_files: List[str] = field.default_factory.list)
    database_info: Dict[str, Any] = field.default_factory.dict)
    api_endpoints: List[str] = field.default_factory.list)
    vulnerabilities: List[str] = field.default_factory.list)
    dependencies_graph: Optional[nx.DiGraph] = None
    file_count: int = 0
    code_size: int = 0  # In bytes
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            "root_path": self.root_path,
            "system_type": self.system_type.name,
            "primary_language": self.primary_language.name,
            "other_languages": [lang.name for lang in self.other_languages],
            "entry_points": self.entry_points,
            "config_files": self.config_files,
            "database_info": self.database_info,
            "api_endpoints": self.api_endpoints,
            "vulnerabilities": self.vulnerabilities,
            "file_count": self.file_count,
            "code_size": self.code_size,
            "dependencies": {k: {"name": v.name, "version": v.version} for k, v in self.dependencies.items()},
        }
        return result

