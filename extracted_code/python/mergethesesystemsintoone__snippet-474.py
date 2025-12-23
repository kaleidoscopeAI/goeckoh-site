class SystemUpgrader:
    def __init__(self):
        self.analyzer = SystemAnalyzer()
        self.transformers = [PythonModernizer()]
        self.logger = logging.getLogger(__name__)

    def upgrade_system(self, root_path: str, config: UpgradeConfig) -> Dict[str, Any]:
        start_time = datetime.datetime.now()
        
        system_info = self.analyzer.analyze_system(root_path)
        backup_path = None
        if config.keep_original:
            backup_path = self._create_backup(root_path)
        
        output_path = tempfile.mkdtemp(prefix="kaleidoscope_upgrade_")
        upgraded_files = []
        errors = []
        transformations = []
        
        for file_path, code_file in system_info.files.items():
            transformer = next((t for t in self.transformers if t.can_transform(code_file)), None)
            if transformer:
                try:
                    new_content, file_transforms = transformer.transform(code_file, system_info)
                    output_file = os.path.join(output_path, file_path)
                    os.makedirs(os.path.dirname(output_file), exist_ok=True)
                    with open(output_file, 'w') as f:
                        f.write(new_content)
                    upgraded_files.append(file_path)
                    transformations.extend(file_transforms)
                except Exception as e:
                    errors.append(f"Failed to transform {file_path}: {str(e)}")
        
        time_taken = (datetime.datetime.now() - start_time).total_seconds()
        size_diff = self._calculate_size_difference(root_path, output_path)
        
        result = UpgradeResult(
            success=len(errors) == 0,
            output_path=output_path,
            strategy_used=config.strategy,
            upgraded_files=upgraded_files,
            errors=errors,
            backup_path=backup_path,
            time_taken_seconds=time_taken,
            size_difference=size_diff,
            applied_transformations=transformations
        )
        return result.__dict__

    def _create_backup(self, root_path: str) -> str:
        backup_dir = f"{root_path}_backup_{uuid.uuid4().hex}"
        shutil.copytree(root_path, backup_dir)
        return backup_dir

    def _calculate_size_difference(self, original_path: str, new_path: str) -> int:
        orig_size = sum(os.path.getsize(os.path.join(root, f)) for root, _, files in os.walk(original_path) for f in files)
        new_size = sum(os.path.getsize(os.path.join(root, f)) for root, _, files in os.walk(new_path) for f in files)
        return new_size - orig_size

class PythonModernizer(CodeTransformer):
    def can_transform(self, code_file: CodeFile) -> bool:
        return code_file.language == LanguageType.PYTHON
    
    def transform(self, code_file: CodeFile, system_info: SystemInfo) -> Tuple[str, List[str]]:
        content = code_file.content
        transformations = []
        
        content, type_transforms = self._add_type_hints(content)
        if type_transforms:
            transformations.append("Added type hints")
        
        content, fstring_count = self._convert_to_fstrings(content)
        if fstring_count > 0:
            transformations.append(f"Converted {fstring_count} string formats to f-strings")
        
        content, modern_transforms = self._modernize_python_features(content)
        transformations.extend(modern_transforms)
        
        return content, transformations
    
    def _add_type_hints(self, content: str) -> Tuple[str, List[str]]:
        if 'from typing import ' not in content and 'import typing' not in content:
            content = "from typing import List, Dict, Tuple, Optional, Any, Union\n" + content
            return content, ["Added typing imports"]
        return content, []
    
    def _convert_to_fstrings(self, content: str) -> Tuple[str, int]:
        pattern = r'([\'"].*?[\'"])\s*\.\s*format\s*\((.*?)\)'
        count = 0
        
        def replace_format(match):
            nonlocal count
            string = match.group(1)[1:-1]
            args = match.group(2).strip()
            if re.match(r'^[\w\s,]+$', args):
                vars = [v.strip() for v in args.split(',')]
                new_str = f"f'{string.format(*['{' + v + '}' for v in vars])}'"
                count += 1
                return new_str
            return match.group(0)
        
        content = re.sub(pattern, replace_format, content)
        return content, count
    
    def _modernize_python_features(self, content: str) -> Tuple[str, List[str]]:
        transformations = []
        if re.search(r'^print\s+[^(\n]', content, re.MULTILINE):
            content = re.sub(r'^print\s+(.+)$', r'print(\1)', content, flags=re.MULTILINE)
            transformations.append("Converted print statements to Python 3 style")
        return content, transformations
