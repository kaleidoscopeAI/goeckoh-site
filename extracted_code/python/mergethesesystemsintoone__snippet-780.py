"""Handles decompilation of binary files into readable code"""

def __init__(self, work_dir: str = None):
    self.work_dir = work_dir or config.DECOMPILED_DIR
    os.makedirs(self.work_dir, exist_ok=True)

def decompile_binary(self, file_path: str, 
                     strategies: List[DecompStrategy] = None) -> List[str]:
    """
    Decompile a binary file using multiple strategies

    Args:
        file_path: Path to binary file
        strategies: List of decompilation strategies to try

    Returns:
        List of paths to decompiled files
    """
    if strategies is None:
        strategies = [
            DecompStrategy.RADARE2,
            DecompStrategy.RETDEC
        ]

    # Create a unique directory for this binary
    file_hash = self._hash_file(file_path)
    binary_name = os.path.basename(file_path)
    output_dir = os.path.join(self.work_dir, f"{binary_name}_{file_hash[:8]}")
    os.makedirs(output_dir, exist_ok=True)

    decompiled_files = []

    # Try each strategy
    for strategy in strategies:
        try:
            result_file = self._decompile_with_strategy(file_path, strategy, output_dir)
            if result_file and os.path.exists(result_file):
                decompiled_files.append(result_file)
                logger.info(f"Successfully decompiled {file_path} using {strategy.value}")
        except Exception as e:
            logger.error(f"Failed to decompile {file_path} using {strategy.value}: {str(e)}")

    if not decompiled_files:
        logger.warning(f"All decompilation strategies failed for {file_path}")

    return decompiled_files

def _hash_file(self, file_path: str) -> str:
    """Create a hash of file contents for unique identification"""
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hasher.update(chunk)
    return hasher.hexdigest()

def _decompile_with_strategy(self, file_path: str, 
                             strategy: DecompStrategy, 
                             output_dir: str) -> Optional[str]:
    """
    Decompile binary using a specific strategy

    Args:
        file_path: Path to binary file
        strategy: Decompilation strategy
        output_dir: Directory to store output

    Returns:
        Path to decompiled file if successful, None otherwise
    """
    if strategy == DecompStrategy.RADARE2:
        return self._decompile_with_radare2(file_path, output_dir)
    elif strategy == DecompStrategy.RETDEC:
        return self._decompile_with_retdec(file_path, output_dir)
    elif strategy == DecompStrategy.GHIDRA:
        return self._decompile_with_ghidra(file_path, output_dir)
    else:
        logger.error(f"Unsupported decompilation strategy: {strategy.value}")
        return None

def _decompile_with_radare2(self, file_path: str, output_dir: str) -> Optional[str]:
    """Decompile using radare2"""
    output_file = os.path.join(output_dir, "radare2_decompiled.c")

    # Create a radare2 script
    script_file = os.path.join(output_dir, "r2_script.txt")
    with open(script_file, 'w') as f:
        f.write("aaa\n")  # Analyze all
        f.write("s main\n")  # Seek to main
        f.write("pdf\n")  # Print disassembly function
        f.write("s sym.main\n")  # Alternative main symbol
        f.write("pdf\n")
        f.write("pdc\n")  # Print decompiled code

    try:
        # Run radare2 with the script
        output = subprocess.check_output(
            [config.RADARE2_PATH, "-q", "-i", script_file, file_path],
            stderr=subprocess.PIPE,
            universal_newlines=True
        )

        with open(output_file, 'w') as f:
            f.write("// Decompiled with radare2\n")
            f.write("// Command: r2 -q -i script.txt " + file_path + "\n\n")
            f.write(output)

        return output_file
    except Exception as e:
        logger.error(f"Radare2 decompilation failed: {str(e)}")
        return None

def _decompile_with_retdec(self, file_path: str, output_dir: str) -> Optional[str]:
    """Decompile using RetDec"""
    output_file = os.path.join(output_dir, "retdec_decompiled.c")

    try:
        # Run RetDec
        subprocess.run(
            [config.RETDEC_PATH, file_path, "-o", output_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )

        return output_file
    except Exception as e:
        logger.error(f"RetDec decompilation failed: {str(e)}")
        return None

def _decompile_with_ghidra(self, file_path: str, output_dir: str) -> Optional[str]:
    """Decompile using Ghidra (requires Ghidra installation)"""
    output_file = os.path.join(output_dir, "ghidra_decompiled.c")

    # This is a simplified version - actual Ghidra integration requires more setup
    try:
        ghidra_path = config.GHIDRA_PATH
        headless_path = os.path.join(ghidra_path, "support", "analyzeHeadless")

        if not os.path.exists(headless_path):
            logger.error(f"Ghidra headless analyzer not found at {headless_path}")
            return None

        project_dir = os.path.join(output_dir, "ghidra_project")
        os.makedirs(project_dir, exist_ok=True)

        # Run Ghidra headless analyzer
        subprocess.run(
            [
                headless_path,
                project_dir,
                "UnravelProject",
                "-import", file_path,
                "-postScript", "DecompileScript.java",
                "-scriptPath", os.path.join(ghidra_path, "scripts"),
                "-noanalysis"
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )

        # Look for the decompiled file
        for root, _, files in os.walk(project_dir):
            for file in files:
                if file.endswith(".c") or file.endswith(".cpp"):
                    found_file = os.path.join(root, file)
                    shutil.copy(found_file, output_file)
                    return output_file

        logger.error("Ghidra decompilation completed but no output file found")
        return None
    except Exception as e:
        logger.error(f"Ghidra decompilation failed: {str(e)}")
        return None

