def launch_avogadro(file_path):
    """Open the generated PDB file in Avogadro2."""
    print(f"Launching Avogadro2 with {file_path}")
    try:
        subprocess.run(["avogadro2", file_path], check=True)
    except FileNotFoundError:
        print("Error: Avogadro2 is not installed or not in your PATH.")
    except Exception as e:
        print(f"An error occurred: {e}")

