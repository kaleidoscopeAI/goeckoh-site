print("Downloading 'en_core_web_sm' model for spaCy. This may take a few minutes...")
import subprocess
subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])

