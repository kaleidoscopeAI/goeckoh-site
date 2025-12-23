import os, subprocess, platform, venv, shutil

def run(cmd):
    subprocess.check_call(cmd, shell=True)

