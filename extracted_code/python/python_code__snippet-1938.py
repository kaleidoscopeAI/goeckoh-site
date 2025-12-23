pip_pkg_dir = os.path.join(os.path.dirname(__file__), "..", "..")
pip_pkg_dir = os.path.abspath(pip_pkg_dir)

return f"pip {__version__} from {pip_pkg_dir} (python {get_major_minor_version()})"


