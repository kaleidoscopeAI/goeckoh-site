import importlib.machinery as importlib_machinery

# access attribute to force import under delayed import mechanisms.
importlib_machinery.__name__
