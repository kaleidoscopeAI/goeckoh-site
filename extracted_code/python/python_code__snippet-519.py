import json
import os
import os.path
import re
import shutil
import sys
import traceback
from glob import glob
from importlib import import_module
from os.path import join as pjoin

# This file is run as a script, and `import wrappers` is not zip-safe, so we
# include write_json() and read_json() from wrappers.py.


def write_json(obj, path, **kwargs):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, **kwargs)


def read_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


class BackendUnavailable(Exception):
    """Raised if we cannot import the backend"""
    def __init__(self, traceback):
        self.traceback = traceback


class BackendInvalid(Exception):
    """Raised if the backend is invalid"""
    def __init__(self, message):
        self.message = message


class HookMissing(Exception):
    """Raised if a hook is missing and we are not executing the fallback"""
    def __init__(self, hook_name=None):
        super().__init__(hook_name)
        self.hook_name = hook_name


def contained_in(filename, directory):
    """Test if a file is located within the given directory."""
    filename = os.path.normcase(os.path.abspath(filename))
    directory = os.path.normcase(os.path.abspath(directory))
    return os.path.commonprefix([filename, directory]) == directory


def _build_backend():
    """Find and load the build backend"""
    # Add in-tree backend directories to the front of sys.path.
    backend_path = os.environ.get('PEP517_BACKEND_PATH')
    if backend_path:
        extra_pathitems = backend_path.split(os.pathsep)
        sys.path[:0] = extra_pathitems

    ep = os.environ['PEP517_BUILD_BACKEND']
    mod_path, _, obj_path = ep.partition(':')
    try:
        obj = import_module(mod_path)
    except ImportError:
        raise BackendUnavailable(traceback.format_exc())

    if backend_path:
        if not any(
            contained_in(obj.__file__, path)
            for path in extra_pathitems
        ):
            raise BackendInvalid("Backend was not loaded from backend-path")

    if obj_path:
        for path_part in obj_path.split('.'):
            obj = getattr(obj, path_part)
    return obj


def _supported_features():
    """Return the list of options features supported by the backend.

    Returns a list of strings.
    The only possible value is 'build_editable'.
    """
    backend = _build_backend()
    features = []
    if hasattr(backend, "build_editable"):
        features.append("build_editable")
    return features


def get_requires_for_build_wheel(config_settings):
    """Invoke the optional get_requires_for_build_wheel hook

    Returns [] if the hook is not defined.
    """
    backend = _build_backend()
    try:
        hook = backend.get_requires_for_build_wheel
    except AttributeError:
        return []
    else:
        return hook(config_settings)


def get_requires_for_build_editable(config_settings):
    """Invoke the optional get_requires_for_build_editable hook

    Returns [] if the hook is not defined.
    """
    backend = _build_backend()
    try:
        hook = backend.get_requires_for_build_editable
    except AttributeError:
        return []
    else:
        return hook(config_settings)


def prepare_metadata_for_build_wheel(
        metadata_directory, config_settings, _allow_fallback):
    """Invoke optional prepare_metadata_for_build_wheel

    Implements a fallback by building a wheel if the hook isn't defined,
    unless _allow_fallback is False in which case HookMissing is raised.
    """
    backend = _build_backend()
    try:
        hook = backend.prepare_metadata_for_build_wheel
    except AttributeError:
        if not _allow_fallback:
            raise HookMissing()
    else:
        return hook(metadata_directory, config_settings)
    # fallback to build_wheel outside the try block to avoid exception chaining
    # which can be confusing to users and is not relevant
    whl_basename = backend.build_wheel(metadata_directory, config_settings)
    return _get_wheel_metadata_from_wheel(whl_basename, metadata_directory,
                                          config_settings)


def prepare_metadata_for_build_editable(
        metadata_directory, config_settings, _allow_fallback):
    """Invoke optional prepare_metadata_for_build_editable

    Implements a fallback by building an editable wheel if the hook isn't
    defined, unless _allow_fallback is False in which case HookMissing is
    raised.
    """
    backend = _build_backend()
    try:
        hook = backend.prepare_metadata_for_build_editable
    except AttributeError:
        if not _allow_fallback:
            raise HookMissing()
        try:
            build_hook = backend.build_editable
        except AttributeError:
            raise HookMissing(hook_name='build_editable')
        else:
            whl_basename = build_hook(metadata_directory, config_settings)
            return _get_wheel_metadata_from_wheel(whl_basename,
                                                  metadata_directory,
                                                  config_settings)
    else:
        return hook(metadata_directory, config_settings)


