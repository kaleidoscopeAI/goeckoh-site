"""
Given the mapping from entrypoint name to callable, return the relevant
console script specs.
"""
# Don't mutate caller's version
console = console.copy()

scripts_to_generate = []

# Special case pip and setuptools to generate versioned wrappers
#
# The issue is that some projects (specifically, pip and setuptools) use
# code in setup.py to create "versioned" entry points - pip2.7 on Python
# 2.7, pip3.3 on Python 3.3, etc. But these entry points are baked into
# the wheel metadata at build time, and so if the wheel is installed with
# a *different* version of Python the entry points will be wrong. The
# correct fix for this is to enhance the metadata to be able to describe
# such versioned entry points, but that won't happen till Metadata 2.0 is
# available.
# In the meantime, projects using versioned entry points will either have
# incorrect versioned entry points, or they will not be able to distribute
# "universal" wheels (i.e., they will need a wheel per Python version).
#
# Because setuptools and pip are bundled with _ensurepip and virtualenv,
# we need to use universal wheels. So, as a stopgap until Metadata 2.0, we
# override the versioned entry points in the wheel and generate the
# correct ones. This code is purely a short-term measure until Metadata 2.0
# is available.
#
# To add the level of hack in this section of code, in order to support
# ensurepip this code will look for an ``ENSUREPIP_OPTIONS`` environment
# variable which will control which version scripts get installed.
#
# ENSUREPIP_OPTIONS=altinstall
#   - Only pipX.Y and easy_install-X.Y will be generated and installed
# ENSUREPIP_OPTIONS=install
#   - pipX.Y, pipX, easy_install-X.Y will be generated and installed. Note
#     that this option is technically if ENSUREPIP_OPTIONS is set and is
#     not altinstall
# DEFAULT
#   - The default behavior is to install pip, pipX, pipX.Y, easy_install
#     and easy_install-X.Y.
pip_script = console.pop("pip", None)
if pip_script:
    if "ENSUREPIP_OPTIONS" not in os.environ:
        scripts_to_generate.append("pip = " + pip_script)

    if os.environ.get("ENSUREPIP_OPTIONS", "") != "altinstall":
        scripts_to_generate.append(f"pip{sys.version_info[0]} = {pip_script}")

    scripts_to_generate.append(f"pip{get_major_minor_version()} = {pip_script}")
    # Delete any other versioned pip entry points
    pip_ep = [k for k in console if re.match(r"pip(\d+(\.\d+)?)?$", k)]
    for k in pip_ep:
        del console[k]
easy_install_script = console.pop("easy_install", None)
if easy_install_script:
    if "ENSUREPIP_OPTIONS" not in os.environ:
        scripts_to_generate.append("easy_install = " + easy_install_script)

    scripts_to_generate.append(
        f"easy_install-{get_major_minor_version()} = {easy_install_script}"
    )
    # Delete any other versioned easy_install entry points
    easy_install_ep = [
        k for k in console if re.match(r"easy_install(-\d+\.\d+)?$", k)
    ]
    for k in easy_install_ep:
        del console[k]

# Generate the console entry points specified in the wheel
scripts_to_generate.extend(starmap("{} = {}".format, console.items()))

return scripts_to_generate


