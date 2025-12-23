"""Format an error message for an OSError

It may occur anytime during the execution of the install command.
"""
parts = []

# Mention the error if we are not going to show a traceback
parts.append("Could not install packages due to an OSError")
if not show_traceback:
    parts.append(": ")
    parts.append(str(error))
else:
    parts.append(".")

# Spilt the error indication from a helper message (if any)
parts[-1] += "\n"

# Suggest useful actions to the user:
#  (1) using user site-packages or (2) verifying the permissions
if error.errno == errno.EACCES:
    user_option_part = "Consider using the `--user` option"
    permissions_part = "Check the permissions"

    if not running_under_virtualenv() and not using_user_site:
        parts.extend(
            [
                user_option_part,
                " or ",
                permissions_part.lower(),
            ]
        )
    else:
        parts.append(permissions_part)
    parts.append(".\n")

# Suggest the user to enable Long Paths if path length is
# more than 260
if (
    WINDOWS
    and error.errno == errno.ENOENT
    and error.filename
    and len(error.filename) > 260
):
    parts.append(
        "HINT: This error might have occurred since "
        "this system does not have Windows Long Path "
        "support enabled. You can find information on "
        "how to enable this at "
        "https://pip.pypa.io/warnings/enable-long-paths\n"
    )

return "".join(parts).strip() + "\n"


