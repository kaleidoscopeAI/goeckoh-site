"""Red Hat patches ``sys.prefix`` and ``sys.exec_prefix``.

Red Hat's ``00251-change-user-install-location.patch`` changes the install
command's ``prefix`` and ``exec_prefix`` to append ``"/local"``. This is
(fortunately?) done quite unconditionally, so we create a default command
object without any configuration to detect this.
"""
from distutils.command.install import install
from distutils.dist import Distribution

cmd: Any = install(Distribution())
cmd.finalize_options()
return (
    cmd.exec_prefix == f"{os.path.normpath(sys.exec_prefix)}/local"
    and cmd.prefix == f"{os.path.normpath(sys.prefix)}/local"
)


