# prevent circular import
from pip._vendor.rich.pretty import install
from pip._vendor.rich.traceback import install as tr_install

install()
tr_install()


