try:
    from ._cmsgpack import Packer, unpackb, Unpacker
except ImportError:
    from .fallback import Packer, unpackb, Unpacker


