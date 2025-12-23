"""Register `namespace_handler` to declare namespace packages

`importer_type` is the type or class of a PEP 302 "Importer" (sys.path item
handler), and `namespace_handler` is a callable like this::

    def namespace_handler(importer, path_entry, moduleName, module):
        # return a path_entry to use for child packages

Namespace handlers are only called if the importer object has already
agreed that it can handle the relevant path item, and they should only
return a subpath if the module __path__ does not already contain an
equivalent subpath.  For an example namespace handler, see
``pkg_resources.file_ns_handler``.
"""
_namespace_handlers[importer_type] = namespace_handler


