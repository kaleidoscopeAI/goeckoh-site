"""Register `provider_factory` to make providers for `loader_type`

`loader_type` is the type or class of a PEP 302 ``module.__loader__``,
and `provider_factory` is a function that, passed a *module* object,
returns an ``IResourceProvider`` for that module.
"""
_provider_factories[loader_type] = provider_factory


