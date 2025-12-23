    from adaptive_ai_node import EnhancedAdaptiveNode
  File "/home/studio/Desktop/node-demo/adaptive_ai_node.py", line 17, in <module>
    @dataclass
     ^^^^^^^^^
  File "/usr/lib/python3.12/dataclasses.py", line 1268, in dataclass
    return wrap(cls)
           ^^^^^^^^^
  File "/usr/lib/python3.12/dataclasses.py", line 1258, in wrap
    return _process_class(cls, init, repr, eq, order, unsafe_hash,
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/dataclasses.py", line 994, in _process_class
    cls_fields.append(_get_field(cls, name, type, kw_only))
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/dataclasses.py", line 852, in _get_field
    raise ValueError(f'mutable default {type(f.default)} for field '
ValueError: mutable default <class 'list'> for field state_history is not allowed: use default_factory

