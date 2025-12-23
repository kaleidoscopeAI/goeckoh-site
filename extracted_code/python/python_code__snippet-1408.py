value_converters = dict(BaseConfigurator.value_converters)
value_converters['inc'] = 'inc_convert'

def __init__(self, config, base=None):
    super(Configurator, self).__init__(config)
    self.base = base or os.getcwd()

def configure_custom(self, config):

    def convert(o):
        if isinstance(o, (list, tuple)):
            result = type(o)([convert(i) for i in o])
        elif isinstance(o, dict):
            if '()' in o:
                result = self.configure_custom(o)
            else:
                result = {}
                for k in o:
                    result[k] = convert(o[k])
        else:
            result = self.convert(o)
        return result

    c = config.pop('()')
    if not callable(c):
        c = self.resolve(c)
    props = config.pop('.', None)
    # Check for valid identifiers
    args = config.pop('[]', ())
    if args:
        args = tuple([convert(o) for o in args])
    items = [(k, convert(config[k])) for k in config if valid_ident(k)]
    kwargs = dict(items)
    result = c(*args, **kwargs)
    if props:
        for n, v in props.items():
            setattr(result, n, convert(v))
    return result

def __getitem__(self, key):
    result = self.config[key]
    if isinstance(result, dict) and '()' in result:
        self.config[key] = result = self.configure_custom(result)
    return result

def inc_convert(self, value):
    """Default converter for the inc:// protocol."""
    if not os.path.isabs(value):
        value = os.path.join(self.base, value)
    with codecs.open(value, 'r', encoding='utf-8') as f:
        result = json.load(f)
    return result


