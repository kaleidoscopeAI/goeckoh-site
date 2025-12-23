def __init__(self, autistic_mode=False, **kwargs):
    self.params = {
        'f0_avg': kwargs.get('f0_avg', 120),
        'tract_scale': kwargs.get('tract_scale', 1.0),
        'speed': kwargs.get('speed', 1.0),
        'jitter': kwargs.get('jitter', 0.005),
        'shimmer': kwargs.get('shimmer', 0.05),
        'breath': kwargs.get('breath', 0.05),
        'int_range': kwargs.get('int_range', 10),
        'vibrato_rate': kwargs.get('vibrato_rate', 5.0),
    }
    if autistic_mode:
        self.params['int_range'] = 2
        self.params['vibrato_rate'] = 0.5
        self.params['jitter'] = 0.002
        self.params['speed'] = 1.2
        logging.info("Autistic prosody applied.")

