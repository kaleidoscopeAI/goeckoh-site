unknown = 'UNKNOWN'

def __init__(self, minval=0, maxval=100):
    assert maxval is None or maxval >= minval
    self.min = self.cur = minval
    self.max = maxval
    self.started = None
    self.elapsed = 0
    self.done = False

def update(self, curval):
    assert self.min <= curval
    assert self.max is None or curval <= self.max
    self.cur = curval
    now = time.time()
    if self.started is None:
        self.started = now
    else:
        self.elapsed = now - self.started

def increment(self, incr):
    assert incr >= 0
    self.update(self.cur + incr)

def start(self):
    self.update(self.min)
    return self

def stop(self):
    if self.max is not None:
        self.update(self.max)
    self.done = True

@property
def maximum(self):
    return self.unknown if self.max is None else self.max

@property
def percentage(self):
    if self.done:
        result = '100 %'
    elif self.max is None:
        result = ' ?? %'
    else:
        v = 100.0 * (self.cur - self.min) / (self.max - self.min)
        result = '%3d %%' % v
    return result

def format_duration(self, duration):
    if (duration <= 0) and self.max is None or self.cur == self.min:
        result = '??:??:??'
    # elif duration < 1:
    #     result = '--:--:--'
    else:
        result = time.strftime('%H:%M:%S', time.gmtime(duration))
    return result

@property
def ETA(self):
    if self.done:
        prefix = 'Done'
        t = self.elapsed
        # import pdb; pdb.set_trace()
    else:
        prefix = 'ETA '
        if self.max is None:
            t = -1
        elif self.elapsed == 0 or (self.cur == self.min):
            t = 0
        else:
            # import pdb; pdb.set_trace()
            t = float(self.max - self.min)
            t /= self.cur - self.min
            t = (t - 1) * self.elapsed
    return '%s: %s' % (prefix, self.format_duration(t))

@property
def speed(self):
    if self.elapsed == 0:
        result = 0.0
    else:
        result = (self.cur - self.min) / self.elapsed
    for unit in UNITS:
        if result < 1000:
            break
        result /= 1000.0
    return '%d %sB/s' % (result, unit)


