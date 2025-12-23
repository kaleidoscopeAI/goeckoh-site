"""
Mixin for running subprocesses and capturing their output
"""

def __init__(self, verbose=False, progress=None):
    self.verbose = verbose
    self.progress = progress

def reader(self, stream, context):
    """
    Read lines from a subprocess' output stream and either pass to a progress
    callable (if specified) or write progress information to sys.stderr.
    """
    progress = self.progress
    verbose = self.verbose
    while True:
        s = stream.readline()
        if not s:
            break
        if progress is not None:
            progress(s, context)
        else:
            if not verbose:
                sys.stderr.write('.')
            else:
                sys.stderr.write(s.decode('utf-8'))
            sys.stderr.flush()
    stream.close()

def run_command(self, cmd, **kwargs):
    p = subprocess.Popen(cmd,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE,
                         **kwargs)
    t1 = threading.Thread(target=self.reader, args=(p.stdout, 'stdout'))
    t1.start()
    t2 = threading.Thread(target=self.reader, args=(p.stderr, 'stderr'))
    t2.start()
    p.wait()
    t1.join()
    t2.join()
    if self.progress is not None:
        self.progress('done.', 'main')
    elif self.verbose:
        sys.stderr.write('done.\n')
    return p


