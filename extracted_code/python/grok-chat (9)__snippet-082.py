  │ import os
  │ os.environ['USE_HEADLESS']='1'
  │ … +14 lines
  └ playsound is relying on another python subprocess. Please use `pip install pygobject` if you want playsound to run more
    efficiently.
    … +4 lines
      File "<stdin>", line 12, in <module>
    TypeError: CompleteUnifiedSystem.__init__() takes 1 positional argument but 3 were given

