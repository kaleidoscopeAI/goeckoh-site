 2  import math
 3 +import time
 4  from dataclasses import dataclass
   ⋮
21          self.logger = SessionLog()
22 +        self.gcl_history: list[float] = []
23 +        self.gcl_ts: list[float] = []
24
   ⋮
66          self.coherence_history.append(coherence)
67 +        self.gcl_history.append(gcl)
68 +        self.gcl_ts.append(time.time())
69
70          # Keep history bounded
66 -        max_history = 100
71 +        max_history = 300
72          if len(self.arousal_history) > max_history:
   ⋮
75              self.coherence_history = self.coherence_history[-max_history
    :]
76 +            self.gcl_history = self.gcl_history[-max_history:]
77 +            self.gcl_ts = self.gcl_ts[-max_history:]
78

