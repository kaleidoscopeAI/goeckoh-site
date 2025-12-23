core, history = demo_run(steps=200, dt=0.01, noisy=True)
# save a quick textual summary
import json
print("Final diagnostics:", history[-1]["diag"])
with open("hybrid_demo_history.json", "w") as f:
    # serialize numeric arrays approximately
    out = [{"t":h["t"], "diag":h["diag"], "p":[float(x) for x in h["p"]]} for h in history]
    json.dump(out, f, indent=2)
print("Wrote hybrid_demo_history.json")
