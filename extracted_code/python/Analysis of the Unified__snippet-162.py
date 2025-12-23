import sys
if len(sys.argv) < 2:
    print("Usage: python hybrid_viz.py <history_json>")
    sys.exit(1)
fname = sys.argv[1]
with open(fname, "r") as f:
    history = json.load(f)
# history entries expected: {t, diag, p}
plot_history(history, save_prefix="hybrid_run")
print("Done")
