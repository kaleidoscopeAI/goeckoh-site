import numpy as np
from server import snapshots

def replay_last(n=100):
    for idx, snap in enumerate(list(snapshots)[-n:]):
        # run deterministic analysis on snapshot for testing
        meanr = np.mean(np.linalg.norm(snap, axis=1))
        print(idx, meanr)

if __name__ == '__main__':
    replay_last(10)
