def run_simulation(steps: int, nodes_count: int, cfg: Config, seed: int = 0, snapshot_every: int = 10):
    nodes = create_nodes(nodes_count, cfg, seed=seed)
    history = []
    for t in range(steps):
        step_update(nodes, cfg, t)
        if t % snapshot_every == 0:
            # capture summary statistics
            avg_U = float(np.mean([np.linalg.norm(n.U) for n in nodes]))
            avg_sigma = float(np.mean([np.linalg.norm(n.sigma) for n in nodes if n.sigma is not None]))
            avg_M = float(np.mean([np.linalg.norm(n.M) for n in nodes]))
            history.append({'t': t, 'avg_U': avg_U, 'avg_sigma': avg_sigma, 'avg_M': avg_M})
            print(f"t={t:04d} avg_U={avg_U:.4f} avg_sigma={avg_sigma:.4f} avg_M={avg_M:.4f}")
    return nodes, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=200, help='simulation steps')
    parser.add_argument('--nodes', type=int, default=32, help='number of nodes')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    args = parser.parse_args()

    cfg = Config()
    run_simulation(args.steps, args.nodes, cfg, seed=args.seed)


