"""
Unified Autonomous AI — Sandboxed Simulation

This is a self-contained, *safe* simulation script implementing the mathematical
unified system (Metamorphosis + Unravel + Mimicry + Roles + Data Ingestion)
for research and testing in a strictly offline/sandboxed environment.

CRITICAL: This script intentionally contains NO real networking, NO web
crawling, and NO offensive capabilities. All "data ingestion" is mocked.
Use for simulations, tuning, visualization, and algorithmic development only.

Usage: python unified_sim.py --steps 100 --nodes 32 --seed 42

"""
import argparse
import math
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# --------------------------- Utilities -------------------------------------

def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a) + 1e-12
    nb = np.linalg.norm(b) + 1e-12
    return float(np.dot(a, b) / (na * nb))


# --------------------------- Config ----------------------------------------

@dataclass
class Config:
    dims: int = 64
    eta: float = 0.1
    lambda_phi: float = 0.2
    gamma: float = 0.5
    delta: float = 0.05
    lambda_M: float = 0.1
    lambda_D: float = 0.05
    lambda_U: float = 0.02
    alpha: float = 1.0
    beta: float = 1.0
    kappa: float = 0.1
    theta: float = 0.1
    zeta: float = 0.05
    noise_scale: float = 0.01


# --------------------------- Node Definition -------------------------------

@dataclass
class Node:
    id: int
    X: np.ndarray  # low-level observation/action vector
    S: float  # confidence scalar
    E: float  # energy scalar
    K: np.ndarray  # knowledge embedding
    Psi: np.ndarray  # perspective embedding
    U: np.ndarray  # unravel vector
    M: np.ndarray  # mimicry embedding
    R: str  # role: 'red'|'blue'|'crawler'|'analyzer'
    D: np.ndarray  # data embedding

    # runtime / derived
    sigma: Optional[np.ndarray] = field(default=None)
    Phi: Optional[np.ndarray] = field(default=None)
    Spec: Optional[np.ndarray] = field(default=None)


# --------------------------- Mocked Encoders / Sources ---------------------

def mock_encode_text_to_vec(text: str, dims: int) -> np.ndarray:
    # deterministic pseudo-encoder for reproducibility
    rnd = np.frombuffer(text.encode('utf8'), dtype=np.uint8)
    h = np.zeros(dims, dtype=float)
    for i, v in enumerate(rnd):
        h[i % dims] += (v + 1) * 0.001
    # normalize
    n = np.linalg.norm(h) + 1e-12
    return h / n


def mock_crawl_source(node: Node, cfg: Config, step: int) -> np.ndarray:
    # produce a synthetic 'crawl' delta X based on role and step
    seed = int(node.id * 1009 + step * 97)
    rng = np.random.RandomState(seed)
    delta = rng.normal(scale=0.1, size=cfg.dims)
    # small signal depending on role
    role_signal = {'red': -0.2, 'blue': 0.2, 'crawler': 0.5, 'analyzer': 0.1}
    delta += role_signal.get(node.R, 0.0)
    return delta


# --------------------------- Interaction Functions ------------------------

def role_rho(a: str, b: str) -> float:
    # cooperative if same role and blue/blue bias, adversarial if red vs blue
    if a == b:
        return 1.0
    if (a == 'red' and b == 'blue') or (a == 'blue' and b == 'red'):
        return -1.0
    return 0.0


def compute_B_matrix(nodes: List[Node], cfg: Config) -> np.ndarray:
    n = len(nodes)
    B = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            KiKj = np.dot(nodes[i].K, nodes[j].K)
            DiDj = np.dot(nodes[i].D, nodes[j].D)
            denom = (np.linalg.norm(nodes[i].K) * np.linalg.norm(nodes[j].K)
                     + np.linalg.norm(nodes[i].D) * np.linalg.norm(nodes[j].D) + 1e-12)
            g = (KiKj + DiDj) / denom
            psi_diff = np.linalg.norm(nodes[i].Psi - nodes[j].Psi)
            g *= math.exp(-cfg.alpha * (psi_diff ** 2))
            rho = role_rho(nodes[i].R, nodes[j].R)
            phi = np.linalg.norm(nodes[i].U - nodes[j].U) / (np.linalg.norm(nodes[i].U) + np.linalg.norm(nodes[j].U) + 1e-12)
            B[i, j] = g * (1 - phi) * rho
    return B


def compute_W_matrix(nodes: List[Node], cfg: Config) -> np.ndarray:
    n = len(nodes)
    W = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            ck = cos_sim(nodes[i].K, nodes[j].K)
            cp = cos_sim(nodes[i].Psi, nodes[j].Psi)
            cd = cos_sim(nodes[i].D, nodes[j].D)
            w = (ck + cp + cd) / 3.0
            w *= math.exp(-cfg.beta * np.linalg.norm(nodes[j].U))
            # role bias: prefer learning from same-role or trusted
            role_bias = 1.0 if nodes[i].R == nodes[j].R else 0.5
            W[i, j] = w * role_bias
    return W


# --------------------------- Engine Primitives ----------------------------

def decoherence(node: Node) -> float:
    # quadratic on misalignment magnitude
    if node.Phi is None:
        return 0.0
    return np.linalg.norm(node.Phi) ** 2


def speculation_operator(node: Node, cfg: Config) -> np.ndarray:
    # simple speculative vector: small nonlinear transform of K + D
    z = np.tanh(node.K + node.D - node.U * 0.01)
    return z * cfg.gamma


# --------------------------- Simulation Step ------------------------------

def step_update(nodes: List[Node], cfg: Config, step: int) -> None:
    n = len(nodes)
    B = compute_B_matrix(nodes, cfg)
    W = compute_W_matrix(nodes, cfg)

    # compute pairwise differences quickly
    Nvecs = np.stack([np.concatenate((node.K, node.Psi, node.D)) for node in nodes])
    # For simplicity, treat N as concatenation of K,Psi,D in dynamics

    # compute gradients / data changes
    delta_D = [mock_crawl_source(node, cfg, step) for node in nodes]

    # compute sigma, Phi, Spec for each node
    for i, node in enumerate(nodes):
        # sigma: sum_j B_ij (N_j - N_i) + kappa U + theta grad D
        Nj_minus_Ni = np.zeros_like(node.K)
        for j in range(n):
            if i == j:
                continue
            Nj_minus_Ni += B[i, j] * (nodes[j].K - node.K)
        node.sigma = Nj_minus_Ni + cfg.kappa * node.U[: node.K.shape[0]] + cfg.theta * delta_D[i]
        # Phi: N - N_hat - gamma_U U
        # Here N_hat is a simple prediction: moving average of neighbors' K
        neighbor_mean = np.mean([nodes[j].K for j in range(n) if j != i], axis=0)
        node.Phi = node.K - neighbor_mean - cfg.lambda_U * node.U[: node.K.shape[0]]
        node.Spec = speculation_operator(node, cfg)

    # update U, M, D, and N (K,Psi,D) according to equations
    for i, node in enumerate(nodes):
        # U update
        psi_sum = np.zeros_like(node.U)
        for j in range(n):
            psi_sum += B[i, j] * nodes[j].U
        node.U = node.U + cfg.lambda_U * (decoherence(node) * np.ones_like(node.U) + psi_sum + cfg.zeta * np.linalg.norm(delta_D[i]))
        # M update
        mimicry_sum = np.zeros_like(node.M)
        for j in range(n):
            mimicry_sum += W[i, j] * (nodes[j].K - node.K)
        node.M = node.M + cfg.lambda_M * mimicry_sum
        # D update (data ingestion) - encode delta into D space
        node.D = node.D + cfg.lambda_D * delta_D[i]

    # Finally update K (as proxy for N) using integrated update
    for i, node in enumerate(nodes):
        stress_term = np.zeros_like(node.K)
        for j in range(n):
            stress_term += B[i, j] * (nodes[j].K - node.K)
        mimicry_term = cfg.lambda_M * sum(W[i, j] * (nodes[j].K - node.K) for j in range(n))
        node.K = (
            node.K
            + cfg.eta * stress_term
            + cfg.lambda_phi * node.Phi
            + cfg.gamma * node.Spec
            - cfg.delta * node.U[: node.K.shape[0]]
            + mimicry_term
        )

    # small normalization / noise to keep stable
    for node in nodes:
        node.K = node.K / (np.linalg.norm(node.K) + 1e-12)
        node.D = node.D / (np.linalg.norm(node.D) + 1e-12 + 1e-12 * random.random())


# --------------------------- Initialization -------------------------------

def create_nodes(count: int, cfg: Config, seed: int = 0) -> List[Node]:
    rng = np.random.RandomState(seed)
    nodes = []
    roles = ['red', 'blue', 'crawler', 'analyzer']
    for i in range(count):
        K = rng.normal(size=cfg.dims)
        K = K / (np.linalg.norm(K) + 1e-12)
        Psi = rng.normal(size=cfg.dims)
        Psi = Psi / (np.linalg.norm(Psi) + 1e-12)
        D = rng.normal(size=cfg.dims)
        D = D / (np.linalg.norm(D) + 1e-12)
        U = rng.normal(scale=0.01, size=cfg.dims)
        M = np.zeros(cfg.dims)
        X = rng.normal(size=cfg.dims)
        node = Node(
            id=i,
            X=X,
            S=1.0,
            E=1.0,
            K=K,
            Psi=Psi,
            U=U,
            M=M,
            R=roles[i % len(roles)],
            D=D,
        )
        nodes.append(node)
    return nodes


# --------------------------- Runner / CLI ---------------------------------

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


if __name__ == '__main__':
    main()
