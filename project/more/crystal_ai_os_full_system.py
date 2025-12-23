"""
Crystal AI OS - Integrated Reference System Script

This single-file reference contains a runnable, modular implementation of the
core Crystal AI OS components discussed in the conversation. It is organized
as lightweight modules (classes and functions) and is intended as a working
skeleton you can expand and split into files later.

Features included:
- Spec aliases and canonical variable names
- Packed-bit kernels (pack, popcount, xor-popcount)
- HybridState (raw + packed bit storage)
- SemanticHamiltonian with analytic gradient and incremental ΔE for bit flips
- GradientFlow using analytic gradients
- MetropolisEngine with Hajek annealing schedule and local ΔE proposals
- Basic cognitive field scaffolding (PhiCalculator, FreeEnergyEngine, ConfidenceSynthesizer)
- Error correction scaffolding (SyndromeDetector, NeuralDecoder stub, HarmonicMemory)
- Category kernel and axiom verifier placeholders
- Utilities: annealing schedules, metrics (GCL)
- Runtime orchestration and a small demo builder + test

This is intentionally pragmatic: numeric stability, shape checks, sparse neighbor
access, and performance-minded packing are included. Replace stubs with production
implementations (numba/C kernels, learned decoders, exact Φ algorithm) as needed.

"""

from __future__ import annotations

# Standard libs
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

# Numerical
import numpy as np
from scipy.special import expit

# --------------------------- spec (canonical names) ---------------------------
class Spec:
    SelFn = Callable[["HybridState", int], Sequence[int]]
    OperatorCallable = Callable[[Any, dict], Any]
    GateFn = Callable[[float], float]

# --------------------------- hardware_interface.bitops -------------------------
_WORD_BITS = 64

def pack_bits(bit_array: np.ndarray) -> np.ndarray:
    arr = np.asarray(bit_array).astype(np.uint8).ravel()
    if arr.size == 0:
        return np.zeros(0, dtype=np.uint64)
    pad = (-arr.size) % _WORD_BITS
    if pad > 0:
        arr = np.concatenate([arr, np.zeros(pad, dtype=np.uint8)])
    arr_bits = arr.reshape(-1, _WORD_BITS)
    # packbits returns bytes (per row) in big-endian or little; we use little
    bytes_rows = np.packbits(arr_bits, axis=1, bitorder='little')
    words = np.frombuffer(bytes_rows.tobytes(), dtype=np.uint64)
    return words

def popcount_u64(arr: np.ndarray) -> int:
    if arr.size == 0:
        return 0
    # efficient path: use vectorized popcount via numpy's unpackbits on uint8 view
    bytes_view = arr.view(np.uint8)
    bits = np.unpackbits(bytes_view, bitorder='little')
    return int(bits.sum())

def popcount_xor(a_packed: np.ndarray, b_packed: np.ndarray) -> int:
    n = max(a_packed.size, b_packed.size)
    if a_packed.size != n:
        a = np.pad(a_packed, (0, n - a_packed.size), constant_values=0)
    else:
        a = a_packed
    if b_packed.size != n:
        b = np.pad(b_packed, (0, n - b_packed.size), constant_values=0)
    else:
        b = b_packed
    xor = np.bitwise_xor(a, b)
    return popcount_u64(xor)

# --------------------------- hybrid_semantic.state ----------------------------
@dataclass
class HybridState:
    E: Dict[int, np.ndarray]
    x: Dict[int, np.ndarray]
    E_packed: Dict[int, np.ndarray] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def copy(self) -> "HybridState":
        E_copy = {k: v.copy() for k, v in self.E.items()}
        x_copy = {k: v.copy() for k, v in self.x.items()}
        E_pack_copy = {k: v.copy() for k, v in self.E_packed.items()} if self.E_packed else {}
        return HybridState(E=E_copy, x=x_copy, E_packed=E_pack_copy, metadata=self.metadata.copy())

    def pack_all(self) -> None:
        for k, v in self.E.items():
            self.E_packed[k] = pack_bits(v)

# --------------------------- hybrid_semantic/hamiltonian ----------------------
class SemanticHamiltonian:
    def __init__(self,
                 nodes: Sequence[int],
                 edges: Sequence[Tuple[int, int]],
                 Sigma_inv: np.ndarray,
                 X_bar: np.ndarray,
                 lambda_bit: float = 1.0,
                 lambda_pos: float = 1.0,
                 lambda_label: float = 1.0) -> None:
        self.nodes = list(nodes)
        self.edges = list(edges)
        self.Sigma_inv = np.asarray(Sigma_inv)
        self.X_bar = np.asarray(X_bar).ravel()
        self.lambda_bit = float(lambda_bit)
        self.lambda_pos = float(lambda_pos)
        self.lambda_label = float(lambda_label)
        self.neighbors: Dict[int, List[int]] = {n: [] for n in self.nodes}
        for (i, j) in self.edges:
            self.neighbors.setdefault(i, []).append(j)
            self.neighbors.setdefault(j, []).append(i)

    def energy(self, state: HybridState) -> float:
        xs = [state.x[n].ravel() for n in sorted(state.x.keys())]
        if len(xs) == 0:
            cont = 0.0
        else:
            X_vec = np.concatenate(xs)
            if X_vec.size != self.X_bar.size or self.Sigma_inv.shape[0] != self.Sigma_inv.shape[1] or self.Sigma_inv.shape[0] != X_vec.size:
                raise ValueError("Sigma_inv or X_bar shape mismatch with state x")
            diff = X_vec - self.X_bar
            cont = 0.5 * float(diff.T @ self.Sigma_inv @ diff)
        pair_energy = 0.0
        for (i, j) in self.edges:
            Ei = np.asarray(state.E[i]).astype(int)
            Ej = np.asarray(state.E[j]).astype(int)
            ham = int(np.bitwise_xor(Ei, Ej).sum())
            pos_diff = np.asarray(state.x[i]) - np.asarray(state.x[j])
            pair_energy += self.lambda_bit * float(ham) + self.lambda_pos * float(np.sum(pos_diff ** 2))
        label_term = 0.0
        return float(cont + pair_energy + label_term)

    def delta_energy_for_bitflip(self, state: HybridState, node: int, bit_idx: int) -> float:
        Ei = np.asarray(state.E[node]).astype(int)
        orig_bit = int(Ei[bit_idx])
        flipped = 1 - orig_bit
        delta = 0.0
        for j in self.neighbors.get(node, []):
            Ej = np.asarray(state.E[j]).astype(int)
            # change in Hamming distance at that bit
            before = 1 if orig_bit != int(Ej[bit_idx]) else 0
            after = 1 if flipped != int(Ej[bit_idx]) else 0
            delta += self.lambda_bit * float(after - before)
        return float(delta)

    def analytic_gradient(self, state: HybridState) -> Dict[int, np.ndarray]:
        grads: Dict[int, np.ndarray] = {}
        xs = [state.x[n].ravel() for n in sorted(state.x.keys())]
        if len(xs) == 0:
            return {n: np.zeros_like(state.x[n]) for n in state.x}
        X_vec = np.concatenate(xs)
        grad_vec = self.Sigma_inv @ (X_vec - self.X_bar)
        sizes = [state.x[n].ravel().size for n in sorted(state.x.keys())]
        idx = 0
        for n, sz in zip(sorted(state.x.keys()), sizes):
            g = grad_vec[idx:idx + sz]
            grads[n] = np.asarray(g).reshape(state.x[n].shape)
            idx += sz
        for i in self.nodes:
            if i not in grads:
                grads[i] = np.zeros_like(state.x[i])
        for (i, j) in self.edges:
            diff = np.asarray(state.x[i]) - np.asarray(state.x[j])
            grads[i] = grads[i] + 2.0 * self.lambda_pos * diff
            grads[j] = grads[j] - 2.0 * self.lambda_pos * diff
        return grads

# --------------------------- hybrid_semantic/gradient_flow ---------------------
class GradientFlow:
    def __init__(self, state: HybridState, hamiltonian: SemanticHamiltonian, lr: float = 0.1) -> None:
        self.state = state
        self.ham = hamiltonian
        self.lr = float(lr)

    def step(self, dt: float) -> None:
        grads = self.ham.analytic_gradient(self.state)
        for n, g in grads.items():
            arr = np.asarray(self.state.x[n]).astype(float)
            self.state.x[n] = (arr - dt * self.lr * g).astype(float)

# --------------------------- hybrid_semantic/metropolis -------------------------
class MetropolisEngine:
    def __init__(self, state: HybridState, hamiltonian: SemanticHamiltonian, anneal_fn: Callable[[int], float]) -> None:
        self.state = state
        self.ham = hamiltonian
        self.anneal_fn = anneal_fn
        self.t: int = 0

    def step(self) -> None:
        T = float(self.anneal_fn(self.t))
        node = random.choice(list(self.state.E.keys()))
        d = int(np.asarray(self.state.E[node]).size)
        bit_idx = random.randrange(d)
        delta = self.ham.delta_energy_for_bitflip(self.state, node, bit_idx)
        accept_prob = min(1.0, math.exp(-delta / (T + 1e-12)))
        if random.random() < accept_prob:
            arr = np.asarray(self.state.E[node]).astype(int).copy()
            arr[bit_idx] = 1 - int(arr[bit_idx])
            self.state.E[node] = arr
            # keep packed representation consistent
            if node in self.state.E_packed:
                self.state.E_packed[node] = pack_bits(self.state.E[node])
        self.t += 1

# --------------------------- cognitive_field ---------------------------------
class PhiCalculator:
    def __init__(self, state: HybridState) -> None:
        self.state = state

    def compute_phi(self) -> float:
        return 0.0

class FreeEnergyEngine:
    def __init__(self, state: HybridState) -> None:
        self.state = state

    def free_energy(self) -> float:
        return 0.0

class ConfidenceSynthesizer:
    def __init__(self, weights: Optional[Dict[str, float]] = None) -> None:
        self.weights: Dict[str, float] = weights or {'w1': 1.0, 'w2': 1.0, 'w3': 1.0, 'w4': 1.0, 'w5': 1.0}

    def synthesize(self, gcl: float, emergence: float, stress_avg: float, harmony: float, delta_c: float) -> float:
        linear = (self.weights['w1'] * gcl + self.weights['w2'] * emergence -
                  self.weights['w3'] * stress_avg + self.weights['w4'] * harmony + self.weights['w5'] * delta_c)
        return float(expit(linear))

# --------------------------- error_correction --------------------------------
class SyndromeDetector:
    def __init__(self, stabilizers: Sequence[Sequence[Tuple[int, int]]]) -> None:
        self.stabilizers = list(stabilizers)

    def detect(self, state: HybridState) -> List[int]:
        syndromes: List[int] = []
        for S in self.stabilizers:
            xor_sum = 0
            for (node, idx) in S:
                arr = np.asarray(state.E[node]).astype(int).ravel()
                xor_sum ^= int(arr[idx])
            syndromes.append(int(xor_sum))
        return syndromes

    def detect_and_repair(self, state: HybridState) -> List[int]:
        synd = self.detect(state)
        return synd

class NeuralDecoder:
    def __init__(self, model: Optional[Any] = None) -> None:
        self.model = model

    def decode(self, syndrome: Sequence[int]) -> Optional[Any]:
        return None

class HarmonicMemory:
    def __init__(self) -> None:
        self.patterns: Dict[Any, Dict[str, Any]] = {}

    def store(self, pattern_id: Any, stress_init: float, confidence: float) -> None:
        self.patterns[pattern_id] = {'A': float(stress_init), 'conf': float(confidence)}

    def retrieve(self, pattern_id: Any, t: float) -> Optional[float]:
        params = self.patterns.get(pattern_id)
        if params is None:
            return None
        return float(params['A'] * math.sin(float(t)))

# --------------------------- category_logic ----------------------------------
class CategoryKernel:
    def __init__(self) -> None:
        self.objects: set = set()
        self.morphisms: Dict[Tuple[Any, Any], List[Callable]] = {}

    def add_object(self, obj: Any) -> None:
        self.objects.add(obj)

    def add_morphism(self, a: Any, b: Any, morphism: Callable) -> None:
        self.morphisms.setdefault((a, b), []).append(morphism)

class AxiomVerifier:
    def __init__(self, axioms: Optional[Dict[str, Any]] = None) -> None:
        self.axioms = axioms or {}

    def check_all(self, state: HybridState) -> Dict[str, bool]:
        return {}

# --------------------------- utils.annealing ---------------------------------
def hajek_schedule(c: float = 1.0) -> Callable[[int], float]:
    def schedule(t: int) -> float:
        return float(c) / max(1e-12, math.log(max(2.0, float(t) + 1.0)))
    return schedule

def constant_schedule(T: float = 1.0) -> Callable[[int], float]:
    return lambda t: float(T)

# --------------------------- utils.metrics ----------------------------------
def gcl(state: HybridState, edges: Sequence[Tuple[int, int]]) -> float:
    if len(edges) == 0:
        return 0.0
    total: float = 0.0
    for (i, j) in edges:
        Ei = np.asarray(state.E[i]).astype(int)
        Ej = np.asarray(state.E[j]).astype(int)
        ham = int(np.bitwise_xor(Ei, Ej).sum())
        d = int(Ei.size)
        total += (1.0 - ham / max(1, d))
    return float(total / len(edges))

# --------------------------- core/runtime -----------------------------------
class Runtime:
    def __init__(self, hybrid: Any, cognitive: Any, error_correction: Any, hardware: Any, config: Optional[Dict[str, Any]] = None) -> None:
        self.hybrid = hybrid
        self.cognitive = cognitive
        self.error_correction = error_correction
        self.hardware = hardware
        self.config = config or {}
        self.running = False

    def step(self, dt: float = 0.01) -> None:
        self.hybrid.gradient_flow.step(dt)
        self.hybrid.metropolis.step()
        self.error_correction.detect_and_repair(self.hybrid.state)
        # cognitive update placeholder
        # invariants could be checked here

    def run(self, dt: float = 0.01, steps: Optional[int] = None) -> None:
        self.running = True
        try:
            step_i = 0
            while self.running:
                self.step(dt)
                step_i += 1
                if steps is not None and step_i >= steps:
                    break
                time.sleep(dt)
        finally:
            self.running = False

    def stop(self) -> None:
        self.running = False

# --------------------------- demo / builder ---------------------------------
def build_minimal_system(node_count: int = 8, bit_dim: int = 64) -> Tuple[HybridState, SemanticHamiltonian, GradientFlow, MetropolisEngine]:
    rng = np.random.default_rng(42)
    E = {i: rng.integers(0, 2, size=(bit_dim,), dtype=np.uint8) for i in range(node_count)}
    x = {i: rng.standard_normal(3,) for i in range(node_count)}
    state = HybridState(E=E, x=x)
    state.pack_all()
    Sigma_inv = np.eye(3 * node_count)
    X_bar = np.zeros((3 * node_count,), dtype=float)
    edges = [(i, (i + 1) % node_count) for i in range(node_count)]
    ham = SemanticHamiltonian(nodes=list(range(node_count)), edges=edges, Sigma_inv=Sigma_inv, X_bar=X_bar,
                              lambda_bit=1.0, lambda_pos=0.1)
    gf = GradientFlow(state, ham, lr=0.05)
    me = MetropolisEngine(state, ham, hajek_schedule(c=1.0))
    return state, ham, gf, me

# --------------------------- quick test ------------------------------------
if __name__ == '__main__':
    state, ham, gf, me = build_minimal_system(node_count=6, bit_dim=64)
    print('Initial energy:', ham.energy(state))
    for t in range(5):
        gf.step(0.01)
        me.step()
        print(f'step {t}: energy={ham.energy(state):.6f}')

# EOF
