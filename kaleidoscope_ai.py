"""
kaleidoscope_ai.py

A single-file, production-grade Python library that consolidates the core components
present in the uploaded "Kaleidoscope AI" chat log into a coherent, runnable module.

Key modules implemented:
- ConsciousCube: dynamic conceptual graph with activity-driven self-architecture
- SuperNode: domain agent with data ingestion, web crawling, causal tests, forecasting
- TimeSeriesForecaster: ARIMA / SARIMAX / Prophet (optional) with robust validation
- SarimaGridSearch: exhaustive SARIMA parameter search with multiprocessing option
- BayesianArimaOptimizer: Gaussian-process Bayesian optimization for ARIMA params
- VoxelVisualizer: voxel-based 3D visualization driven by cube state

This file intentionally contains no simulations or "example runs" in __main__.
It is designed to be imported and used by an application layer.
"""

from __future__ import annotations

import dataclasses
import itertools
import logging
import math
import multiprocessing as mp
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

# NetworkX is required for the cube graph
import networkx as nx

# Statsmodels is required for ARIMA/SARIMAX and Granger causality
from statsmodels.tsa.api import grangercausalitytests
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Requests + BeautifulSoup are used for web crawling
import requests
from bs4 import BeautifulSoup

# Optional Prophet. Imported lazily in TimeSeriesForecaster when used.
# from prophet import Prophet

# Optional sklearn (Gaussian process) for Bayesian optimization. Imported lazily.
# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel


__all__ = [
    "KaleidoscopeError",
    "ValidationError",
    "ModelUnavailableError",
    "FetchError",
    "ConsciousCube",
    "SuperNode",
    "TimeSeriesForecaster",
    "SarimaGridSearch",
    "BayesianArimaOptimizer",
    "VoxelVisualizer",
]


# -----------------------------
# Logging
# -----------------------------

_LOGGER = logging.getLogger("kaleidoscope_ai")
if not _LOGGER.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    _LOGGER.addHandler(handler)
_LOGGER.setLevel(logging.INFO)


# -----------------------------
# Errors
# -----------------------------

class KaleidoscopeError(RuntimeError):
    """Base exception for this library."""


class ValidationError(KaleidoscopeError):
    """Raised when an input validation fails."""


class ModelUnavailableError(KaleidoscopeError):
    """Raised when an optional dependency/model is requested but not available."""


class FetchError(KaleidoscopeError):
    """Raised when web crawling/data fetching fails."""


# -----------------------------
# Helpers
# -----------------------------

def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _as_series(data: Union[pd.Series, Sequence[float], np.ndarray], name: str) -> pd.Series:
    if isinstance(data, pd.Series):
        if data.name is None:
            return data.rename(name)
        return data
    arr = np.asarray(list(data), dtype=float)
    _require(arr.size > 0, f"{name}: data must be non-empty.")
    return pd.Series(arr, name=name)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _clip01(x: float) -> float:
    if math.isnan(x) or math.isinf(x):
        return 0.0
    return float(min(1.0, max(0.0, x)))


# -----------------------------
# Conscious Cube
# -----------------------------

@dataclass(frozen=True)
class NodeViz:
    color: str
    size: float
    form: str


@dataclass(frozen=True)
class EdgeViz:
    thickness: float
    glow: bool


class ConsciousCube:
    """
    Dynamic conceptual graph ("Conscious Cube") with activity-driven topology refinement.

    - Nodes hold (strength, activity_score)
    - Edges hold weight in [0,1]
    - A viz_map stores conceptual render properties for nodes/edges
    """

    def __init__(self) -> None:
        self.graph: nx.Graph = nx.Graph()
        self.node_states: Dict[str, Dict[str, float]] = {}
        self.viz_map_nodes: Dict[str, NodeViz] = {}
        self.viz_map_edges: Dict[Tuple[str, str], EdgeViz] = {}

        _LOGGER.info("ConsciousCube initialized.")

    def add_conceptual_node(self, node_id: str, initial_strength: float = 0.1) -> None:
        _require(isinstance(node_id, str) and node_id.strip() != "", "node_id must be a non-empty string.")
        _require(isinstance(initial_strength, (int, float)), "initial_strength must be numeric.")
        initial_strength_f = _clip01(float(initial_strength))

        if node_id in self.graph:
            return

        self.graph.add_node(node_id)
        self.node_states[node_id] = {"strength": initial_strength_f, "activity_score": 0.0}
        self._update_node_viz(node_id)

    def add_weighted_edge(self, u: str, v: str, weight: float = 0.5) -> None:
        _require(isinstance(u, str) and u.strip() != "", "u must be a non-empty string.")
        _require(isinstance(v, str) and v.strip() != "", "v must be a non-empty string.")
        _require(u != v, "u and v must be different node ids.")
        _require(isinstance(weight, (int, float)), "weight must be numeric.")

        if u not in self.graph:
            self.add_conceptual_node(u, initial_strength=0.1)
        if v not in self.graph:
            self.add_conceptual_node(v, initial_strength=0.1)

        w = _clip01(float(weight))
        self.graph.add_edge(u, v, weight=w)
        self._update_edge_viz(u, v)

    def update_node_activity(self, node_id: str, activity_score: float) -> None:
        _require(isinstance(node_id, str) and node_id.strip() != "", "node_id must be a non-empty string.")
        _require(node_id in self.node_states, f"Node '{node_id}' does not exist; add it first.")
        _require(isinstance(activity_score, (int, float)), "activity_score must be numeric.")

        a = _clip01(float(activity_score))
        state = self.node_states[node_id]
        state["activity_score"] = a
        # Smooth strength update; activity nudges strength upward, inactivity downward
        state["strength"] = _clip01(state["strength"] * 0.90 + a * 0.10)
        self._update_node_viz(node_id)

        self._self_architect(central_node_id=node_id)

    def _self_architect(self, central_node_id: str) -> None:
        # High activity -> reinforce existing edges and create emergent links
        center = self.node_states.get(central_node_id)
        if not center:
            return

        try:
            if center["activity_score"] > 0.7:
                for other in list(self.graph.nodes()):
                    if other == central_node_id:
                        continue

                    # reinforce if already connected
                    if self.graph.has_edge(central_node_id, other):
                        current = float(self.graph[central_node_id][other].get("weight", 0.5))
                        new_w = _clip01(current + 0.1)
                        self.graph[central_node_id][other]["weight"] = new_w
                        self._update_edge_viz(central_node_id, other)

                    # emergent link if both strong and not connected
                    other_state = self.node_states.get(other)
                    if not other_state:
                        continue
                    if (
                        center["strength"] > 0.6
                        and other_state["strength"] > 0.6
                        and not self.graph.has_edge(central_node_id, other)
                    ):
                        # Probabilistic formation; deterministic alternative is possible if caller supplies rules.
                        if random.random() < 0.2:
                            self.graph.add_edge(central_node_id, other, weight=0.5)
                            self._update_edge_viz(central_node_id, other)

            # prune inactive isolated nodes (never prune the node that triggered the update)
            for nid, st in list(self.node_states.items()):
                if nid == central_node_id:
                    continue
                if st["strength"] < 0.1 and self.graph.degree(nid) == 0:
                    if nid in self.graph:
                        self.graph.remove_node(nid)
                    self.node_states.pop(nid, None)
                    self.viz_map_nodes.pop(nid, None)
        except Exception as e:
            _LOGGER.exception("Self-architecture error: %s", e)
            raise

    def _update_node_viz(self, node_id: str) -> None:
        st = self.node_states.get(node_id)
        if not st:
            return
        strength = float(st["strength"])
        activity = float(st["activity_score"])
        color = "green" if activity < 0.4 else ("yellow" if activity < 0.7 else "red")
        size = 0.5 + strength * 1.5
        form = "sphere" if strength < 0.7 else "pyramid_active"
        self.viz_map_nodes[node_id] = NodeViz(color=color, size=size, form=form)

    def _update_edge_viz(self, u: str, v: str) -> None:
        if not self.graph.has_edge(u, v):
            return
        w = float(self.graph[u][v].get("weight", 0.0))
        thickness = 0.1 + _clip01(w) * 0.9
        glow = bool(w > 0.8)
        key = (u, v) if u <= v else (v, u)
        self.viz_map_edges[key] = EdgeViz(thickness=thickness, glow=glow)

    def get_viz_state(self) -> Dict[str, Any]:
        nodes = {nid: dataclasses.asdict(v) for nid, v in self.viz_map_nodes.items()}
        edges = {}
        for u, v in self.graph.edges():
            key = (u, v) if u <= v else (v, u)
            ev = self.viz_map_edges.get(key)
            if ev is None:
                self._update_edge_viz(u, v)
                ev = self.viz_map_edges.get(key)
            edges[f"{key[0]}::{key[1]}"] = dataclasses.asdict(ev) if ev else {"thickness": 0.2, "glow": False}
        return {"nodes": nodes, "edges": edges}

    def snapshot(self) -> Dict[str, Any]:
        """A serializable snapshot of the cube's graph and states."""
        nodes = []
        for nid in self.graph.nodes():
            st = self.node_states.get(nid, {"strength": 0.0, "activity_score": 0.0})
            nodes.append({"id": nid, "strength": float(st["strength"]), "activity_score": float(st["activity_score"])})
        edges = []
        for u, v, d in self.graph.edges(data=True):
            edges.append({"u": u, "v": v, "weight": float(d.get("weight", 0.5))})
        return {"nodes": nodes, "edges": edges, "timestamp_utc": _utc_now().isoformat()}


# -----------------------------
# Forecasting + Optimization
# -----------------------------

@dataclass(frozen=True)
class ForecastResult:
    model_name: str
    yhat: np.ndarray
    yhat_lower: Optional[np.ndarray]
    yhat_upper: Optional[np.ndarray]
    metrics: Dict[str, float]


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    _require(y_true.shape == y_pred.shape, "RMSE: shapes must match.")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    _require(y_true.shape == y_pred.shape, "MAE: shapes must match.")
    return float(np.mean(np.abs(y_true - y_pred)))


class TimeSeriesForecaster:
    """
    Fit and forecast time series using ARIMA, SARIMAX, and Prophet (optional).

    Input:
      - y: series values
      - index: optional DatetimeIndex or numeric index
    """

    def __init__(self) -> None:
        pass

    def forecast_arima(
        self,
        y: Union[pd.Series, Sequence[float], np.ndarray],
        order: Tuple[int, int, int],
        steps: int,
        enforce_stationarity: bool = True,
        enforce_invertibility: bool = True,
    ) -> ForecastResult:
        ys = _as_series(y, "y")
        _require(isinstance(order, tuple) and len(order) == 3, "order must be a (p,d,q) tuple.")
        _require(isinstance(steps, int) and steps > 0, "steps must be a positive integer.")

        model = ARIMA(
            ys,
            order=order,
            enforce_stationarity=enforce_stationarity,
            enforce_invertibility=enforce_invertibility,
        )
        fit = model.fit()
        fc = fit.get_forecast(steps=steps)
        mean = np.asarray(fc.predicted_mean, dtype=float)

        conf = fc.conf_int(alpha=0.05)
        lower = np.asarray(conf.iloc[:, 0], dtype=float) if conf is not None else None
        upper = np.asarray(conf.iloc[:, 1], dtype=float) if conf is not None else None

        return ForecastResult(
            model_name=f"ARIMA{order}",
            yhat=mean,
            yhat_lower=lower,
            yhat_upper=upper,
            metrics={},
        )

    def forecast_sarimax(
        self,
        y: Union[pd.Series, Sequence[float], np.ndarray],
        order: Tuple[int, int, int],
        seasonal_order: Tuple[int, int, int, int],
        steps: int,
        enforce_stationarity: bool = True,
        enforce_invertibility: bool = True,
    ) -> ForecastResult:
        ys = _as_series(y, "y")
        _require(isinstance(order, tuple) and len(order) == 3, "order must be a (p,d,q) tuple.")
        _require(isinstance(seasonal_order, tuple) and len(seasonal_order) == 4, "seasonal_order must be (P,D,Q,m).")
        _require(isinstance(steps, int) and steps > 0, "steps must be a positive integer.")

        model = SARIMAX(
            ys,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=enforce_stationarity,
            enforce_invertibility=enforce_invertibility,
        )
        fit = model.fit(disp=False)
        fc = fit.get_forecast(steps=steps)
        mean = np.asarray(fc.predicted_mean, dtype=float)

        conf = fc.conf_int(alpha=0.05)
        lower = np.asarray(conf.iloc[:, 0], dtype=float) if conf is not None else None
        upper = np.asarray(conf.iloc[:, 1], dtype=float) if conf is not None else None

        return ForecastResult(
            model_name=f"SARIMAX{order}x{seasonal_order}",
            yhat=mean,
            yhat_lower=lower,
            yhat_upper=upper,
            metrics={},
        )

    def forecast_prophet(
        self,
        y: Union[pd.Series, Sequence[float], np.ndarray],
        ds: Optional[Union[pd.DatetimeIndex, Sequence[datetime]]] = None,
        steps: int = 10,
        freq: str = "D",
        interval_width: float = 0.95,
    ) -> ForecastResult:
        """
        Prophet requires a dataframe with columns: ds (datetime), y (float).

        If ds is not provided:
          - if y is a Series with DatetimeIndex, use that index
          - else raise ValidationError (Prophet needs timestamps)
        """
        try:
            from prophet import Prophet  # type: ignore
        except Exception as e:
            raise ModelUnavailableError("Prophet is not available. Install 'prophet' to use forecast_prophet().") from e

        ys = _as_series(y, "y")
        _require(isinstance(steps, int) and steps > 0, "steps must be a positive integer.")
        _require(0.5 <= float(interval_width) <= 0.99, "interval_width must be between 0.5 and 0.99.")

        if ds is None:
            if isinstance(ys.index, pd.DatetimeIndex):
                ds_index = ys.index
            else:
                raise ValidationError("Prophet requires ds timestamps. Provide ds or use a Series with DatetimeIndex.")
        else:
            ds_index = pd.DatetimeIndex(ds)

        _require(len(ds_index) == len(ys), "ds length must match y length.")

        df = pd.DataFrame({"ds": ds_index, "y": ys.values.astype(float)})
        m = Prophet(interval_width=float(interval_width))
        m.fit(df)

        future = m.make_future_dataframe(periods=int(steps), freq=str(freq), include_history=False)
        pred = m.predict(future)

        yhat = pred["yhat"].to_numpy(dtype=float)
        lower = pred.get("yhat_lower")
        upper = pred.get("yhat_upper")

        return ForecastResult(
            model_name="Prophet",
            yhat=yhat,
            yhat_lower=lower.to_numpy(dtype=float) if lower is not None else None,
            yhat_upper=upper.to_numpy(dtype=float) if upper is not None else None,
            metrics={},
        )

    def blend_prophet_sarimax(
        self,
        y: Union[pd.Series, Sequence[float], np.ndarray],
        sarimax_order: Tuple[int, int, int],
        sarimax_seasonal_order: Tuple[int, int, int, int],
        steps: int,
        ds: Optional[Union[pd.DatetimeIndex, Sequence[datetime]]] = None,
        freq: str = "D",
        alpha: float = 0.5,
    ) -> ForecastResult:
        """
        Blend SARIMAX and Prophet forecasts: yhat = alpha * prophet + (1-alpha) * sarimax
        """
        _require(0.0 <= float(alpha) <= 1.0, "alpha must be in [0,1].")
        sar = self.forecast_sarimax(y=y, order=sarimax_order, seasonal_order=sarimax_seasonal_order, steps=steps)
        pro = self.forecast_prophet(y=y, ds=ds, steps=steps, freq=freq)
        _require(sar.yhat.shape == pro.yhat.shape, "Forecast shapes must match for blending.")
        yhat = float(alpha) * pro.yhat + (1.0 - float(alpha)) * sar.yhat
        return ForecastResult(
            model_name=f"Blend(Prophet,SARIMAX,alpha={alpha:.2f})",
            yhat=yhat,
            yhat_lower=None,
            yhat_upper=None,
            metrics={},
        )


@dataclass(frozen=True)
class SarimaCandidate:
    order: Tuple[int, int, int]
    seasonal_order: Tuple[int, int, int, int]
    aic: float
    bic: float


class SarimaGridSearch:
    """
    Exhaustive SARIMAX grid search with optional multiprocessing.

    This evaluates models by AIC/BIC and returns the best candidates.

    Note: SARIMAX fitting can be expensive; restrict search spaces deliberately.
    """

    def __init__(self, max_workers: Optional[int] = None) -> None:
        self.max_workers = max_workers

    @staticmethod
    def _fit_one(args: Tuple[np.ndarray, Tuple[int, int, int], Tuple[int, int, int, int], bool, bool]) -> Optional[SarimaCandidate]:
        y, order, seasonal_order, enforce_stationarity, enforce_invertibility = args
        try:
            model = SARIMAX(
                y,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=enforce_stationarity,
                enforce_invertibility=enforce_invertibility,
            )
            res = model.fit(disp=False)
            return SarimaCandidate(order=order, seasonal_order=seasonal_order, aic=float(res.aic), bic=float(res.bic))
        except Exception:
            return None

    def search(
        self,
        y: Union[pd.Series, Sequence[float], np.ndarray],
        p: Sequence[int],
        d: Sequence[int],
        q: Sequence[int],
        P: Sequence[int],
        D: Sequence[int],
        Q: Sequence[int],
        m: Sequence[int],
        enforce_stationarity: bool = True,
        enforce_invertibility: bool = True,
        top_k: int = 5,
    ) -> List[SarimaCandidate]:
        ys = _as_series(y, "y").to_numpy(dtype=float)
        _require(len(ys) >= 10, "Need at least 10 samples for SARIMAX grid search.")
        _require(isinstance(top_k, int) and top_k > 0, "top_k must be a positive integer.")

        combos: List[Tuple[Tuple[int, int, int], Tuple[int, int, int, int]]] = []
        for order in itertools.product(p, d, q):
            for seas in itertools.product(P, D, Q, m):
                combos.append((order, seas))

        tasks = [(ys, o, s, enforce_stationarity, enforce_invertibility) for (o, s) in combos]

        results: List[SarimaCandidate] = []
        if self.max_workers and self.max_workers > 1:
            with mp.Pool(processes=self.max_workers) as pool:
                for cand in pool.imap_unordered(self._fit_one, tasks):
                    if cand is not None:
                        results.append(cand)
        else:
            for t in tasks:
                cand = self._fit_one(t)
                if cand is not None:
                    results.append(cand)

        if not results:
            raise KaleidoscopeError("SARIMAX grid search produced no successful fits.")

        results.sort(key=lambda c: c.aic)
        return results[:top_k]


@dataclass(frozen=True)
class ArimaParams:
    p: int
    d: int
    q: int

    def as_order(self) -> Tuple[int, int, int]:
        return (int(self.p), int(self.d), int(self.q))


class BayesianArimaOptimizer:
    """
    Bayesian optimization for ARIMA(p,d,q) using Gaussian Processes (sklearn).

    Objective: minimize validation RMSE on a holdout tail segment.
    """

    def __init__(
        self,
        p_range: Tuple[int, int],
        d_range: Tuple[int, int],
        q_range: Tuple[int, int],
        n_initial: int = 10,
        n_iter: int = 25,
        random_state: int = 42,
    ) -> None:
        _require(p_range[0] <= p_range[1], "p_range invalid.")
        _require(d_range[0] <= d_range[1], "d_range invalid.")
        _require(q_range[0] <= q_range[1], "q_range invalid.")
        _require(n_initial > 0, "n_initial must be > 0.")
        _require(n_iter > 0, "n_iter must be > 0.")

        self.p_range = p_range
        self.d_range = d_range
        self.q_range = q_range
        self.n_initial = int(n_initial)
        self.n_iter = int(n_iter)
        self.random_state = int(random_state)

    @staticmethod
    def _fit_score(y_train: np.ndarray, y_val: np.ndarray, order: Tuple[int, int, int]) -> float:
        try:
            model = ARIMA(y_train, order=order)
            res = model.fit()
            fc = res.get_forecast(steps=len(y_val)).predicted_mean.to_numpy(dtype=float)
            return _rmse(y_val, fc)
        except Exception:
            return float("inf")

    def optimize(
        self,
        y: Union[pd.Series, Sequence[float], np.ndarray],
        val_size: int,
    ) -> Tuple[ArimaParams, Dict[str, Any]]:
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor  # type: ignore
            from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel  # type: ignore
        except Exception as e:
            raise ModelUnavailableError(
                "BayesianArimaOptimizer requires scikit-learn (GaussianProcessRegressor). Install 'scikit-learn'."
            ) from e

        ys = _as_series(y, "y").to_numpy(dtype=float)
        _require(isinstance(val_size, int) and 2 <= val_size < len(ys) // 2, "val_size must be >=2 and < len(y)/2.")

        y_train = ys[:-val_size]
        y_val = ys[-val_size:]

        rng = np.random.default_rng(self.random_state)

        def sample_params() -> ArimaParams:
            return ArimaParams(
                p=int(rng.integers(self.p_range[0], self.p_range[1] + 1)),
                d=int(rng.integers(self.d_range[0], self.d_range[1] + 1)),
                q=int(rng.integers(self.q_range[0], self.q_range[1] + 1)),
            )

        X: List[List[float]] = []
        y_obj: List[float] = []

        # initial random exploration
        seen: set = set()
        while len(X) < self.n_initial:
            params = sample_params()
            if params.as_order() in seen:
                continue
            seen.add(params.as_order())
            score = self._fit_score(y_train, y_val, params.as_order())
            X.append([params.p, params.d, params.q])
            y_obj.append(score)

        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-3)
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=self.random_state)

        # iterative BO: propose by maximizing expected improvement over random candidate pool
        def expected_improvement(mu: np.ndarray, sigma: np.ndarray, best: float) -> np.ndarray:
            sigma = np.maximum(sigma, 1e-9)
            z = (best - mu) / sigma
            # EI = (best-mu)*Phi(z) + sigma*phi(z)
            # Use scipy-free approximations for Phi/phi via erf
            phi = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * z**2)
            Phi = 0.5 * (1.0 + np.vectorize(math.erf)(z / np.sqrt(2.0)))
            return (best - mu) * Phi + sigma * phi

        for _ in range(self.n_iter):
            gp.fit(np.asarray(X, dtype=float), np.asarray(y_obj, dtype=float))
            best = float(np.min(y_obj))

            # candidate pool
            candidates: List[ArimaParams] = []
            while len(candidates) < 256:
                p = int(rng.integers(self.p_range[0], self.p_range[1] + 1))
                d = int(rng.integers(self.d_range[0], self.d_range[1] + 1))
                q = int(rng.integers(self.q_range[0], self.q_range[1] + 1))
                ap = ArimaParams(p=p, d=d, q=q)
                if ap.as_order() in seen:
                    continue
                candidates.append(ap)

            Xcand = np.asarray([[c.p, c.d, c.q] for c in candidates], dtype=float)
            mu, std = gp.predict(Xcand, return_std=True)
            ei = expected_improvement(mu=np.asarray(mu, dtype=float), sigma=np.asarray(std, dtype=float), best=best)
            pick = int(np.argmax(ei))
            chosen = candidates[pick]
            seen.add(chosen.as_order())

            score = self._fit_score(y_train, y_val, chosen.as_order())
            X.append([chosen.p, chosen.d, chosen.q])
            y_obj.append(score)

        best_idx = int(np.argmin(y_obj))
        best_params = ArimaParams(p=int(X[best_idx][0]), d=int(X[best_idx][1]), q=int(X[best_idx][2]))

        meta = {
            "best_rmse": float(y_obj[best_idx]),
            "evaluations": [{"p": int(x[0]), "d": int(x[1]), "q": int(x[2]), "rmse": float(s)} for x, s in zip(X, y_obj)],
        }
        return best_params, meta


# -----------------------------
# Web Crawling + Data Ingestion
# -----------------------------

@dataclass(frozen=True)
class CrawlResult:
    url: str
    fetched_at_utc: str
    status_code: int
    text: str
    title: Optional[str]


class WebCrawler:
    """
    A minimal, robust crawler that fetches HTML pages and extracts visible text.

    - Respects timeouts
    - Sanitizes whitespace
    - Allows custom headers
    """

    def __init__(
        self,
        user_agent: str = "kaleidoscope-ai/1.0 (+https://example.invalid)",
        timeout_s: float = 12.0,
        max_bytes: int = 2_000_000,
    ) -> None:
        _require(timeout_s > 0, "timeout_s must be > 0")
        _require(max_bytes > 0, "max_bytes must be > 0")
        self.user_agent = user_agent
        self.timeout_s = float(timeout_s)
        self.max_bytes = int(max_bytes)

    def fetch_text(self, url: str, headers: Optional[Mapping[str, str]] = None) -> CrawlResult:
        _require(isinstance(url, str) and url.strip() != "", "url must be a non-empty string.")
        req_headers = {"User-Agent": self.user_agent}
        if headers:
            req_headers.update(dict(headers))

        try:
            resp = requests.get(url, headers=req_headers, timeout=self.timeout_s)
        except Exception as e:
            raise FetchError(f"Request failed for url={url!r}: {e}") from e

        status = int(resp.status_code)
        raw = resp.content[: self.max_bytes]

        if status < 200 or status >= 300:
            raise FetchError(f"Non-2xx response for url={url!r}: status={status}")

        soup = BeautifulSoup(raw, "html.parser")
        title = soup.title.get_text(strip=True) if soup.title else None

        # Remove script/style/noscript tags
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)
        text = re.sub(r"\s+", " ", text).strip()

        return CrawlResult(
            url=url,
            fetched_at_utc=_utc_now().isoformat(),
            status_code=status,
            text=text,
            title=title,
        )


# -----------------------------
# Super Node
# -----------------------------

class SuperNode:
    """
    Domain agent that can:
    - ingest time series data
    - crawl web pages into a local knowledge cache
    - infer Granger causal links between stored series
    - train/forecast ARIMA/SARIMAX/Prophet
    """

    def __init__(self, node_id: str, domain_description: str) -> None:
        _require(isinstance(node_id, str) and node_id.strip() != "", "node_id must be a non-empty string.")
        _require(isinstance(domain_description, str) and domain_description.strip() != "", "domain_description must be non-empty.")

        self.node_id = node_id
        self.domain_description = domain_description
        self.historical_data: Dict[str, pd.Series] = {}
        self.crawl_cache: Dict[str, CrawlResult] = {}
        self.forecaster = TimeSeriesForecaster()
        self.crawler = WebCrawler()

        _LOGGER.info("SuperNode '%s' initialized for domain: %s", self.node_id, self.domain_description)

    def ingest_historical_data(self, series_name: str, data: Union[pd.Series, Sequence[float], np.ndarray]) -> None:
        _require(isinstance(series_name, str) and series_name.strip() != "", "series_name must be a non-empty string.")
        s = _as_series(data, series_name).astype(float)
        self.historical_data[series_name] = s

    def crawl_and_ingest_text(self, url: str) -> CrawlResult:
        res = self.crawler.fetch_text(url)
        self.crawl_cache[url] = res
        return res

    def granger_causality(self, series_x: str, series_y: str, max_lag: int = 3) -> Dict[str, Any]:
        """
        Returns:
          {
            "x_granger_causes_y": bool,
            "p_value": float,
            "max_lag": int
          }
        """
        _require(isinstance(series_x, str) and series_x.strip() != "", "series_x must be non-empty.")
        _require(isinstance(series_y, str) and series_y.strip() != "", "series_y must be non-empty.")
        _require(isinstance(max_lag, int) and max_lag > 0, "max_lag must be a positive integer.")
        _require(series_x in self.historical_data, f"Missing series_x={series_x!r}")
        _require(series_y in self.historical_data, f"Missing series_y={series_y!r}")

        x = self.historical_data[series_x]
        y = self.historical_data[series_y]
        if len(x) != len(y):
            n = min(len(x), len(y))
            x = x.iloc[:n]
            y = y.iloc[:n]

        _require(len(x) > max_lag + 1, f"Not enough data for max_lag={max_lag}.")

        df = pd.DataFrame({series_y: y.values, series_x: x.values})
        tests = grangercausalitytests(df[[series_y, series_x]], maxlag=max_lag, verbose=False)
        p_value = float(tests[max_lag][0]["ssr_ftest"][1])
        return {"x_granger_causes_y": bool(p_value < 0.05), "p_value": p_value, "max_lag": int(max_lag)}

    def forecast_arima(self, series_name: str, order: Tuple[int, int, int], steps: int) -> ForecastResult:
        _require(series_name in self.historical_data, f"Missing series {series_name!r}.")
        return self.forecaster.forecast_arima(self.historical_data[series_name], order=order, steps=steps)

    def forecast_sarimax(
        self,
        series_name: str,
        order: Tuple[int, int, int],
        seasonal_order: Tuple[int, int, int, int],
        steps: int,
    ) -> ForecastResult:
        _require(series_name in self.historical_data, f"Missing series {series_name!r}.")
        return self.forecaster.forecast_sarimax(self.historical_data[series_name], order=order, seasonal_order=seasonal_order, steps=steps)

    def forecast_prophet(
        self,
        series_name: str,
        steps: int,
        freq: str = "D",
    ) -> ForecastResult:
        _require(series_name in self.historical_data, f"Missing series {series_name!r}.")
        s = self.historical_data[series_name]
        _require(isinstance(s.index, pd.DatetimeIndex), "Prophet requires a DatetimeIndex on the series.")
        return self.forecaster.forecast_prophet(y=s, ds=s.index, steps=steps, freq=freq)

    def optimize_arima_bayes(
        self,
        series_name: str,
        val_size: int,
        p_range: Tuple[int, int],
        d_range: Tuple[int, int],
        q_range: Tuple[int, int],
        n_initial: int = 10,
        n_iter: int = 25,
        random_state: int = 42,
    ) -> Tuple[ArimaParams, Dict[str, Any]]:
        _require(series_name in self.historical_data, f"Missing series {series_name!r}.")
        opt = BayesianArimaOptimizer(
            p_range=p_range,
            d_range=d_range,
            q_range=q_range,
            n_initial=n_initial,
            n_iter=n_iter,
            random_state=random_state,
        )
        return opt.optimize(self.historical_data[series_name], val_size=val_size)

    def optimize_sarima_grid(
        self,
        series_name: str,
        p: Sequence[int],
        d: Sequence[int],
        q: Sequence[int],
        P: Sequence[int],
        D: Sequence[int],
        Q: Sequence[int],
        m: Sequence[int],
        top_k: int = 5,
        max_workers: Optional[int] = None,
    ) -> List[SarimaCandidate]:
        _require(series_name in self.historical_data, f"Missing series {series_name!r}.")
        gs = SarimaGridSearch(max_workers=max_workers)
        return gs.search(self.historical_data[series_name], p=p, d=d, q=q, P=P, D=D, Q=Q, m=m, top_k=top_k)


# -----------------------------
# Voxel Visualization
# -----------------------------

class VoxelVisualizer:
    """
    Voxel-based visualization helper. This is a pure helper that converts a cube snapshot into
    a voxel occupancy grid and renders it with matplotlib.
    """

    def __init__(self, grid_size: int = 16, seed: int = 42) -> None:
        _require(isinstance(grid_size, int) and grid_size >= 4, "grid_size must be an int >= 4.")
        self.grid_size = int(grid_size)
        self.rng = np.random.default_rng(int(seed))

    def assign_positions(self, cube: ConsciousCube) -> Dict[str, Tuple[int, int, int]]:
        positions: Dict[str, Tuple[int, int, int]] = {}
        for nid in cube.graph.nodes():
            positions[nid] = (
                int(self.rng.integers(0, self.grid_size)),
                int(self.rng.integers(0, self.grid_size)),
                int(self.rng.integers(0, self.grid_size)),
            )
        return positions

    def to_voxel_grid(self, cube: ConsciousCube, positions: Optional[Dict[str, Tuple[int, int, int]]] = None) -> Tuple[np.ndarray, Dict[str, Tuple[int, int, int]]]:
        if positions is None:
            positions = self.assign_positions(cube)
        occ = np.zeros((self.grid_size, self.grid_size, self.grid_size), dtype=bool)
        for nid, (x, y, z) in positions.items():
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size and 0 <= z < self.grid_size:
                occ[x, y, z] = True
        return occ, positions

    def render(self, cube: ConsciousCube, positions: Optional[Dict[str, Tuple[int, int, int]]] = None) -> None:
        try:
            import matplotlib.pyplot as plt
        except Exception as e:
            raise ModelUnavailableError("matplotlib is required for VoxelVisualizer.render().") from e

        occ, positions2 = self.to_voxel_grid(cube, positions=positions)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        # Use a simple facecolor map based on node activity
        facecolors = np.empty(occ.shape, dtype=object)
        facecolors[:] = "white"

        inv_pos = {pos: nid for nid, pos in positions2.items()}
        for (x, y, z), nid in inv_pos.items():
            st = cube.node_states.get(nid, {"activity_score": 0.0})
            a = float(st.get("activity_score", 0.0))
            facecolors[x, y, z] = "green" if a < 0.4 else ("yellow" if a < 0.7 else "red")

        ax.voxels(occ, facecolors=facecolors, edgecolor="k")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.title("ConsciousCube Voxel View")
        plt.show()
