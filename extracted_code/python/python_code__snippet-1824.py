import pytest
import math
from goeckoh.heart.logic_core import CrystallineHeart

def test_gcl_stability():
    heart = CrystallineHeart()
    
    # Test 1: Idle state should be stable
    metrics = heart.compute_metrics()
    assert metrics.gcl > 0.9
    assert metrics.mode_label == "FLOW"

    # Test 2: Meltdown injection
    # Massively increase lattice energy
    heart.nodes = [10.0] * 1024
    metrics = heart.compute_metrics()
    
    assert metrics.gcl < 0.2
    assert metrics.mode_label == "MELTDOWN"

def test_input_normalization():
    heart = CrystallineHeart()
    resp, _ = heart.process_input("You are bad")
    # Ensure semantic mirror works
    assert "I" in resp
    assert "bad" in resp
    assert "You" not in resp.lower()
