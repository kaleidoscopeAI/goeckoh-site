import unittest
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from heart import CrystallineHeart

class TestHeart(unittest.TestCase):

    def test_initial_state(self):
        heart = CrystallineHeart()
        self.assertEqual(heart.gcl, 1.0)
        self.assertTrue(np.all(heart.state == 0))

    def test_pulse_low_entropy(self):
        heart = CrystallineHeart()
        # Low energy and latency should result in minimal change and high GCL
        gcl, entropy = heart.pulse(energy=0.01, latency=100)
        self.assertAlmostEqual(gcl, 1.0, places=5)
        self.assertTrue(gcl <= 1.0 and gcl >= 0.0)

    def test_pulse_high_entropy(self):
        heart = CrystallineHeart()
        # High energy and latency should introduce noise and lower GCL
        initial_state = heart.state.copy()
        gcl, entropy = heart.pulse(energy=0.8, latency=500)
        self.assertNotAlmostEqual(gcl, 1.0)
        self.assertTrue(gcl < 0.9) # GCL should have dropped significantly
        self.assertFalse(np.all(heart.state == initial_state))

    def test_gcl_clipping(self):
        heart = CrystallineHeart()
        # Even with massive energy, GCL should be clipped between 0 and 1
        gcl, _ = heart.pulse(energy=10.0, latency=1000)
        self.assertTrue(gcl >= 0.0 and gcl <= 1.0)

if __name__ == '__main__':
    unittest.main()
