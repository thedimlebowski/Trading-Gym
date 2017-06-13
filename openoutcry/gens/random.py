import numpy as np
from openoutcry.core import DataGenerator


class RandomWalk(DataGenerator):
    """Random walk data generator for one product
    """
    @staticmethod
    def _generator(ba_spread=0):
        val = 0
        while True:
            val += np.random.standard_normal()
            yield val, val + ba_spread


class AR1(DataGenerator):
    """Standardised AR1 data generator
    """
    @staticmethod
    def _generator(a, ba_spread=0):
        """Generator for standardised AR1

        Args:
            a (float): AR1 coefficient
            ba_spread (float): spread between bid/ask

        Yields:
            (tuple): bid ask spread
        """
        assert abs(a) < 1
        sigma = np.sqrt(1 - a**2)
        val = np.random.normal(scale=sigma)
        while True:
            val += (a - 1) * val + np.random.normal(scale=sigma)
            yield val, val + ba_spread
