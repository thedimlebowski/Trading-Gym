import numpy as np
from tgym.core import DataGenerator


class RandomWalk(DataGenerator):
    """Random walk data generator for one product
    """
    @staticmethod
    def _generator(ba_spread=0):
        """Generator for a pure random walk

        Args:
            ba_spread (float): spread between bid/ask

        Yields:
            (tuple): bid ask prices
        """
        val = 0
        while True:
            yield val, val + ba_spread
            val += np.random.standard_normal()


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
            (tuple): bid ask prices
        """
        assert abs(a) < 1
        sigma = np.sqrt(1 - a**2)
        val = np.random.normal(scale=sigma)
        while True:
            yield val, val + ba_spread
            val += (a - 1) * val + np.random.normal(scale=sigma)
