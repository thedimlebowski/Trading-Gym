import numpy as np
from tgym.envs import SpreadTrading
from tgym.gens import WavySignal


class TestSpreadTrading(object):

    data_generator = WavySignal(period_1=3, period_2=1, epsilon=0)
    st = SpreadTrading(
        data_generator=data_generator,
        spread_coefficients=[1],
        episode_length=1000,
        trading_fee=0.2,
        time_fee=0.1,
        history_length=1
    )

    def test_init(self):
        assert self.st._data_generator == self.data_generator
        assert self.st._spread_coefficients == [1]
        assert self.st._first_render
        assert self.st._trading_fee == 0.2
        assert self.st._time_fee == 0.1
        assert self.st._episode_length == 1000
        assert self.st.n_actions == 3
        assert self.st._history_length == 1
        assert len(self.st._prices_history) == 1

    def test_step(self):
        # Buy
        state = self.st.step(np.array([0, 1, 0]))
        assert state[0][0] == state[0][1]
        assert (state[0][-3:] == np.array([0, 1, 0])).all()
        assert self.st._entry_price != 0
        assert self.st._exit_price == 0
        # Hold
        state = self.st.step(np.array([1, 0, 0]))
        assert (state[0][-3:] == np.array([0, 1, 0])).all()
        assert self.st._entry_price != 0
        assert self.st._exit_price == 0
        # Sell
        state = self.st.step(np.array([0, 0, 1]))
        assert (state[0][-3:] == np.array([1, 0, 0])).all()
        assert self.st._entry_price == 0
        assert self.st._exit_price != 0

    def test_reset(self):
        pass
