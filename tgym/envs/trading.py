import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from tgym.core import Environment
from tgym.utils import calc_spread

plt.style.use('dark_background')
mpl.rcParams.update(
    {
        "font.size": 15,
        "axes.labelsize": 15,
        "lines.linewidth": 2,
        "lines.markersize": 15
    }
)


class SpreadTrading(Environment):
    """Class for a discrete (buy/hold/sell) tuple spread trading environment.
    """

    _actions = {
        'hold': np.array([1, 0, 0]),
        'buy': np.array([0, 1, 0]),
        'sell': np.array([0, 0, 1])
    }

    _positions = {
        'flat': np.array([1, 0, 0]),
        'long': np.array([0, 1, 0]),
        'short': np.array([0, 0, 1])
    }

    def __init__(self, data_generator, spread_coefficients, game_length=1000, trading_fee=0, time_fee=0, history_length=2):
        """Initialisation function

        Args:
            data_generator (tgym.core.DataGenerator): A data
                generator object yielding a 1D array of bid-ask prices.
            spread_coefficients (list): A list of signed integers defining
                how much of each product to buy (positive) or sell (negative)
                when buying or selling the spread.
            game_length (int): number of steps to play the game for
            trading_fee (float): penalty for trading
            time_fee (float): time fee
            history_length (int): number of historical states to stack in the
                observation vector.
        """

        assert data_generator.n_products == len(spread_coefficients)
        self._data_generator = data_generator
        self._spread_coefficients = spread_coefficients
        self._first_render = True
        self._trading_fee = trading_fee
        self._time_fee = time_fee
        self._game_length = game_length
        self.n_actions = 3
        self._prices_history = []
        self._history_length = history_length
        self.reset()

    def reset(self):
        """Reset the trading environment. Reset rewards, data generator,
        observation buffer etc.

        Returns:
            observation (numpy.array): observation of the state
        """
        self._buffer = {
            'observations': [],
            'actions': []
        }  # For rendering
        self._iteration = 0
        self._data_generator.rewind()
        self._total_reward = 0
        self._total_pnl = 0
        self._position = self._positions['flat']
        self._entry_price = 0
        self._exit_price = 0

        for i in range(self._history_length):
            # TODO: self._prices_history and self._buffer['observations'] are redondant
            self._prices_history.append(self._data_generator.next())
            observation = self._get_observation()
            self.state_shape = observation.shape
            self._buffer['observations'].append(observation)
            self._action = self._actions['hold']
            self._buffer['actions'].append(self._action)
        return observation

    def step(self, action):
        """Take an action (buy/sell/hold) and computes the immediate reward.
        """

        assert any([(action == x).all() for x in self._actions.values()])
        self._action = action
        self._buffer['actions'].append(self._action)
        self._iteration += 1

        reward = -self._time_fee
        instant_pnl = 0
        info = {}
        done = False
        if all(action == self._actions['buy']):
            reward -= self._trading_fee
            if all(self._position == self._positions['flat']):
                self._position = self._positions['long']
                self._entry_price = calc_spread(
                    self._prices_history[-1], self._spread_coefficients)[1]  # Ask
                info['entry_price'] = self._entry_price
                info['action'] = ['hold', 'buy', 'sell'][list(action).index(1)]
            elif all(self._position == self._positions['short']):
                self._exit_price = calc_spread(
                    self._prices_history[-1], self._spread_coefficients)[1]  # Ask
                info['exit_price'] = self._exit_price
                info['action'] = ['hold', 'buy', 'sell'][list(action).index(1)]
                instant_pnl = self._calc_int_pnl()
                info['instant_pnl'] = instant_pnl
                self._position = self._positions['flat']
                self._entry_price = 0
        elif all(action == self._actions['sell']):
            reward -= self._trading_fee
            if all(self._position == self._positions['flat']):
                self._position = self._positions['short']
                self._entry_price = calc_spread(
                    self._prices_history[-1], self._spread_coefficients)[0]  # Bid
                info['entry_price'] = self._entry_price
                info['action'] = ['hold', 'buy', 'sell'][list(action).index(1)]
            elif all(self._position == self._positions['long']):
                self._exit_price = calc_spread(
                    self._prices_history[-1], self._spread_coefficients)[0]  # Bid
                info['exit_price'] = self._exit_price
                info['action'] = ['hold', 'buy', 'sell'][list(action).index(1)]
                instant_pnl = self._calc_int_pnl()
                info['instant_pnl'] = instant_pnl
                self._position = self._positions['flat']
                self._entry_price = 0

        reward += instant_pnl
        self._total_pnl += instant_pnl
        self._total_reward += reward

        # Game over logic
        try:
            self._prices_history.append(self._data_generator.next())
        except StopIteration:
            done = True
            info['status'] = 'No more data.'
        if self._iteration >= self._game_length:
            done = True
            info['status'] = 'Time out.'

        observation = self._get_observation()
        self._buffer['observations'].append(observation)
        return observation, reward, done, info

    def render(self, savefig=False, filename='myfig'):
        """Matlplotlib rendering of each step.

        Args:
            savefig (bool): Whether to save the figure as an image or not.
            filename (str): Name of the image file.
        """
        iterations = np.arange(max(0, self._iteration + self._history_length - 80),
                               self._iteration + self._history_length)

        if self._first_render:
            self._f, self._ax = plt.subplots(
                len(self._spread_coefficients) + int(len(self._spread_coefficients) > 1),
                sharex=True
            )
            if len(self._spread_coefficients) == 1:
                self._ax = [self._ax]
            self._f.set_size_inches(12, 6)
            self._first_render = False
        if len(self._spread_coefficients) > 1:
            # TODO: To be checked
            for prod_i in range(len(self._spread_coefficients)):
                bid = np.array(
                    map(lambda i: self._buffer['observations'][i][-4 - 2 * len(self._spread_coefficients) + 2 * prod_i], iterations))
                ask = np.array(
                    map(lambda i: self._buffer['observations'][i][-4 - 2 * len(self._spread_coefficients) + 2 * prod_i + 1], iterations))
                self._ax[prod_i].clear()
                self._ax[prod_i].scatter(
                    iterations, bid, color='white', marker='_', linewidth=3)
                self._ax[prod_i].scatter(
                    iterations, ask, color='white', marker='_', linewidth=3)
                self._ax[prod_i].set_title('Product {} (spread coef {})'.format(
                    prod_i, str(self._spread_coefficients[prod_i])))

        actions = np.array(map(lambda i: self._buffer['actions'][i], iterations))

        sell_indices = np.where(map(np.all, actions == np.array(self._actions['sell'])))
        buy_indices = np.where(map(np.all, actions == np.array(self._actions['buy'])))
        # Spread price
        prices = map(lambda i: self._buffer['observations'][i]
                     [-4 - 2 * len(self._spread_coefficients):-4], iterations)
        bid_ask = map(lambda price: calc_spread(price, self._spread_coefficients), prices)
        bid = np.array(map(lambda x: x[0], bid_ask))
        ask = np.array(map(lambda x: x[1], bid_ask))
        self._ax[-1].clear()
        self._ax[-1].scatter(iterations, bid, color='white', marker='_')
        self._ax[-1].scatter(iterations, ask, color='white', marker='_')

        ymin, ymax = self._ax[-1].get_ylim()
        yrange = ymax - ymin
        self._ax[-1].scatter(iterations[sell_indices] - 1,
                             bid[map(lambda x: x - 1, sell_indices)] + 0.03 * yrange, color='orangered', marker='v')
        self._ax[-1].scatter(iterations[buy_indices] - 1,
                             ask[map(lambda x: x - 1, buy_indices)] - 0.03 * yrange, color='lawngreen', marker='^')
        plt.suptitle('Cumulated Reward: ' + "%.2f" % self._total_reward + ' ~ ' +
                     'Cumulated PnL: ' + "%.2f" % self._total_pnl + ' ~ ' +
                     'Position: ' + ['flat', 'long', 'short'][list(self._position).index(1)] + ' ~ ' +
                     'Entry Price: ' + "%.2f" % self._entry_price)
        self._f.tight_layout()
        plt.xticks(range(iterations[-1])[::5])
        plt.xlim([iterations[0], iterations[-1]])
        plt.subplots_adjust(top=0.85)
        plt.pause(0.01)
        if savefig:
            plt.savefig(filename)

    def _get_observation(self):
        """Concatenate all necessary elements to create the observation.

        Returns:
            numpy.array: observation array.
        """
        return np.concatenate(
            [prices for prices in self._prices_history[-self._history_length:]] +
            [
                np.array([self._entry_price]),
                np.array(self._position)
            ]
        )

    def _calc_int_pnl(self):
        """Calculate the PnL at each position closed.

        Returns:
            float: pnl for this closed trade
        """
        if all(self._position == self._positions['long']):
            return self._exit_price - self._entry_price
        if all(self._position == self._positions['short']):
            return self._entry_price - self._exit_price

    @staticmethod
    def random_action_fun():
        """The default random action for exploration.
        We hold 80% of the time and buy or sell 10% of the time each.

        Returns:
            numpy.array: array with a 1 on the action index, 0 elsewhere.
        """
        return np.random.multinomial(1, [0.8, 0.1, 0.1])
