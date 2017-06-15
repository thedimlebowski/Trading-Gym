"""
Inspired from https://github.com/keon/deep-q-learning
"""
import gym
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from tgym.envs.trading import SpreadTrading

class DQNAgent:
    def __init__(self,
                state_size,
                action_size,
                episodes,
                game_length,
                memory_size=2000,
                train_interval=100,
                gamma=0.95,
                learning_rate=0.001,
                batch_size=64,
                epsilon_min = 0.01
                ):
        self.state_size = state_size
        self.action_size = action_size
        self.memory_size = memory_size
        self.memory = [None]*memory_size
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_min = epsilon_min
        self.epsilon_increment = 1.1*(1-epsilon_min)*train_interval\
                                 /(episodes*game_length)
        self.learning_rate = learning_rate
        self.train_interval = train_interval
        self.batch_size = batch_size
        self.brain = self._build_brain()
        self.i=0

    def _build_brain(self):
        """Build the agent's brain
        """
        brain = Sequential()
        nbr_layers = 24
        activation = "relu"
        brain.add(Dense(nbr_layers,
                        input_dim=self.state_size,
                        activation=activation))
        brain.add(Dense(nbr_layers, activation=activation))
        brain.add(Dense(self.action_size, activation='linear'))
        brain.compile(loss='mse',optimizer=Adam(lr=self.learning_rate))
        return brain

    def act(self, state):
        """Acting Policy of the DQNAgent
        """
        action = np.zeros(self.action_size)
        if np.random.rand() <= self.epsilon:
            action[random.randrange(self.action_size)]=1
        else:
            state = state.reshape(1,self.state_size)
            act_values = self.brain.predict(state)
            action[np.argmax(act_values[0])] = 1
        return action

    def observe(self, state, action, reward, next_state, done,warming_up=False):
        """Memory Management and training schedule of the agent
        """
        self.i = (self.i+1) % self.memory_size
        self.memory[self.i] = (state, action, reward, next_state, done)
        if (not warming_up) and (self.i % self.train_interval)==0 :
            if self.epsilon > self.epsilon_min:
                self.epsilon -= self.epsilon_increment
            return self._iterate()


    def end(self):
        pass

    def _iterate(self):
        """Training of the agent
        """
        state, action, reward, next_state, done = self._get_batches()
        reward += (self.gamma
                   * np.logical_not(done)
                   * np.amax(self.brain.predict(next_state),
                   axis=1))
        q_target = self.brain.predict(state)
        q_target[action[0],action[1]] = reward
        return self.brain.fit(state,q_target,
                       batch_size=self.batch_size,
                       epochs=1,
                       verbose=False)

    def _get_batches(self):
        """Selecting a batch of memory
           Split it into categorical subbatches
           Process action_batch into a position vector
        """
        batch = np.array(random.sample(self.memory, self.batch_size))
        state_batch = np.concatenate(batch[:,0])\
                      .reshape(self.batch_size,self.state_size)
        action_batch = np.concatenate(batch[:,1])\
                    .reshape(self.batch_size,self.action_size)
        reward_batch = batch[:,2]
        next_state_batch = np.concatenate(batch[:,3])\
                           .reshape(self.batch_size,self.state_size)
        done_batch = batch[:,4]
        # action processing
        action_batch = np.where(action_batch==1)
        return state_batch,action_batch,reward_batch,next_state_batch,done_batch


if __name__=="__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    from tgym.core import DataGenerator
    from tgym.envs import SpreadTrading
    from tgym.gens.deterministic import WavySignal
    # Instantiating the environmnent
    generator = WavySignal(period_1=25, period_2=50, epsilon=-0.5)
    episodes = 20
    game_length = 400
    trading_fee = .2
    time_fee = 0
    history_length = 2
    environment = SpreadTrading(spread_coefficients=[1],
                                data_generator=generator,
                                trading_fee=trading_fee,
                                time_fee=time_fee,
                                history_length=history_length,
                                game_length=game_length)
    state = environment.reset()
    # Instantiating the agent
    memory_size = 3000
    state_size = len(state)
    gamma = 0.96
    epsilon_min = 0.01
    batch_size = 64
    action_size = len(SpreadTrading._actions)
    train_interval = 10
    learning_rate = 0.001
    agent = DQNAgent(state_size = state_size,
                     action_size = action_size,
                     memory_size = memory_size,
                     episodes = episodes,
                     game_length = game_length,
                     train_interval = train_interval,
                     gamma = gamma,
                     learning_rate = learning_rate,
                     batch_size=batch_size,
                     epsilon_min =epsilon_min)
    # Warming up the agent
    for _ in range(memory_size):
        action = agent.act(state)
        next_state, reward, done, _ = environment.step(action)
        agent.observe(state, action, reward, next_state, done,warming_up=True)
    # Training the agent
    for ep in range(episodes):
        state = environment.reset()
        rew=0
        for _ in range(game_length):
            action = agent.act(state)
            next_state,reward,done,_ = environment.step(action)
            loss = agent.observe(state, action, reward, next_state, done)
            state = next_state
            rew+=reward
        print("Ep:"+str(ep)
              +"| rew:"+str(round(rew,2))
              +"| eps:"+str(round(agent.epsilon,2))
              +"| loss:"+str(round(loss.history["loss"][0],4)))
    # Running the agent
    done = False
    state = environment.reset()
    while not done:
        action = agent.act(state)
        state,_,done,_ = environment.step(action)
        environment.render()

