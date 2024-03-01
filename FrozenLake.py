"""
Universidad del Valle de Guatemala
Facultad de Ingeniería
Departamento de Ciencias de la computación
Inteligencia Artificial 

Integrantes: 
- Marlon Hernández - 15177
- Andres Emilio Quinto - 18288

Referencia de la libreria gym: https://www.gymlibrary.dev/environments/toy_text/frozen_lake/
"""
#Librarys and progressBar
from typing import NamedTuple
import numpy as np
import gym
from gym.envs.toy_text.frozen_lake import generate_random_map
from tqdm import tqdm
import pandas as pd

from typing import NamedTuple

class FrozenLakeParams(NamedTuple):
    """
    Represents the parameters for the FrozenLake environment.

    Attributes:
        total_episodes (int): The total number of episodes to run.
        learning_rate (float): The learning rate for the agent.
        gamma (float): The discount factor for future rewards.
        epsilon (float): The exploration rate for the agent.
        map_size (int): The size of the FrozenLake map.
        seed (int): The seed for random number generation.
        is_slippery (bool): Whether the environment is slippery or not.
        action_size (int): The number of possible actions in the environment.
        state_size (int): The number of possible states in the environment.
        proba_frozen (float): The probability of a tile being frozen.
    """
    total_episodes: int
    learning_rate: float
    gamma: float
    epsilon: float
    map_size: int
    seed: int
    is_slippery: bool 
    action_size: int
    state_size: int
    proba_frozen: float

params = FrozenLakeParams(
    total_episodes=10000,
    learning_rate=0.8,
    gamma=0.95,
    epsilon=0.1,
    map_size=5,
    seed=123,
    is_slippery=True,
    action_size=None,
    state_size=None,
    proba_frozen=0.9,
)

# Set the seed
rng = np.random.default_rng(params.seed)

env = gym.make(
    "FrozenLake-v1",
    is_slippery=params.is_slippery,
    render_mode="rgb_array",
    desc=generate_random_map(size=4),
)

params = params._replace(action_size=env.action_space.n)
params = params._replace(state_size=env.observation_space.n)

class Qlearning:
    """
    Q-learning algorithm implementation.
    """

    def __init__(self, learning_rate, gamma, state_size, action_size):
        """
        Initialize the Q-learning algorithm.

        Args:
        - learning_rate: The learning rate for updating the Q-values.
        - gamma: The discounting rate for future rewards.
        - state_size: The number of possible states in the environment.
        - action_size: The number of possible actions in the environment.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.reset_qtable()

    def update(self, state, action, reward, new_state):
        """
        Update the Q-value for a given state-action pair.

        Args:
        - state: The current state.
        - action: The action taken in the current state.
        - reward: The reward received for taking the action in the current state.
        - new_state: The new state reached after taking the action.

        Returns:
        - The updated Q-value for the state-action pair.
        """
        delta = (
            reward
            + self.gamma * np.max(self.qtable[new_state, :])
            - self.qtable[state, action]
        )
        q_update = self.qtable[state, action] + self.learning_rate * delta
        return q_update

    def reset_qtable(self):
        """
        Reset the Q-table to initial values.
        """
        self.qtable = np.zeros((self.state_size, self.action_size))


class EpsilonGreedy:
    """
    Epsilon-greedy exploration strategy implementation.
    """

    def __init__(self, epsilon):
        """
        Initialize the epsilon-greedy exploration strategy.

        Args:
        - epsilon: The exploration probability.
        """
        self.epsilon = epsilon

    def choose_action(self, action_space, state, qtable):
        """
        Choose an action in the current world state using epsilon-greedy strategy.

        Args:
        - action_space: The action space of the environment.
        - state: The current state.
        - qtable: The Q-table containing the Q-values for each state-action pair.

        Returns:
        - The chosen action.
        """
        explor_exploit_tradeoff = rng.uniform(0, 1)

        if explor_exploit_tradeoff < self.epsilon:
            action = action_space.sample()
        else:
            if np.all(qtable[state, :]) == qtable[state, 0]:
                action = action_space.sample()
            else:
                action = np.argmax(qtable[state, :])
        return action

def run_env():
    """
    Runs multiple episodes of the FrozenLake environment with randomly 
    generated maps, trains a QLearning agent, and collects metrics.

    The environment is instantiated with randomly generated maps. The
    QLearning agent is initialized and the EpsilonGreedy exploration strategy
    is set. The environment is then simulated for a number of episodes, 
    training the agent and collecting reward, step count, and other metrics.

    After simulation, the metrics are returned. This allows testing the 
    agent's performance over multiple episodes on different maps.
    
    Returns:
    - rewards: An array of length 'total_episodes' containing the cumulative rewards for each episode.
    - steps: An array of length 'total_episodes' containing the number of steps taken in each episode.
    - episodes: An array of length 'total_episodes' containing the episode numbers.
    - qtables: A 2D array of shape (state_size, action_size) representing the Q-table.
    - all_states: A list containing all the states visited during the simulation.
    - all_actions: A list containing all the actions taken during the simulation.
    """
    rewards = np.zeros((params.total_episodes))
    steps = np.zeros((params.total_episodes))
    episodes = np.arange(params.total_episodes)
    qtables = np.zeros((params.state_size, params.action_size))
    all_states = []
    all_actions = []
    points = 0
    learner.reset_qtable()
    for episode in tqdm(
        episodes, desc="Epochs", leave=False
    ):
        state = env.reset(seed=params.seed)[0]  # Reset the environment
        step = 0
        done = False
        total_rewards = 0

        while not done:
            action = explorer.choose_action(
                action_space=env.action_space, state=state, qtable=learner.qtable
            )

            all_states.append(state)
            all_actions.append(action)

            new_state, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated

            learner.qtable[state, action] = learner.update(
                state, action, reward, new_state
            )

            total_rewards += reward
            step += 1

            state = new_state
        
        if reward > 0:
            print("WIN")
            points +=1
        else:
            print("LOSE")
        if points == 5:
            break
        print("Total_Score: ",points)
        print("- - - - - - - - - - - - - - - - - -")
            # Print the Q-table
        print("Q-table:")
        print(learner.qtable)
        
    return rewards, steps, episodes, qtables, all_states, all_actions


res_all = pd.DataFrame()
st_all = pd.DataFrame()

# Runs the FrozenLake environment 5 times with different 
# randomly generated maps to test the QLearning agent.
for _ in range(5):
    env = gym.make(
        "FrozenLake-v1",
        is_slippery=params.is_slippery,
        render_mode="human",
        desc=generate_random_map(size=4),
    )

    params = params._replace(action_size=env.action_space.n)
    params = params._replace(state_size=env.observation_space.n)
    env.action_space.seed(
        params.seed
    )
    learner = Qlearning(
        learning_rate=params.learning_rate,
        gamma=params.gamma,
        state_size=params.state_size,
        action_size=params.action_size,
    )
    explorer = EpsilonGreedy(
        epsilon=params.epsilon,
    )

    rewards, steps, episodes, qtables, all_states, all_actions = run_env()

    env.close()