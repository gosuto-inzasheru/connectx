import gym
import numpy as np
import random
#scipy
#pytorch

from kaggle_environments import evaluate, make, utils

from submission import agent_smit

class ConnectX(gym.Env):

    def __init__(self):
        self.env = make("connectx", debug=True)
        self.trainer = self.env.train([None, "random"])

        # Define required gym fields (examples):
        config = self.env.configuration
        self.action_space = gym.spaces.Discrete(config.columns)
        self.observation_space = gym.spaces.Discrete(config.columns * config.rows)

    def get_config(self):
        return self.env.configuration

    def step(self, action):
        return self.trainer.step(action)

    def reset(self):
        return self.trainer.reset()

    def render(self, **kwargs):
        return self.env.render(**kwargs)

def loop():
    env = ConnectX()
    conf = env.get_config()
    obs = env.reset()
    done = False

    def find_pref(matrix, pref):
        found = False
        rows = matrix.shape[0]

        for r in reversed(range(rows)):
            if matrix[r][pref] == 0:
                found = True
                break
        if found == False:
            return find_pref(matrix, pref + random.choice([-1, 1]))
        return pref


    def act(obs, conf):
        matrix = np.array(obs.board).reshape(conf.rows, conf.columns)
        centre_col = conf.columns // 2
        return find_pref(matrix, centre_col)

    while not done:
        action = agent_smit(obs, conf)
        obs, reward, done, info = env.step(action)
        matrix = np.array(obs.board).reshape(conf.rows, conf.columns)

    print(matrix)


loop()
