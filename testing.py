import numpy as np
from humanoidBiped import  humanoidBiped

actionDim = 12
obsDim = 32
bot = humanoidBiped(actionDim, obsDim)
bot.reset()
for i in range(10000):
    act = [np.random.uniform(low=-1, high = 1) for _ in range(12)]
    obs, r, done, _ = bot.step(action=act)
    print("Observations: {} \n, reward: {} \n, done: {}\n".format(obs, r, done))