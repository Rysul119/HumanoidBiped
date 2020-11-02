# HumanoidBiped
This project is to create a custom humanoid [Gym](https://gym.openai.com/) environment for Reinforcement learning training using [Pybullet](https://pybullet.org/wordpress/) physics engine. The environment will load a [MJCF](http://mujoco.org/book/modeling.html) `xml` file for the robot model. The environment has a `observation space` of 32 values (ranging from -inf to inf) and `action space` of 12 values (ranging from -1 to 1). Both of the spaces are continuous. In order to control movement of the humanoid robot, `setJointMotorControl2` is used so that the the joints move with the given torques.
This is the humanoid biped:
<a href="url"><img src="https://github.com/Rysul119/HumanoidBiped/blob/master/assets/snap.png" align="left" height="48" width="48" ></a>
To check out the environment you can use this code:

```python
import numpy as np
from humanoidBipedEnv import  humanoidBiped

actionDim = 12
obsDim = 32
bot = humanoidBiped(actionDim, obsDim)
bot.reset()

for i in range(10000):
    act = [np.random.uniform(low=-1, high = 1) for _ in range(12)]
    obs, r, done, _ = bot.step(action=act)
    print("Observations: {} \n, reward: {} \n, done: {}\n".format(obs, r, done))

```

### To be Contd. 
Have to add more points like installation after publishing in pypi. 


