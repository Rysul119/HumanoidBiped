import gym
from gym import spaces

import pybullet as p
import pybullet_data
from pybullet_utils import bullet_client

import numpy as np
import os


# a humanoid biped custom environment class for gym
class humanoidBiped(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, action_dim, obs_dim):
        self._p = bullet_client.BulletClient(connection_mode = p.GUI) # to connect to a renderable server
        # defining action and observation space
        high = np.ones([action_dim])
        self.action_space = spaces.Box(-high, high)
        high = np.inf * np.ones([obs_dim])
        self.observation_space = spaces.Box(-high, high)

        #self.jointDict = {} # may not need
        #self.linkDict = {} # may not need
        #self.orderedJoints = [] # may not need

        self.dt = 1/240 # may have to modify?
        self.power = 100 # hard coding the power value
        self.targetX = 1e3
        self.targetY = 0
        self.electricityCost = - 2.0
        self.stallTorqueCost = - 0.1
        self.jointAtLimitCost = - 0.1
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # for loadURDF


    def robot(self, bullet_client):
        self._p = bullet_client
        linkDict = {}
        jointDict ={}
        orderedJoints = []
        # for the body iterate over all the joints store all the link instances and joint instates in the dictionary also in the ordered_joint
        for j in range(self._p.getNumJoints(self.botId)):
            self._p.setJointMotorControl2(bodyIndex = self.botId, jointIndex = j, controlMode = p.POSITION_CONTROL, positionGain = 0.1, velocityGain = 0.1, force = 0)
            jointInfo = self._p.getJointInfo(self.botId, j)
            jointName = jointInfo[1].decode('utf8')
            linkName = jointInfo[12].decode('utf8')
            # get dictionary of links
            linkDict[linkName] = Link(self._p, self.botId, j)
            # get dictionary of joints
            if (jointName[:8] != 'jointfix'):
                jointDict[jointName] = Joint(self._p, self.botId, jointName, j)
                orderedJoints.append(jointDict[jointName])
                jointDict[jointName].power_coeff = 100

        robotBase = Link(self._p, self.botId, -1) # get the robot base instance
        return linkDict, jointDict, orderedJoints, robotBase

    def set_action(self, action):
        for i, j in enumerate(self.orderedJoints):
            torque = self.power * j.power_coeff * float(np.clip(action[i], -1, 1))
            j.set_jointTorque(torque)

    def get_observation(self):
        # starting with joint
        joints = []
        for j in self.orderedJoints:
            joints.append(j.current_relativePosition())
        jointsArr = np.array(joints).flatten() # position and velocity

        self.jointSpeed = jointsArr[1::2] # using a stepping of 2 starting from 1
        self.jointAtLimit = np.count_nonzero(np.abs(jointsArr[::2])>0.99) # counting the total number of joint which have angles > 0.99

        # now starting with links
        baseXYZ = self.robotBase.current_position() # gets the xyz coordinates of the base
        baseZ = baseXYZ[2]
        baseR, self.baseP, baseY = p.getEulerFromQuaternion(self.robotBase.current_orientation())
        zDiff = baseZ - self.initialZ # define self.initialZ in the reset
        baseVel = self.robotBase.get_velocity() # get the linear velocity, what about angular one?

        linkXYZ = []
        for links in self.linkDict.values():
            linkXYZ.append(links.current_position())
        linkXYZ = np.array(linkXYZ).flatten()
        # a mean coordinate values to use as the position of the whole body
        bodyXYZ = (linkXYZ[::3].mean(), linkXYZ[1:3].mean(), baseXYZ[2]) # using the z value of the base

        self.targetTheta = np.arctan2((self.targetY - bodyXYZ[1]),(self.targetX - bodyXYZ[0]))
        targetAngle =  self.targetTheta - baseY # of the whole body
        self.targetDist = np.linalg.norm([self.targetY - bodyXYZ[1], self.targetX - bodyXYZ[0]])

        # feet contacts (with the floor or other body part) condition

        obs = np.concatenate([np.array([zDiff, np.cos(targetAngle), np.sin(targetAngle), baseR, self.baseP])] + [baseVel*0.3] + [jointsArr]) # rescaling baseVel

        obs = np.clip(obs, -5, +5) # observation size = ( 5 + 3 + numJoints*2 ,) = (5 + 3 + 12*2 , ) = (32, )

        return  obs

    def get_rewards(self, z, action):
        if(z > 0.8 and abs(self.baseP) < 1.0):
            self._alive = 1.0
        else:
            self._alive = - 1.0

        oldPotential = self.potential # self.potential in the reset
        self.potential = -self.targetDist / self.dt # change dt?
        progress = self.potential - oldPotential

        elecCost = self.electricityCost * (np.abs(action * self.jointSpeed).mean()) + self.stallTorqueCost * (np.square(action).mean())

        jointLimitCost = self.jointAtLimitCost * self.jointAtLimit

        self.rewards = [self._alive, progress, elecCost, jointLimitCost]

        totalReward = sum(self.rewards)

        return totalReward

    def get_done(self):
        return self._alive < 0 # may add self.potential == 0 as that means that it reached the target + maybe stepCounter for like 10000

    def reset(self):
        # reset the simulation < check
        self._p.resetSimulation()
        self._p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        self._p.setGravity(0, 0, -10)
        p.setTimeStep(self.dt)
        urdfRootPath = "/Users/rysul/URDFs"
        _, self.botId = p.loadMJCF(os.path.join(urdfRootPath, "biped.xml"), flags= p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS | p.URDF_GOOGLEY_UNDEFINED_COLORS )
        #self.rewards = 0
        #self.done = 0

        self.linkDict, self.jointDict, self.orderedJoints, self.robotBase = self.robot(self._p)

        # reset the joints at arbitrary position and 0 velocity
        for j in self.orderedJoints:
            j.reset_state(np.random.uniform(low= -1, high = 1), 0)

        self.initialZ = self.robotBase.current_position()[2] # modify?
        observation = self.get_observation()
        self.potential = -self.targetDist / self.dt

        # add contact features later

        self._p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

        return  observation

    def step(self, action):
        self.set_action(action) # perform the action
        self._p.stepSimulation() # do one step simulation
        state = self.get_observation()
        reward = self.get_rewards(state[0]+self.initialZ, action)
        done = self.get_done()

        return state, reward, done, {}

    def render(self):
        view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.7, 0, 0.05],
                                                          distance=.7,
                                                          yaw=90,
                                                          pitch=-70,
                                                          roll=0,
                                                          upAxisIndex=2)
        proj_matrix = p.computeProjectionMatrixFOV(fov=60,
                                                   aspect=float(960) / 720,
                                                   nearVal=0.1,
                                                   farVal=100.0)
        (_, _, px, _, _) = p.getCameraImage(width=960,
                                            height=720,
                                            viewMatrix=view_matrix,
                                            projectionMatrix=proj_matrix,
                                            renderer=p.ER_BULLET_HARDWARE_OPENGL)

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (720, 960, 4))

        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def close(self):
        self._p.disconnect()


# a link class
class Link:
    def __init__(self, bullet_client, botIndex, linkIndex):
        self._p = bullet_client
        self.botIndex = botIndex
        self.linkIndex = linkIndex
        self.initial_position = self.current_position()
        self.initial_orientation = self.current_orientation()


    def pose(self):
        if(self.linkIndex == -1):
            (x, y, z), (a, b, c, d) = self._p.getBasePositionAndOrientation(self.botIndex)
        else:
            (x, y, z), (a, b, c, d), _, _, _, _ = self._p.getLinkState(self.botIndex, self.linkIndex)

        return np.array([x, y, z, a, b, c, d])

    # cartesian
    def current_position(self):
        return self.pose()[:3]

    # quaternion
    def current_orientation(self):
        return self.pose()[3:]

    # get linear velocity of a link
    def get_velocity(self):
        if (self.linkIndex == -1):
           (vx, vy, vz), _ = self._p.getBaseVelocity(self.botIndex)
        else:
            _, _, _, _, _, _, (vx, vy, vz), _ = self._p.getLinkState(self.botIndex, self.linkIndex)

        return np.array([vx, vy, vz])

    def reset_position(self, position):
        self._p.resetBasePositionAndOrientation(self.botIndex, position, self.current_orientation())

    def reset_orientation(self, orientation):
        self._p.resetBasePositionAndOrientation(self.botIndex, self.current_position(), orientation)

    def reset_velocity(self, linearVelocity, angularVelocity):
        self._p.resetBaseVelocity(self.botIndex, linearVelocity, angularVelocity)

    # contact list

# a joint class

class Joint:
    def __init__(self, bullet_client, botIndex, jointName, jointIndex):
        self._p = bullet_client
        self.botIndex = botIndex
        self.jointName = jointName
        self.jointIndex = jointIndex

        self.jointInfo = self._p.getJointInfo(self.botIndex, self.jointIndex)
        self.lowerLimit = self.jointInfo[8]
        self.upperLimit = self.jointInfo[9]
        self.power_coeff = 0

    def get_state(self):
        x, vx, _, _ = self._p.getJointState(self.botIndex, self.jointIndex)
        return np.array([x, vx])

    def current_position(self):
        return self.get_state()[0]

    def current_relativePosition(self):
        dist, vel = self.get_state()
        dist_mid = 0.5 * (self.upperLimit + self.lowerLimit)
        return (2 * (dist - dist_mid)/(self.upperLimit - self.lowerLimit), 0.1 * vel)

    def current_velocity(self):
        return self.get_state()[1]

    def set_jointTorque(self, torque):
        self._p.setJointMotorControl2(bodyIndex = self.botIndex, jointIndex = self.jointIndex, controlMode = p.TORQUE_CONTROL, force = torque)

    def reset_state(self, position, velocity):
        self._p.resetJointState(self.botIndex, self.jointIndex, position,velocity)

    def reset_position(self, position, velocity):
        self.reset_state(position, velocity)
        self._p.setJointMotorControl2(bodyIndex = self.botIndex, jointIndex = self.jointIndex, controlMode = p.POSITION_CONTROL, targetPosition = 0, targetVelocity = 0, positionGain = 0.1, velocityGain = 0.1, force = 0)