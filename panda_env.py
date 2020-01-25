import os
import sys
#sys.path.insert(0, 'src')
#sys.path.insert(0, '../utils')
#sys.path.insert(0, 'utils')

import copy
import matplotlib.pyplot as plt
from utils_3d import *
from utils_depth import *
from panda import Panda
import pybullet_data
from pkg_resources import parse_version
import random
import pybullet as p
import time
import numpy as np
from gym.utils import seeding
from gym import spaces
import gym
import math as m
import cv2

# import cv2  # video/image


class pandaEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'],
                'video.frames_per_second': 50}

    def __init__(self,
                 objPath="mesh.urdf",
                 urdfRoot=pybullet_data.getDataPath(),
                 objPos=[0.50, 0, 0],
                 objOrn=[0, 0, 0],
                 objMass=0.1,
                 imgInd=0,
                 mu=0.5,
                 sigma=0.1,
                 draw=False,
                 jitterDepth=False):
        self._objPath = objPath
        self._objPos = objPos
        self._objOrn = objOrn
        self._urdfRoot = urdfRoot
        self._timeStep = 1./240.
        self._envStepCounter = 0
        self._params = getParameters()
        self._draw = draw  # depends if using GUI
        self._jitterDepth = jitterDepth

        self._blockId = None
        self._planeId = None
        self._tableId = None

        # Dynamics parameters, torsional coeff: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=897778
        # Contacts: https://github.com/bulletphysics/bullet3/blob/master/data/cube_soft.urdf, https://github.com/bulletphysics/bullet3/blob/master/data/quadruped/minitaur_v1.urdf, https://github.com/bulletphysics/bullet3/blob/master/data/cube_no_friction.urdf, https://github.com/bulletphysics/bullet3/blob/master/data/cube_gripper_left.urdf
        self._lateralFrictionCoeff = mu
        self._spinningFrictionCoeff = sigma
        self._rollingFrictionCoeff = 0

        # return info
        self._list_contact_timestep = []
        self._success = False
        self._truncated_depth = None
        self._pcl = None
        self._blockPosInitial = None
        self._graspPos = None
        self._graspOrn = None
        self._blockPos = None
        self._blockOrn = None

        # self._seed()
        #self.reset_env()

    # def _seed(self, seed=None):
        # self.np_random, seed = seeding.np_random(seed)
        # return [seed]

    def load_arm(self, physicsClientId=0):
        # Load arm, no need to settle (joint angle set instantly)
        self._panda = Panda()
        self._panda.load(physicsClientId=physicsClientId)

        # Set friction coefficients for objects and gripper fingers
        p.changeDynamics(self._panda.pandaId, self._panda.pandaLeftFingerLinkIndex, lateralFriction=self._lateralFrictionCoeff,spinningFriction=self._spinningFrictionCoeff, rollingFriction=self._rollingFrictionCoeff, physicsClientId=physicsClientId)
        p.changeDynamics(self._panda.pandaId, self._panda.pandaRightFingerLinkIndex, lateralFriction=self._lateralFrictionCoeff,spinningFriction=self._spinningFrictionCoeff, rollingFriction=self._rollingFrictionCoeff, physicsClientId=physicsClientId)

        # i = p.getDynamicsInfo(self._blockId, -1)
        # print(i)

    def reset_arm_joints_ik(self, pos, orn, physicsClientId=0):
        jointPoses = list(p.calculateInverseKinematics(self._panda.pandaId, self._panda.pandaEndEffectorLinkIndex, pos, orn, jointDamping=self._panda.jd, residualThreshold=1e-5, physicsClientId=physicsClientId))

        jointPoses = jointPoses[:7] + [0, 0, self._panda.fingerOpenPos, 0.00, self._panda.fingerOpenPos, 0.00]

        self._panda.reset(jointPoses, physicsClientId=physicsClientId)

    def reset_arm_joints(self, joints, physicsClientId=0):
        jointPoses = joints + [0, -np.pi/4, self._panda.fingerOpenPos,
                0.00, self._panda.fingerOpenPos, 0.00]
        self._panda.reset(jointPoses, physicsClientId=physicsClientId)

    def reset_env(self, physicsClientId=0):
        p.resetSimulation(physicsClientId=physicsClientId)
        p.setPhysicsEngineParameter(numSolverIterations=150, physicsClientId=physicsClientId)
        p.setTimeStep(self._timeStep, physicsClientId=physicsClientId)

        # Set gravity
        p.setGravity(0, 0, -9.81, physicsClientId=physicsClientId)

        # Load plane and table
        self._planeId = p.loadURDF(self._urdfRoot+'/plane.urdf', basePosition=[0, 0, -1], useFixedBase=1, physicsClientId=physicsClientId)
        self._tableId = p.loadURDF(self._urdfRoot+'/table/table.urdf', basePosition=[0.4000000, 0.00000, -0.63+0.005], baseOrientation=[0, 0, 0, 1.0], useFixedBase=1, physicsClientId=physicsClientId)

        # Load object, no specified orientation yet, PyBullet re-calculates inertia from imported mass and volume
        orn = p.getQuaternionFromEuler(self._objOrn, physicsClientId=physicsClientId)
        self._blockId = p.loadURDF(self._objPath, basePosition=[self._objPos[0], self._objPos[1], self._objPos[2]], baseOrientation=[orn[0], orn[1], orn[2], orn[3]], physicsClientId=physicsClientId)

        # Let the object settle
        for _ in range(50):
            p.stepSimulation(physicsClientId=physicsClientId)

        # Get block position
        self._blockPosInitial, _ = p.getBasePositionAndOrientation(self._blockId, physicsClientId=physicsClientId)

        # Get depth
        self._obs = self.getDepth(physicsClientId=physicsClientId)

    def move(self, pos, orn, maxJointVel, steps, physicsClientId=0):
        jointPoses = list(p.calculateInverseKinematics(self._panda.pandaId, self._panda.pandaEndEffectorLinkIndex, pos, orn, jointDamping=self._panda.jd, residualThreshold=1e-5, physicsClientId=physicsClientId))

        # Move arm
        for steps in range(steps):
            # arm joints
            for i in range(self._panda.numJointsArm):
                p.setJointMotorControl2(self._panda.pandaId, i, p.POSITION_CONTROL, targetPosition=jointPoses[i], positionGain=0.25, velocityGain=0.75, force=self._panda.maxJointForce, maxVelocity=maxJointVel, physicsClientId=physicsClientId)

            # gripper joints
            p.setJointMotorControl2(self._panda.pandaId,  self._panda.pandaLeftFingerJointIndex, p.POSITION_CONTROL,targetPosition=self._panda.fingerOpenPos, positionGain=0.25,  velocityGain=0.75, force=self._panda.maxFingerForce, physicsClientId=physicsClientId)
            p.setJointMotorControl2(self._panda.pandaId, self._panda.pandaRightFingerJointIndex, p.POSITION_CONTROL,targetPosition=self._panda.fingerOpenPos, positionGain=0.25, velocityGain=0.75, force=self._panda.maxFingerForce, physicsClientId=physicsClientId)

            # Step simulation
            p.stepSimulation(physicsClientId=physicsClientId)

    # Implements open-loop grasping
    def grasp(self, maxFingerVel, steps, physicsClientId=0):
        # Record grasp/block pos/orn
        self._graspPos, self._graspOrn = self._panda.getEE(physicsClientId=physicsClientId)
        self._blockPos, self._blockOrn = p.getBasePositionAndOrientation(self._blockId, physicsClientId=physicsClientId)

        # armPoses = self._panda.getArmJoints()

        for step in range(steps):
            # for i in range(self._panda.numJointsArm):
            #     p.setJointMotorControl2(self._panda.pandaId, i, p.POSITION_CONTROL, targetPosition=armPoses[i], positionGain=0.25, velocityGain=0.75, force=self._panda.maxJointForce, maxVelocity=maxJointVel)

            # gripper joints
            p.setJointMotorControl2(self._panda.pandaId,
                                    self._panda.pandaLeftFingerJointIndex,
                                    p.POSITION_CONTROL,
                                    targetPosition=self._panda.fingerClosedPos,
                                    force=self._panda.maxFingerForce,
                                    maxVelocity=maxFingerVel,
                                    physicsClientId=physicsClientId)
            p.setJointMotorControl2(self._panda.pandaId,
                                    self._panda.pandaRightFingerJointIndex,
                                    p.POSITION_CONTROL,
                                    targetPosition=self._panda.fingerClosedPos,
                                    force=self._panda.maxFingerForce,
                                    maxVelocity=maxFingerVel,
                                    physicsClientId=physicsClientId)

            p.stepSimulation(physicsClientId=physicsClientId)
            # time.sleep(0.01)

    # Implements lifting up
    def lift(self, height, maxJointVel, maxFingerVel, steps, physicsClientId=0):
        pos, orn = self._panda.getEE(physicsClientId=physicsClientId)  # keep current x/y when lifting
        pos[2] = height

        jointPoses = list(p.calculateInverseKinematics(self._panda.pandaId, self._panda.pandaEndEffectorLinkIndex, pos, orn, jointDamping=self._panda.jd, residualThreshold=1e-5, physicsClientId=physicsClientId))

        for _ in range(steps):
            # arm joints
            for i in range(self._panda.numJointsArm):
                p.setJointMotorControl2(self._panda.pandaId, i, p.POSITION_CONTROL, targetPosition=jointPoses[i], positionGain=0.25, velocityGain=0.75, force=self._panda.maxJointForce, maxVelocity=maxJointVel, physicsClientId=physicsClientId)

            # gripper joints
            p.setJointMotorControl2(self._panda.pandaId,
                                    self._panda.pandaLeftFingerJointIndex,
                                    p.POSITION_CONTROL,
                                    targetPosition=self._panda.fingerClosedPos,
                                    force=self._panda.maxFingerForce,
                                    maxVelocity=maxFingerVel,
                                    physicsClientId=physicsClientId)
            p.setJointMotorControl2(self._panda.pandaId,
                                    self._panda.pandaRightFingerJointIndex,
                                    p.POSITION_CONTROL,
                                    targetPosition=self._panda.fingerClosedPos,
                                    force=self._panda.maxFingerForce,
                                    maxVelocity=maxFingerVel,
                                    physicsClientId=physicsClientId)

            p.stepSimulation(physicsClientId=physicsClientId)


        # Check if grasp was successful
        left_contacts = p.getContactPoints(self._panda.pandaId, self._blockId, linkIndexB=-1, linkIndexA=self._panda.pandaLeftFingerLinkIndex, physicsClientId=physicsClientId)
        right_contacts = p.getContactPoints(self._panda.pandaId, self._blockId, linkIndexB=-1, linkIndexA=self._panda.pandaRightFingerLinkIndex, physicsClientId=physicsClientId)
        left_idx = [x for x in range(len(left_contacts)) if left_contacts[x][9] > 3.0]
        right_idx = [x for x in range(len(right_contacts)) if right_contacts[x][9] > 3.0]
        self._success = (len(left_idx) > 0 and len(right_idx) > 0)

        output = {
            "success": self._success,
            "blockPos": self._blockPos,
            "blockOrn": self._blockOrn,
            "contact_timestep": self._list_contact_timestep,
            "graspPos": self._graspPos,
            "graspOrn": self._graspOrn,
            "obs": self._obs
        }

        return output

    def getDepth(self, physicsClientId=0):
        viewMat = self._params['viewMatPanda']
        projMat = self._params['projMatPanda']
        width = self._params['imgW']
        height = self._params['imgH']
        near = self._params['near']
        far = self._params['far']

        img_arr = p.getCameraImage(width=width, height=height, viewMatrix=viewMat, projectionMatrix=projMat, flags=p.ER_NO_SEGMENTATION_MASK, physicsClientId=physicsClientId)
        # depth = depth[186:186+224,208:208+224]  # center 224x224 from 640x640
        depth = img_arr[3][168:168+128, 176:176+128]  # center 128x128 from 480x480
        # depth = depth[90:90+128,96:96+128]  # center 128x128 from 320x320
        # depth = depth[64:64+96,72:72+96]  # center 96x96 from 240x240
        # depth = depth[40:40+64,48:48+64]  # center 64x64 from 160x160
        #depth = img_arr[3][154:154+150, 162:162+150]  # center 144x144 from 480x480

        depth = cv2.resize(depth, (128, 128))

        depth = far*near/(far - (far - near)*depth)

        if self._jitterDepth:
            depth += np.random.normal(0, 0.1, size=depth.shape)

        # plt.imshow(depth.clip(max=26), cmap='Greys', interpolation='nearest')
        # plt.show()

        return depth

    def checkContact(self, step, draw=False, physicsClientId=0):
        # Collect contact info between the object and fingers
        left_contacts = p.getContactPoints(self._panda.pandaId, self._blockId, linkIndexB=-1, linkIndexA=self._panda.pandaLeftFingerLinkIndex, physicsClientId=physicsClientId)
        right_contacts = p.getContactPoints(self._panda.pandaId, self._blockId, linkIndexB=-1, linkIndexA=self._panda.pandaRightFingerLinkIndex, physicsClientId=physicsClientId)

        # process any contact with normal force > 0.2N
        left_idx = [x for x in range(
            len(left_contacts)) if left_contacts[x][9] > 1.0]
        right_idx = [x for x in range(
            len(right_contacts)) if right_contacts[x][9] > 1.0]

        if len(left_idx) > 0 and len(right_idx) > 0:

            self._list_contact_timestep += [step]
