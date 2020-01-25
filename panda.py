# import os,  inspect
# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(os.path.dirname(currentdir))
# os.sys.path.insert(0,parentdir)

import pybullet as p
import numpy as np
import time

class Panda:
    def __init__(self):
        # self.urdfRootPath = "geometry/franka/panda_arm_physics.urdf"
        self.urdfRootPath = "geometry/franka/panda_arm_physics_finger.urdf"
        self.pandaId = None

        self.numJoints = 13
        self.numJointsArm = 7 # Number of joints in arm (not counting hand)

        self.pandaEndEffectorLinkIndex = 8  # hand, index=7 is link8 (virtual one)

        # self.pandaLeftFingerLinkIndex = 9
        # self.pandaRightFingerLinkIndex = 10
        # self.pandaLeftFingerJointIndex = 9
        # self.pandaRightFingerJointIndex = 10
        self.pandaLeftFingerLinkIndex = 10  # lower
        self.pandaRightFingerLinkIndex = 12
        self.pandaLeftFingerJointIndex = 9
        self.pandaRightFingerJointIndex = 11

        self.maxJointForce = 70.0
        self.maxFingerForce = 20.0
        # self.maxFingerForce = 35  # office documentation says 70N continuous force, divide 2 here since we use separate motors in PB instead of a single one in actual robot
        # self.maxFingerForce = 50.0
        # self.maxJointVelocity = 0.4

        self.jd = [0.00001,0.00001,0.00001,0.00001,0.00001,0.00001,0.00001,0.00001,0.00001,0.00001,0.00001,0.00001,0.00001] # joint damping coefficient
        self.jointUpperLimit = [2.90, 1.76,	2.90, -0.07, 2.90, 3.75, 2.90]
        self.jointLowerLimit = [-2.90, -1.76, -2.90, -3.07, -2.90, -0.02, -2.90]
        self.jointRange = [5.8, 3.5, 5.8, 3, 5.8, 3.8, 5.8]
        self.jointRestPose = [0, -1.4, 0, -1.4, 0, 1.2, 0]

        self.fingerOpenPos = 0.04
        self.fingerClosedPos = 0.0

        # self.reset()

    def load(self, physicsClientId=0):
        self.pandaId = p.loadURDF(self.urdfRootPath, basePosition = [0,0,0], baseOrientation = [0,0,0,1], useFixedBase = 1, flags=(p.URDF_USE_SELF_COLLISION and p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT), physicsClientId=physicsClientId)
        # self.pandaId = p.loadURDF(self.urdfRootPath, basePosition = [0,0,0], baseOrientation = [0,0,0,1], useFixedBase = 1)
        iniJointAngles = [0, -1.4, 0, -2, 0, 1.8, 0.785,
                        0, -np.pi/4, self.fingerOpenPos, 0.00, self.fingerOpenPos, 0.00]
        self.reset(iniJointAngles, physicsClientId=physicsClientId)

    def reset(self, angles, physicsClientId=0):  # use list
        if len(angles) < self.numJoints:  # 7
            angles += [0, -np.pi/4, self.fingerOpenPos, 0.00, self.fingerOpenPos, 0.00]
        for i in range(self.numJoints):  # 13
            p.resetJointState(self.pandaId, i, angles[i], physicsClientId=physicsClientId)

    def getArmJoints(self, physicsClientId=0):  # use list
        info = p.getJointStates(self.pandaId, [0,1,2,3,4,5,6], physicsClientId=physicsClientId)
        angles = [x[0] for x in info]
        return angles

    def getEE(self, physicsClientId=0):
        info = p.getLinkState(self.pandaId, self.pandaEndEffectorLinkIndex, physicsClientId=physicsClientId)
        return np.array(info[4]), np.array(info[5])

    def getLeftFinger(self, physicsClientId=0):
        info = p.getLinkState(self.pandaId, self.pandaLeftFingerLinkIndex, physicsClientId=physicsClientId)
        return np.array(info[4]), np.array(info[5])

    def getRightFinger(self, physicsClientId=0):
        info = p.getLinkState(self.pandaId, self.pandaRightFingerLinkIndex, physicsClientId=physicsClientId)
        return np.array(info[4]), np.array(info[5])

    # def getObservation(self):
    #     observation = []
    #     state = p.getLinkState(self.pandaId, self.pandaEndEffectorLinkIndex, computeLinkVelocity = 1)
    #     pos = state[0]
    #     orn = state[1]
    #     euler = p.getEulerFromQuaternion(orn)
    #     observation.extend(list(pos))
    #     observation.extend(list(euler)) #roll, pitch, yaw
    #     velL = state[6]
    #     velA = state[7]
    #     observation.extend(list(velL))
    #     observation.extend(list(velA))

    #     jointStates = p.getJointStates(self.pandaId,range(11))
    #     jointPoses = [x[0] for x in jointStates]
    #     observation.extend(list(jointPoses))

    #     return observation
