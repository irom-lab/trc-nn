import torch as pt
import numpy as np
import pybullet as pb
import pybullet_data
import cv2
import imutils

from panda_env import pandaEnv
import pybullet as pb
from utils_3d import *
from utils_depth import *

from abc import ABC, abstractmethod

class Scenario(ABC):
    @abstractmethod
    def sample_initial_dist(self):
        pass

    @abstractmethod
    def dynamics(self, state: np.ndarray, input: np.ndarray, t: int) -> np.ndarray:
        pass

    @abstractmethod
    def sensor(self, state: np.ndarray, t: int) -> np.ndarray:
        pass

    @abstractmethod
    def cost(self, state: np.ndarray, input: np.ndarray, t: int) -> float:
        pass

    @abstractmethod
    def terminal_cost(self, state: np.ndarray) -> float:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def nstates(self) -> int:
        pass

    @property
    @abstractmethod
    def ninputs(self) -> int:
        pass

    @property
    @abstractmethod
    def noutputs(self) -> int:
        pass

class LavaScenario(Scenario):
    def __init__(self, sample_initial_dist, sample_sensor_noise):
        self._sample_initial_dist = sample_initial_dist
        self._sample_sensor_noise = sample_sensor_noise
        self.device = pt.device('cpu')

        super().__init__()

    def sample_initial_dist(self):
        sample_pos = np.inf

        while sample_pos > 5:
            sample_pos = self._sample_initial_dist()

        return pt.tensor([sample_pos, 0], device=self.device)

    def dynamics(self, state: np.ndarray, input: np.ndarray, t: int) -> np.ndarray:
        updated_state = pt.tensor([state[0] + state[1], state[1] + input])

        if updated_state[0] < 0:
            updated_state[0] = 0.0
            updated_state[1] = 0.0
        elif updated_state[0] > 5:
            updated_state[1] = 0

        return updated_state.to(device=self.device)

    def sensor(self, state: np.ndarray, t: int) -> np.ndarray:
        return state + self._sample_sensor_noise().to(device=self.device)

    def cost(self, state: np.ndarray, input: np.ndarray, t: int) -> float:
        if state[0] > 5:
            return pt.tensor([3.0], device=self.device)
        else:
            return pt.tensor([pt.norm(state - pt.tensor([3.0, 0.0]).to(device=self.device))], device=self.device) + pt.norm(input)

    def terminal_cost(self, state: np.ndarray) -> float:
        return 100 * self.cost(state, pt.zeros(self.ninputs), -1)

    @property
    def name(self) -> str:
        return 'Lava'

    @property
    def nstates(self):
        return 2

    @property
    def ninputs(self):
        return 1

    @property
    def noutputs(self):
        return 2

class LQRScenario(Scenario):
    def __init__(self, sample_initial_dist, sample_sensor_noise):
        self._sample_initial_dist = sample_initial_dist
        self._sample_sensor_noise = sample_sensor_noise

        super().__init__()

    def sample_initial_dist(self):
        return pt.tensor([self._sample_initial_dist(), 0])

    def dynamics(self, state: np.ndarray, input: np.ndarray, t: int) -> np.ndarray:
        updated_state = pt.tensor([state[0] + state[1], state[1] + input])

        return updated_state

    def sensor(self, state: np.ndarray, t: int) -> np.ndarray:
        return state + self._sample_sensor_noise()

    def cost(self, state: np.ndarray, input: np.ndarray, t: int) -> float:
        return pt.tensor([pt.norm(state - pt.tensor([3, 0])) ** 2])

    def terminal_cost(self, state: np.ndarray) -> float:
        return 100 * self.cost(state, 0, -1)

    @property
    def name(self) -> str:
        return 'LQR'

    @property
    def nstates(self):
        return 2

    @property
    def ninputs(self):
        return 1

    @property
    def noutputs(self):
        return 2

class SimpleBallScenario(Scenario):
    def __init__(self, robot_init_range: float,
                     ball_init_x: float, ball_init_y: float, ball_x_vel_range: float,
                     ball_y_vel: float, camera_height: float, dt: float=(1/30.0)):
        self._robot_init_range= robot_init_range
        self._ball_init_x = ball_init_x
        self._ball_x_vel_range = ball_x_vel_range
        self._ball_init_y = ball_init_y
        self._ball_y_vel = ball_y_vel
        self._camera_height = camera_height
        self._dt = dt
        self._gravity = 9.8
        self.device = pt.device('cpu')

    def sample_initial_dist(self):
        return pt.tensor([np.random.uniform(self._robot_init_range[0], self._robot_init_range[1]),
                          self._ball_init_x,
                          self._ball_init_y,
                          0,
                          np.random.uniform(self._ball_x_vel_range[0], self._ball_x_vel_range[1]),
                          self._ball_y_vel], device=self.device)

    def dynamics(self, state: np.ndarray, input: np.ndarray, t: int) -> np.ndarray:
        updated_state = pt.tensor([state[0] + self._dt * state[3],
                                   state[1] + self._dt * state[4],
                                   state[2] + self._dt * state[5],
                                   state[3] + input,
                                   state[4],
                                   state[5] - self._dt * self._gravity], device=self.device)

        return updated_state

    def sensor(self, state: np.ndarray, t: int) -> np.ndarray:
        return pt.clamp(pt.tensor([(state[2] - self._camera_height) / (state[1] - state[0])], device=self.device), -1000, 1000).detach()

    def cost(self, state: np.ndarray, input: np.ndarray, t: int) -> float:
        return  pt.zeros(1)#0.01 * pt.norm(input).flatten()

    def terminal_cost(self, state: np.ndarray) -> float:
        return pt.norm(state[0] - state[1]).flatten()

    @property
    def horizon(self):
        return int((2 * self._ball_y_vel / self._gravity) / self._dt)

    @property
    def name(self) -> str:
        return 'SimpleBall'

    @property
    def nstates(self):
        return 6

    @property
    def ninputs(self):
        return 1

    @property
    def noutputs(self):
        return 1



class BallScenario(Scenario):
    def __init__(self, robot_init_range: float,
                 ball_init_x: float, ball_init_y: float, ball_x_vel_range: float,
                 ball_y_vel: float, ball_radius: float, camera_height: float,
                 camera_angle: float, dt: float=(1/30.0), brick_texture='brick2.jpg', ball_color=(0, 1, 0), mode=pb.DIRECT, thirdperson=False):
        self._robot_init_range = robot_init_range
        self._ball_init_x = ball_init_x
        self._ball_x_vel_range = ball_x_vel_range
        self._ball_init_y = ball_init_y
        self._ball_y_vel = ball_y_vel
        self._ball_radius = ball_radius
        self._camera_height = camera_height
        self._camera_angle = camera_angle
        self._dt = dt
        self._gravity = 9.8
        self.device = pt.device('cpu')
        self._thirdperson = thirdperson

        pb.connect(mode)
        pb.setGravity(0, 0, 0)
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())

        self._text_id = pb.loadTexture(brick_texture)
        self._back_id = pb.loadURDF('plane.urdf', globalScaling=4)
        self._floor_id = pb.loadURDF('plane.urdf')
        pb.changeVisualShape(self._back_id, -1, textureUniqueId=self._text_id)

        if thirdperson:
            self._husky_id = pb.loadURDF('/home/vince/Code/bullet3/data/husky/husky.urdf', globalScaling=1.5)
        #    self._side_id = pb.loadURDF('plane.urdf', globalScaling=4)
        #    pb.changeVisualShape(self._side_id, -1, textureUniqueId=self._text_id)

        self._ball_vis_id = pb.createVisualShape(shapeType=pb.GEOM_SPHERE,
                                                 radius=self._ball_radius,
                                                 rgbaColor=[ball_color[0], ball_color[1], ball_color[2], 1],
                                                 specularColor=[0.4, .4, 0],
                                                 visualFramePosition=[0, 0, 0])

        self._ball_col_id = pb.createCollisionShape(shapeType=pb.GEOM_SPHERE,
                                                    radius=self._ball_radius,
                                                    collisionFramePosition=[0, 0, 0])

        self._ball_id = pb.createMultiBody(baseMass=1,
                      baseInertialFramePosition=[0, 0, 0],
                      baseCollisionShapeIndex=self._ball_col_id,
                      baseVisualShapeIndex=self._ball_vis_id,
                      basePosition=[0, 0, 0],
                      useMaximalCoordinates=True)

    def __del__(self):
        pb.disconnect()

    def sample_initial_dist(self):
        return pt.tensor([np.random.uniform(self._robot_init_range[0], self._robot_init_range[1]),
                          self._ball_init_x,
                          self._ball_init_y,
                          np.random.uniform(self._ball_x_vel_range[0], self._ball_x_vel_range[1]),
                          self._ball_y_vel], device=self.device)

    def dynamics(self, state: np.ndarray, input: np.ndarray, t: int) -> np.ndarray:
        """
        State order: [rx, bx, by, bx_dot, by_dot]
        """
        dt = self._dt

        return pt.tensor([state[0] + 0.1 * input,
                          state[1] + dt * state[3],
                          state[2] + dt * state[4],
                          state[3],
                          state[4] - dt * self._gravity], device=self.device)

    def sensor(self, state: np.ndarray, t: int) -> np.ndarray:
        po = pb.getBasePositionAndOrientation(self._ball_id)

        pb.resetBasePositionAndOrientation(self._ball_id, [state[1] - state[0], 0, state[2] - self._camera_height], [0, 0, 0, 1])
        pb.resetBasePositionAndOrientation(self._floor_id, [0, 0, -self._camera_height],
                                           pb.getQuaternionFromEuler([0, 0, 0]))
        pb.resetBasePositionAndOrientation(self._back_id, [10 - state[0], 0, self._camera_height + 40],
                                           pb.getQuaternionFromEuler([0, -np.pi / 2, 0]))

        if self._thirdperson:
           pb.resetBasePositionAndOrientation(self._husky_id, [0, 0, -self._camera_height],
                                              pb.getQuaternionFromEuler([0, 0, 0]))


        view_mat = pb.computeViewMatrix([0, 0, self._camera_height], [np.cos(self._camera_angle), 0, np.sin(self._camera_angle) + self._camera_height], [0, 0, 1])
        proj_mat = pb.computeProjectionMatrixFOV(90, 1, 0.1, 20)
        image = pb.getCameraImage(64, 64, view_mat, proj_mat)[2][:, :, :-1] / 255.0
        permuted_image = image.transpose([2, 0, 1]).reshape(1, 3, 64, 64)

        return pt.from_numpy(permuted_image).float().flatten()

    def cost(self, state: np.ndarray, input: np.ndarray, t: int) -> float:
        return  0.01 * pt.norm(input).flatten()

    def terminal_cost(self, state: np.ndarray) -> float:
        return 100 * pt.norm(state[0] - state[1]).flatten()

    @property
    def horizon(self):
        return int((2 * self._ball_y_vel / self._gravity) / self._dt)

    @property
    def name(self) -> str:
        return 'Ball'

    @property
    def nstates(self):
        return 5

    @property
    def ninputs(self):
        return 1

    @property
    def image_shape(self):
        return (1, 3, 64, 64)

    @property
    def noutputs(self):
        return 64 * 64 * 3

class NoisyBallScenario(Scenario):
    def __init__(self, robot_init_range: float,
                 ball_init_x: float, ball_init_y: float, ball_x_vel_range: float,
                 ball_y_vel: float, ball_radius: float, camera_height: float,
                 camera_angle: float, dt: float=(1/30.0), brick_texture='brick2.jpg', ball_color=(0, 1, 0), mode=pb.DIRECT, noise=0.0):
        self._robot_init_range = robot_init_range
        self._ball_init_x = ball_init_x
        self._ball_x_vel_range = ball_x_vel_range
        self._ball_init_y = ball_init_y
        self._ball_y_vel = ball_y_vel
        self._ball_radius = ball_radius
        self._camera_height = camera_height
        self._camera_angle = camera_angle
        self._dt = dt
        self._gravity = 9.8
        self.device = pt.device('cpu')
        self._noise = noise

        pb.connect(mode)
        pb.setGravity(0, 0, 0)
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())

        self._text_id = pb.loadTexture(brick_texture)
        self._back_id = pb.loadURDF('plane.urdf', globalScaling=4)
        self._floor_id = pb.loadURDF('plane.urdf')
        pb.changeVisualShape(self._back_id, -1, textureUniqueId=self._text_id)

        self._ball_vis_id = pb.createVisualShape(shapeType=pb.GEOM_SPHERE,
                                                 radius=self._ball_radius,
                                                 rgbaColor=[ball_color[0], ball_color[1], ball_color[2], 1],
                                                 specularColor=[0.4, .4, 0],
                                                 visualFramePosition=[0, 0, 0])

        self._ball_col_id = pb.createCollisionShape(shapeType=pb.GEOM_SPHERE,
                                                    radius=self._ball_radius,
                                                    collisionFramePosition=[0, 0, 0])

        self._ball_id = pb.createMultiBody(baseMass=1,
                      baseInertialFramePosition=[0, 0, 0],
                      baseCollisionShapeIndex=self._ball_col_id,
                      baseVisualShapeIndex=self._ball_vis_id,
                      basePosition=[0, 0, 0],
                      useMaximalCoordinates=True)

    def __del__(self):
        pb.disconnect()

    # def sample_initial_dist(self):
    #     return pt.tensor([np.random.uniform(self._robot_init_range[0], self._robot_init_range[1]),
    #                       self._ball_init_x,
    #                       self._ball_init_y,
    #                       0,
    #                       np.random.uniform(self._ball_x_vel_range[0], self._ball_x_vel_range[1]),
    #                       self._ball_y_vel])

    def sample_initial_dist(self):
        return pt.tensor([np.random.uniform(self._robot_init_range[0], self._robot_init_range[1]),
                          self._ball_init_x,
                          self._ball_init_y,
                          np.random.uniform(self._ball_x_vel_range[0], self._ball_x_vel_range[1]),
                          self._ball_y_vel], device=self.device)

    # def dynamics(self, state: np.ndarray, input: np.ndarray, t: int) -> np.ndarray:
    #     """
    #     State order: [rx, bx, by, rx_dot, bx_dot, by_dot]
    #     """
    #     dt = self._dt
    #
    #     return pt.tensor([state[0] + dt * state[3],
    #                       state[1] + dt * state[4],
    #                       state[2] + dt * state[5],
    #                       state[3] - input,
    #                       state[4],
    #                       state[5] - dt * self._gravity])

    def dynamics(self, state: np.ndarray, input: np.ndarray, t: int) -> np.ndarray:
        """
        State order: [rx, bx, by, bx_dot, by_dot]
        """
        dt = self._dt

        return pt.tensor([state[0] + 0.1 * input,
                          state[1] + dt * state[3],
                          state[2] + dt * state[4],
                          state[3],
                          state[4] - dt * self._gravity], device=self.device)

    def sensor(self, state: np.ndarray, t: int) -> np.ndarray:
        po = pb.getBasePositionAndOrientation(self._ball_id)

        pb.resetBasePositionAndOrientation(self._ball_id, [state[1] - state[0], 0, state[2] - self._camera_height], [0, 0, 0, 1])
        pb.resetBasePositionAndOrientation(self._floor_id, [0, 0, -self._camera_height],
                                           pb.getQuaternionFromEuler([0, 0, 0]))
        pb.resetBasePositionAndOrientation(self._back_id, [10 - state[0], 0, self._camera_height + 40],
                                           pb.getQuaternionFromEuler([0, -np.pi / 2, 0]))

        view_mat = pb.computeViewMatrix([0, 0, self._camera_height], [np.cos(self._camera_angle), 0, np.sin(self._camera_angle) + self._camera_height], [0, 0, 1])
        proj_mat = pb.computeProjectionMatrixFOV(90, 1, 0.1, 20)
        image = pb.getCameraImage(64, 64, view_mat, proj_mat)[2][:, :, :-1] / 255.0
        image += np.random.normal(scale=self._noise, size=(64, 64, 3))
        image -= image.min()
        image /= image.max()
        permuted_image = image.transpose([2, 0, 1]).reshape(1, 3, 64, 64)

        return pt.from_numpy(permuted_image).float().flatten()

    def cost(self, state: np.ndarray, input: np.ndarray, t: int) -> float:
        return  0.01 * pt.norm(input).flatten()

    def terminal_cost(self, state: np.ndarray) -> float:
        return 100 * pt.norm(state[0] - state[1]).flatten()

    @property
    def horizon(self):
        return int((2 * self._ball_y_vel / self._gravity) / self._dt)

    @property
    def name(self) -> str:
        return 'Ball'

    @property
    def nstates(self):
        return 5

    @property
    def ninputs(self):
        return 1

    @property
    def image_shape(self):
        return (1, 3, 64, 64)

    @property
    def noutputs(self):
        return 64 * 64 * 3

def find_ball_center(img):
    lower = (35, 170, 110)
    upper = (65, 255, 230)

    resized = cv2.resize(img, (256, 256))
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)

        if M['m00'] == 0:
            return -100

        return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))[1]
    else:
        return -100


class SimpleCameraBallScenario(Scenario):
    def __init__(self, robot_init_range: float,
                 ball_init_x: float, ball_init_y: float, ball_x_vel_range: float,
                 ball_y_vel: float, ball_radius: float, camera_height: float,
                 camera_angle: float, dt: float=(1/30.0), mode=pb.DIRECT):
        self._robot_init_range = robot_init_range
        self._ball_init_x = ball_init_x
        self._ball_x_vel_range = ball_x_vel_range
        self._ball_init_y = ball_init_y
        self._ball_y_vel = ball_y_vel
        self._ball_radius = ball_radius
        self._camera_height = camera_height
        self._camera_angle = camera_angle
        self._dt = dt
        self._gravity = 9.8

        pb.connect(mode)
        pb.setGravity(0, 0, 0)
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())

        self._text_id = pb.loadTexture('brick2.jpg')
        self._back_id = pb.loadURDF('plane.urdf', globalScaling=4)
        self._floor_id = pb.loadURDF('plane.urdf')
        pb.changeVisualShape(self._back_id, -1, textureUniqueId=self._text_id)

        self._ball_vis_id = pb.createVisualShape(shapeType=pb.GEOM_SPHERE,
                                                 radius=self._ball_radius,
                                                 rgbaColor=[0, 1, 0, 1],
                                                 specularColor=[0.4, .4, 0],
                                                 visualFramePosition=[0, 0, 0])

        self._ball_col_id = pb.createCollisionShape(shapeType=pb.GEOM_SPHERE,
                                                    radius=self._ball_radius,
                                                    collisionFramePosition=[0, 0, 0])

        self._ball_id = pb.createMultiBody(baseMass=1,
                      baseInertialFramePosition=[0, 0, 0],
                      baseCollisionShapeIndex=self._ball_col_id,
                      baseVisualShapeIndex=self._ball_vis_id,
                      basePosition=[0, 0, 0],
                      useMaximalCoordinates=True)

    def __del__(self):
        pb.disconnect()

    def sample_initial_dist(self):
        return pt.tensor([np.random.uniform(self._robot_init_range[0], self._robot_init_range[1]),
                          self._ball_init_x,
                          self._ball_init_y,
                          0,
                          np.random.uniform(self._ball_x_vel_range[0], self._ball_x_vel_range[1]),
                          self._ball_y_vel])

    def dynamics(self, state: np.ndarray, input: np.ndarray, t: int) -> np.ndarray:
        """
        State order: [rx, bx, by, rx_dot, bx_dot, by_dot]
        """
        dt = self._dt

        return pt.tensor([state[0] + dt * state[3],
                          state[1] + dt * state[4],
                          state[2] + dt * state[5],
                          state[3] + input,
                          state[4],
                          state[5] - dt * self._gravity])

    def sensor(self, state: np.ndarray, t: int) -> np.ndarray:
        po = pb.getBasePositionAndOrientation(self._ball_id)

        pb.resetBasePositionAndOrientation(self._ball_id, [state[1] - state[0], 0, state[2] - self._camera_height], [0, 0, 0, 1])
        pb.resetBasePositionAndOrientation(self._floor_id, [0, 0, -self._camera_height],
                                           pb.getQuaternionFromEuler([0, 0, 0]))
        pb.resetBasePositionAndOrientation(self._back_id, [10 - state[0], 0, self._camera_height + 40],
                                           pb.getQuaternionFromEuler([0, -np.pi / 2, 0]))

        view_mat = pb.computeViewMatrix([0, 0, self._camera_height], [np.cos(self._camera_angle), 0, np.sin(self._camera_angle) + self._camera_height], [0, 0, 1])
        proj_mat = pb.computeProjectionMatrixFOV(90, 1, 0.1, 20)
        image = pb.getCameraImage(64, 64, view_mat, proj_mat)[2][:, :, :-1]

        return pt.tensor([find_ball_center(image)]).float()

    def cost(self, state: np.ndarray, input: np.ndarray, t: int) -> float:
        return  0.01 * pt.norm(input).flatten()

    def terminal_cost(self, state: np.ndarray) -> float:
        return 100 * pt.norm(state[0] - state[1]).flatten()

    @property
    def horizon(self):
        return int((2 * self._ball_y_vel / self._gravity) / self._dt)

    @property
    def name(self) -> str:
        return 'CameraBall2'

    @property
    def nstates(self):
        return 6

    @property
    def ninputs(self):
        return 1

    @property
    def noutputs(self):
        return 1

class GraspScenario(Scenario):
    def __init__(self, mode=p.DIRECT):
        self._mode = mode
        self._env = pandaEnv(objPath='urdfs/mug/022_mug.urdf')

    def sample_initial_dist(self, physicsClientId=0, force_state=None):
        # state = np.array([0.5,
        #                   0.0,
        #                   0.0, 0.0])

        if force_state is None:
            state = np.array([np.random.uniform(0.45, 0.55),
                              np.random.uniform(-0.05, 0.05),
                              0.0,
                              np.random.uniform(-np.pi / 4, np.pi / 4)])
        else:
            state = force_state.detach().numpy()
        objPos = state[:3]
        objYaw = state[-1]

        self._env.reset_env(physicsClientId=physicsClientId)

        pb.resetDebugVisualizerCamera(0.8, 180, -60, [0.5, 0, 0], physicsClientId=physicsClientId)
        pb.resetBasePositionAndOrientation(self._env._blockId, posObj=objPos, ornObj=p.getQuaternionFromEuler([0, 0, objYaw]), physicsClientId=physicsClientId)

        return pt.from_numpy(state).float()

    def dynamics(self, state: np.ndarray, input: np.ndarray, t: int, physicsClientId=0) -> np.ndarray:
        numpy_state = state.cpu().numpy()
        numpy_input = 0.1 * input.cpu().numpy() + np.array([0.5, 0, 0.06, np.pi / 6])

        objPos = numpy_state[:3]
        objYaw = numpy_state[-1]

        graspPos = numpy_input[:3]
        graspYaw = numpy_input[-1]

        objOrn = np.array([0,0,objYaw])

        graspPos += np.array([0, 0 , 0.1]) # offset
        graspOrn = np.array([-1, 0, 0, 0])
        graspOrn = quatMult(graspOrn, np.array([0,0,np.sin(graspYaw)/2, np.cos(graspYaw)/2]))

        self._env.load_arm(physicsClientId=physicsClientId)
        graspPosBefore = graspPos + [0,0,0.2]
        self._env.reset_arm_joints_ik(graspPosBefore, graspOrn, physicsClientId=physicsClientId)
        graspOrn = self._env._panda.getEE(physicsClientId=physicsClientId)[1]

        if self._mode == pb.GUI:
            time.sleep(1)

        # Move to grasp location by steps
        self._env.move(graspPos, graspOrn, maxJointVel=0.15, steps=1000, physicsClientId=physicsClientId)

        if self._mode == pb.GUI:
            time.sleep(1)

        #print('Now: ', r._panda.getEE()[0])

        # Grasp
        self._env.grasp(maxFingerVel=0.05, steps=100, physicsClientId=physicsClientId)

        if self._mode == pb.GUI:
            time.sleep(1)

        # Lift, use current pos and orn check if success
        self._env.lift(height=0.3, maxJointVel=0.1, maxFingerVel=0.05, steps=500, physicsClientId=physicsClientId)

        if self._mode == pb.GUI:
            time.sleep(5)

        pos, orn = pb.getBasePositionAndOrientation(self._env._blockId, physicsClientId=physicsClientId)
        euler = pb.getEulerFromQuaternion(orn, physicsClientId=physicsClientId)

        return pt.tensor([pos[0], pos[1], pos[2], euler[2]]).float()

    def sensor(self, state: np.ndarray, t: int, physicsClientId=0) -> np.ndarray:
        depths = self._env.getDepth(physicsClientId=physicsClientId)

        return pt.from_numpy(depths.reshape(1, 1, 128, 128)).float()

    def cost(self, state: np.ndarray, input: np.ndarray, t: int, physicsClientId=0) -> float:
        return pt.tensor([0.0])

    def terminal_cost(self, state: np.ndarray, physicsClientId=0) -> float:
        if state[2] > 0.05:
            return pt.tensor([0.0])
        else:
            return pt.tensor([1.0])

    @property
    def name(self) -> str:
        return 'Grasp23'

    @property
    def image_shape(self):
        return (1, 1, 128, 128)

    @property
    def nstates(self) -> int:
        return 4

    @property
    def ninputs(self) -> int:
        return 4

    @property
    def noutputs(self) -> int:
        return 128 * 128
