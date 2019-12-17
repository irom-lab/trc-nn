import torch as pt
import numpy as np
import pybullet as pb
import pybullet_data

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

        super().__init__()

    def sample_initial_dist(self):
        sample = np.inf
        while sample > 5:
            sample = self._sample_initial_dist()

        return pt.tensor([sample, 0])

    def dynamics(self, state: np.ndarray, input: np.ndarray, t: int) -> np.ndarray:
        updated_state = pt.tensor([state[0] + state[1], state[1] + input])

        if updated_state[0] < 0:
            updated_state[0] = 0
            updated_state[1] *= 0.5

        if updated_state[0] > 5:
            updated_state[1] = 0

        return updated_state

    def sensor(self, state: np.ndarray, t: int) -> np.ndarray:
        return state[0] + self._sample_sensor_noise()

    def cost(self, state: np.ndarray, input: np.ndarray, t: int) -> float:
        if state[0] > 5:
            return pt.tensor([1000.0])
        else:
            return (1 / 2) * (30 * pt.norm(state - pt.tensor([3, 0])) ** 2 + input ** 2)

    def terminal_cost(self, state: np.ndarray) -> float:
        return 10 * self.cost(state, 0, -1)

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
        return 1

class BallScenario(Scenario):
    def __init__(self, robot_init_cov: float,
                 ball_init_x: float, ball_init_y: float, ball_x_vel_mean: float,
                 ball_x_vel_cov: float, ball_y_vel: float,
                 ball_radius: float, camera_height: float, camera_angle: float,
                 dt: float=(1/30.0), mode=pb.DIRECT):
        self._robot_init_cov = robot_init_cov
        self._ball_init_x = ball_init_x
        self._ball_x_vel_mean = ball_x_vel_mean
        self._ball_x_vel_cov = ball_x_vel_cov
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

        self._text_id = pb.loadTexture('brick.jpg')
        self._back_id = pb.loadURDF('plane.urdf', globalScaling=4)
        self._floor_id = pb.loadURDF('plane.urdf')
        pb.changeVisualShape(self._back_id, -1, textureUniqueId=self._text_id)

        self._ball_vis_id = pb.createVisualShape(shapeType=pb.GEOM_SPHERE,
                                                 radius=self._ball_radius,
                                                 rgbaColor=[1, 0, 0, 1],
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
        return pt.tensor([np.random.normal(0, np.sqrt(self._robot_init_cov)),
                          self._ball_init_x,
                          self._ball_init_y,
                          0,
                          np.random.normal(self._ball_x_vel_mean, np.sqrt(self._ball_x_vel_cov)),
                          self._ball_y_vel])

    def dynamics(self, state: np.ndarray, input: np.ndarray, t: int) -> np.ndarray:
        """
        State order: [rx, bx, by, rx_dot, bx_dot, by_dot]
        """
        dt = self._dt

        return pt.tensor([state[0] + dt * state[3],
                          state[1] + dt * state[4],
                          state[2] + dt * state[5],
                          state[3] + dt * input,
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
        return pt.from_numpy(pb.getCameraImage(64, 64, view_mat, proj_mat)[2] / 255.0)

    def cost(self, state: np.ndarray, input: np.ndarray, t: int) -> float:
        pass

    def terminal_cost(self, state: np.ndarray) -> float:
        pass

    @property
    def nstates(self):
        return 6

    @property
    def ninputs(self):
        return 1

    @property
    def noutputs(self):
        return 64 * 64 * 3
