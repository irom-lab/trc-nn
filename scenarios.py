import torch as pt
import numpy as np

from abc import ABC, abstractmethod

class Scenario(ABC):
    @abstractmethod
    def sample_initial_dist(self, n: int):
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
    def nstates(self):
        pass

    @property
    @abstractmethod
    def ninputs(self):
        pass

    @property
    @abstractmethod
    def noutputs(self):
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
    def nstates(self):
        return 2

    @property
    def ninputs(self):
        return 1

    @property
    def noutputs(self):
        return 1
