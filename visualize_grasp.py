#!/usr/bin/env python
import policies
import scenarios
from networks import PiNetTV, QNetTV, PiNetShared, QNetShared
import torch as pt
import torch.nn as nn
import numpy as np
import sys
import pybullet as pb
import ray
import cv2

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.uniform import Uniform

np.random.seed(0)
pt.manual_seed(0)
ray.init()

scenario = scenarios.GraspScenario()

def make_preprocess_net():
    return nn.Sequential(
        nn.Conv2d(1, 6, 6, stride=2),
        nn.ELU(),
        #nn.Conv2d(6, 6, 4, stride=2),
        #nn.ELU(),
        nn.Conv2d(6, 6, 4, stride=2),
    )

test_net = make_preprocess_net()
net_out_size = test_net(pt.rand((1, 1, 128, 128))).numel()
print(net_out_size)

ntrvs = 16
horizon = 1
batch_size = 10
epochs = 1000
lr = 0.001
tradeoff = -1

class Mine(nn.Module):
    def __init__(self):
        super().__init__()
        self._net = nn.Sequential(
            nn.Linear(scenario.nstates + ntrvs, 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self._net(x)

def make_pi_sequence(t: int):
    return nn.Sequential(
        nn.Linear(ntrvs, 64),
        nn.ELU(),
        nn.Linear(64, scenario.ninputs*2)
    )

def make_q_sequence(t: int):
    return nn.Sequential(
        nn.Linear(net_out_size+ntrvs, 128),
        nn.Tanh(),
        nn.Linear(128, 64),
        nn.Tanh(),
        nn.Linear(64, ntrvs*2),
        nn.Tanh()
    )

pi_net = PiNetShared(make_pi_sequence)
q_net = QNetShared(make_q_sequence, make_preprocess_net, reshape_to=scenario.image_shape)
#loaded_models = pt.load('models/Grasp4_tradeoff_-1_epoch_300_mi_0.000')
loaded_models = pt.load('models/Grasp18_tradeoff_-1_epoch_25_mi_0.000')
pi_net.load_state_dict(loaded_models['pi_net_state_dict'])
q_net.load_state_dict(loaded_models['q_net_state_dict'])

states, outputs, samples, trvs, inputs, costs = policies.multiprocess_rollout(pi_net, q_net, ntrvs, scenario, horizon, batch_size, pt.device('cpu'), True)


failures = (costs[-1, :]).flatten().nonzero()
print(costs.flatten())

if len(failures) > 0:
    print('Found a failure')
else:
    print('No failures')
    sys.exit()

state = states[:, 0, failures[0]].flatten()
input = inputs[:, 0, failures[0]].flatten()

id = pb.connect(pb.GUI)
print(f'Input: {0.1 * input.cpu().numpy() + np.array([0.5, 0, 0.06, np.pi / 6])}')
scenario.sample_initial_dist(force_state=state)
img = scenario.sensor(state, 0).reshape((128, 128))
final = scenario.dynamics(state, input, 0, id)
cv2.imwrite('failure.png', cv2.cvtColor(np.uint8(255 * img), cv2.COLOR_GRAY2BGR))
pb.disconnect()

print(state)
print(states[:, -1, failures[0]].flatten())
print(final)
