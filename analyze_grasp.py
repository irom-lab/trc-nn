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

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.uniform import Uniform

np.random.seed(0)
pt.manual_seed(0)
ray.init()

scenario = scenarios.GraspScenario()
noisy_scenario = scenarios.NoisyGraspScenario()

def make_preprocess_net():
    return nn.Sequential(
        nn.Conv2d(1, 6, 6, stride=2),
        nn.ELU(),
        nn.Conv2d(6, 6, 4, stride=2),
    )

test_net = make_preprocess_net()
net_out_size = test_net(pt.rand((1, 1, 128, 128))).numel()
print(net_out_size)

ntrvs = 16
horizon = 1
batch_size = 500
tradeoffs = [10]

class Mine(nn.Module):
    def __init__(self):
        super().__init__()
        self._net = nn.Sequential(
            nn.Linear(scenario.nstates + ntrvs, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, 1)
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

filenames = ['init_grasp', 'best/Grasp23_tradeoff_20_epoch_28_mi_2.682']

for name in filenames:
    np.random.seed(0)
    pt.manual_seed(0)

    loaded_models = pt.load(f'models/{name}')
    pi_net.load_state_dict(loaded_models['pi_net_state_dict'])
    q_net.load_state_dict(loaded_models['q_net_state_dict'])

    _, _, _, _, _, train_costs = policies.multiprocess_rollout(pi_net, q_net, ntrvs, scenario, horizon, batch_size, pybullet=True)
    _, _, _, _, _, test_costs = policies.multiprocess_rollout(pi_net, q_net, ntrvs, noisy_scenario, horizon, batch_size, pybullet=True)

    print(f'File: {name}')
    print(f'Train:\t\tMean: {train_costs.sum(axis=0).mean()}\t\tStd: {train_costs.sum(axis=0).std()}')
    print(f'Test:\t\tMean: {test_costs.sum(axis=0).mean()}\t\tStd: {test_costs.sum(axis=0).std()}')
