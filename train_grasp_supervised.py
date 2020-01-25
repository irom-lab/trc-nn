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

def make_preprocess_net():
    return nn.Sequential(
        nn.Conv2d(1, 6, 4, stride=2),
        nn.ELU(),
        nn.Conv2d(6, 6, 4, stride=2),
    )

test_net = make_preprocess_net()
net_out_size = test_net(pt.rand((1, 1, 128, 128))).numel()
print(net_out_size)

ntrvs = 8
horizon = 1
batch_size = 100
epochs = 100
lr = 0.001
tradeoff = -1


def make_pi_sequence(t: int):
    return nn.Sequential(
        nn.Linear(ntrvs, scenario.ninputs*2)
    )

def make_q_sequence(t: int):
    return nn.Sequential(
        nn.Linear(net_out_size+ntrvs, 32),
        nn.Tanh(),
        nn.Linear(32, ntrvs*2),
        nn.Tanh()

    )

pi_net = PiNetShared(make_pi_sequence)
q_net = QNetShared(make_q_sequence, make_preprocess_net, reshape_to=scenario.image_shape)

for epoch in range(epochs):
    print(epoch)
    states, outputs, samples, trvs, inputs, costs = multiprocess_rollout(pi_net, q_net, ntrvs, scenario, horizon, batch_size, pt.device('cpu'), pybullet)
