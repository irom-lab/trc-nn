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
batch_size = 400
epochs = 1000
lr = 0.00001
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

loaded_models = pt.load('models/init_grasp')
pi_net.load_state_dict(loaded_models['pi_net_state_dict'])
q_net.load_state_dict(loaded_models['q_net_state_dict'])

lowest_mi = np.inf

for tradeoff in tradeoffs:
    np.random.seed(0)
    pt.manual_seed(0)

    lowest_mi = policies.train_mine_policy(scenario, horizon, batch_size, epochs,
                          ntrvs, Mine, {'epochs' : 1000, 'batch_size' : 100},
                          q_net, pi_net, tradeoff,
                          lr, f'{scenario.name}_tradeoff_{tradeoff}',
                          save_every=25,
                          minibatch_size=batch_size,
                          opt_iters=1,
                          cutoff=0.2,
                          lowest_mi=lowest_mi,
                          pybullet=True,
                          multiprocess=True,
                          log_video_every=None,
                          device=pt.device('cuda'))
