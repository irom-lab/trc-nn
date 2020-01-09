#!/usr/bin/env python
import policies
import scenarios
from networks import PiNetTV, QNetTV, PiNetShared, QNetShared
import torch as pt
import torch.nn as nn
import numpy as np
import sys

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.uniform import Uniform

np.random.seed(0)
pt.manual_seed(0)



scenario = scenarios.SimpleCameraBallScenario(robot_init_range=(-1, 1),
                                        ball_radius=0.235,
                                        camera_angle=np.pi/6,
                                        ball_x_vel_range=(-5, -3),
                                        ball_init_x=8,
                                        ball_y_vel=7.85,
                                        ball_init_y=1,
                                        camera_height=1,
                                        dt=1.0/30.0)

ntrvs = 4
horizon = scenario.horizon
tradeoff = int(sys.argv[1])
batch_size = 200
epochs = 300
lr = 0.01

print(horizon)

class Mine(nn.Module):
    def __init__(self):
        super().__init__()
        self._net = nn.Sequential(
            nn.Linear(scenario.nstates + ntrvs, 512),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        return self._net(x)

def make_pi_sequence(t: int):
    return nn.Sequential(
        nn.Linear(ntrvs, 32),
        nn.ELU(),
        nn.Linear(32, 32),
        nn.ELU(),
        nn.Linear(32, scenario.ninputs * 2)
    )

def make_q_sequence(t: int):
    return nn.Sequential(
        nn.Linear(scenario.noutputs + ntrvs, 32),
        nn.Tanh(),
        nn.Linear(32, ntrvs * 2),
        nn.Tanh()
    )


pi_net = PiNetShared(make_pi_sequence)
q_net = QNetShared(make_q_sequence)

print(f'Tradeoff: {tradeoff}')

policies.train_mine_policy(scenario, horizon, batch_size, epochs,
                      ntrvs, Mine, {'epochs' : 100},
                      q_net, pi_net, tradeoff,
                      lr, f'{scenario.name}_lr_{lr}_tradeoff_{tradeoff}',
                      save_every=25,
                      device=pt.device('cpu'))
