#!/usr/bin/env python
import policies
import scenarios
from networks import PiNetTV, QNetTV, PiNetShared, QNetShared
import torch as pt
import torch.nn as nn
import numpy as np
import sys
import pybullet as pb

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.uniform import Uniform

np.random.seed(1)
pt.manual_seed(1)

scenario = scenarios.BallScenario(ball_radius=0.235, # baseball
                                            robot_init_range=(-1, 1),
                                            ball_x_vel_range=(-5, -3),
                                            ball_init_x=8,
                                            ball_y_vel=7.85,
                                            ball_init_y=1,
                                            camera_height=1,
                                            camera_angle=np.pi/6,
                                            mode=pb.DIRECT,
                                            dt=1.0/15.0)

def make_preprocess_net():
    return nn.Sequential(
        nn.Conv2d(3, 6, 4, stride=2),
        nn.ELU(),
        nn.Conv2d(6, 6, 4, stride=2),
    )

test_net = make_preprocess_net()
net_out_size = test_net(pt.rand((1, 3, 64, 64))).numel()
print(net_out_size)

ntrvs = 8
horizon = scenario.horizon
tradeoff = int(sys.argv[1])
batch_size = 200
epochs = 5000
lr = 0.001

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
        #nn.ELU(),
        #nn.Linear(32, 32),
        #nn.ELU(),
        nn.Linear(32, scenario.ninputs * 2)
    )

def make_q_sequence(t: int):
    return nn.Sequential(
        nn.Linear(net_out_size + ntrvs, 32),
        #nn.Tanh(),
        nn.Linear(32, ntrvs * 2),
        nn.Tanh()
    )

pi_net = PiNetShared(make_pi_sequence)
q_net = QNetShared(make_q_sequence, make_preprocess_net, reshape_to=scenario.image_shape)
#q_net = QNetShared(make_q_sequence)

print(q_net(scenario.sensor(scenario.sample_initial_dist(), 0), pt.zeros(ntrvs), 0))
print(f'Tradeoff: {tradeoff}')

policies.train_mine_policy(scenario, horizon, batch_size, epochs,
                      ntrvs, Mine, {'epochs' : 100},
                      q_net, pi_net, tradeoff,
                      lr, f'{scenario.name}_lr_{lr}_tradeoff_{tradeoff}',
                      save_every=25,
                      minibatch_size=20,
                      opt_iters=10,
                      device=pt.device('cuda'))
