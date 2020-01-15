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
                                            robot_init_range=(-2.0, 2.0), # (-1, 1),
                                            ball_x_vel_range=(-4.5, -4.5), # (-5, -3),
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

        # nn.Conv2d(3, 6, 10, stride=2),
        # nn.ELU(),
        # nn.Conv2d(6, 3, 4, stride=2),
        # nn.ELU(),
        # nn.Conv2d(3,3,4, stride=2),
        # nn.ELU(),
        # nn.Conv2d(3,3,4,stride=2)

    )

test_net = make_preprocess_net()
net_out_size = test_net(pt.rand((1, 3, 64, 64))).numel()
print(net_out_size)

ntrvs = 8
horizon = scenario.horizon
#tradeoff = float(sys.argv[1])
batch_size = 200
epochs = 100
lr = 0.001 # 0.001

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
        # nn.Linear(ntrvs, 32),
        # #nn.ELU(),
        # #nn.Linear(32, 32),
        # #nn.ELU(),
        # nn.Linear(32, scenario.ninputs * 2)

        # Ani
        nn.Linear(ntrvs, scenario.ninputs*2)

    )

def make_q_sequence(t: int):
    return nn.Sequential(
        # nn.Linear(net_out_size + ntrvs, 32),
        # #nn.Tanh(),
        # nn.Linear(32, ntrvs * 2),
        # nn.Tanh()

        # Ani
        nn.Linear(net_out_size+ntrvs, 32),
        nn.Tanh(),
        nn.Linear(32, ntrvs*2),
        nn.Tanh()

    )

pi_net = PiNetShared(make_pi_sequence)
q_net = QNetShared(make_q_sequence, make_preprocess_net, reshape_to=scenario.image_shape)
#q_net = QNetShared(make_q_sequence)

# # Ani: load pre-trained weights
# loaded_models = pt.load("models/Ball8_lr_0.001_tradeoff_-1_epoch_825")
# pi_net.load_state_dict(loaded_models['pi_net_state_dict'])
# q_net.load_state_dict(loaded_models['q_net_state_dict'])

# Ani: load pre-trained weights
# loaded_models = pt.load("models/Ball17_lr_0.005_tradeoff_-1_epoch_25")
loaded_models = pt.load("models/good_initialization")
pi_net.load_state_dict(loaded_models['pi_net_state_dict'])
q_net.load_state_dict(loaded_models['q_net_state_dict'])

lowest_mi = 7.609

for tradeoff in range(18, 42, 2):
    lowest_mi = policies.train_mine_policy(scenario, horizon, batch_size, epochs,
                          ntrvs, Mine, {'epochs' : 100},
                          q_net, pi_net, tradeoff,
                          lr, f'{scenario.name}_tradeoff_{tradeoff}',
                          save_every=25,
                          minibatch_size=200,
                          opt_iters=1,
                          cutoff=23.5,
                          lowest_mi=lowest_mi,
                          device=pt.device('cuda'))
