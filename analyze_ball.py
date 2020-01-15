#!/usr/bin/env python
import policies
import scenarios
from networks import PiNetTV, QNetTV, PiNetShared, QNetShared
import torch as pt
import torch.nn as nn
import numpy as np
import sys
import pybullet as pb
import matplotlib.pyplot as plt
import cv2

from policies import rollout

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.uniform import Uniform

brick_texture = 'brick2.jpg'
noise=0.25

print(brick_texture)
print(noise)


# tested for noise in [0.1, 0.3] successfully
# and with BallScenario backgroudns 2-8

scenario = scenarios.NoisyBallScenario(ball_radius=0.235, # baseball
                                            robot_init_range=(-2.0, 2.0), # (-2, 2),
                                            ball_x_vel_range=(-4.5, -4.5), # (-4.5, -4.5),
                                            ball_init_x=8,
                                            ball_y_vel=7.85,
                                            ball_init_y=1,
                                            camera_height=1,
                                            camera_angle=np.pi/6,
                                            mode=pb.DIRECT,
                                            dt=1.0/15.0,
                                            brick_texture=brick_texture,
                                            ball_color=(0, 1, 0),
                                            noise=noise)

def make_preprocess_net():
    return nn.Sequential(
        nn.Conv2d(3, 6, 4, stride=2),
        nn.ELU(),
        nn.Conv2d(6, 6, 4, stride=2),
    )

test_net = make_preprocess_net()
net_out_size = test_net(pt.rand((1, 3, 64, 64))).numel()

ntrvs = 8
horizon = scenario.horizon
batch_size = 1000

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
        nn.Linear(ntrvs, scenario.ninputs*2)

    )

def make_q_sequence(t: int):
    return nn.Sequential(
        nn.Linear(net_out_size + ntrvs, 32),
        nn.Tanh(),
        nn.Linear(32, ntrvs*2),
        nn.Tanh()
    )

pi_net = PiNetShared(make_pi_sequence)
q_net = QNetShared(make_q_sequence, make_preprocess_net, reshape_to=scenario.image_shape)

filenames = [ 'models/Ball_tradeoff_20_epoch_1_mi_6.203', 'models/good_initialization']

fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True)


for f in filenames:
    np.random.seed(1)
    pt.manual_seed(1)

    loaded_models = pt.load(f)
    pi_net.load_state_dict(loaded_models['pi_net_state_dict'])
    q_net.load_state_dict(loaded_models['q_net_state_dict'])

    states, outputs, samples, trvs, inputs, costs = rollout(pi_net, q_net, ntrvs, scenario, horizon, batch_size, pt.device('cpu'))
    total_costs = costs.sum(axis=0).detach().numpy()

    print(f'Tradeoff: {f[:10]} Mean: {total_costs.mean()},\t Std: {total_costs.std()}')
    axs.hist(total_costs, bins=30, edgecolor='black', alpha=0.5, label=f[:10])

    outputs = outputs.numpy()

    # for i in range(outputs.shape[2]):
    #     print(f'test_videos/{f.split("_")[4]}/{i}.avi')
    #     writer = cv2.VideoWriter(f'test_videos/{f.split("_")[4]}/{i}.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (64, 64))
    #
    #     for t in range(1, outputs.shape[1]):
    #         frame1 = outputs[:, t - 1, i].reshape((3, 64, 64)).transpose([1, 2, 0])
    #         frame2 = outputs[:, t, i].reshape((3, 64, 64)).transpose([1, 2, 0])
    #         frame = frame2 - frame1
    #
    #         writer.write(cv2.cvtColor(np.uint8(frame * 255), cv2.COLOR_RGB2BGR))
    #
    #     writer.release()

plt.legend()
plt.show()
