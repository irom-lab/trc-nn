#!/usr/bin/env python
import policies
import scenarios
from networks import PiNetTV, QNetTV, PiNetShared, QNetShared
import torch as pt
import torch.nn as nn
import numpy as np
import pybullet as pb
import pybullet_data
import sys

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.uniform import Uniform
from torch.utils.tensorboard import SummaryWriter

np.random.seed(0)
pt.manual_seed(0)



rel_scenario = scenarios.RelativeBallScenario(robot_init_range=(-1, 1),
                                        ball_x_vel_range=(-5, -3),
                                        ball_init_x=8,
                                        ball_y_vel=7.85,
                                        ball_init_y=1,
                                        camera_height=1,
                                        dt=1.0/15.0)

ball_scenario = scenarios.BallScenario(ball_radius=0.235, # baseball
                                            robot_init_range=(-1, 1),
                                            ball_x_vel_range=(-5, -3),
                                            ball_init_x=8,
                                            ball_y_vel=7.85,
                                            ball_init_y=1,
                                            camera_height=1,
                                            camera_angle=np.pi/6,
                                            mode=pb.DIRECT,
                                            dt=1.0/15.0)

def relative_to_img(relative, ball_scenario):
    return ball_scenario.sensor(pt.tensor([0.0, relative[0], relative[1], 0.0, 0.0, 0.0]), 0)

def relatives_to_imgs(relatives, ball_scenario):
    images = pt.zeros((64 * 64 * 3, relatives.shape[1], relatives.shape[2]))

    for t in range(relatives.shape[1]):
        for s in range(relatives.shape[2]):
            images[:, t, s] = relative_to_img(relatives[:, t, s], ball_scenario)

    return images

ntrvs = 4
horizon = rel_scenario.horizon
lr = 0.001


train_rollout_size = 500
test_rollout_size = 50
minibatch_size = 50
epochs = 100000
device = pt.device('cuda')

def make_pi_sequence(t: int):
    return nn.Sequential(
        nn.Linear(ntrvs, 32),
        nn.Linear(32, rel_scenario.ninputs * 2)
    )

def make_q_sequence(t: int):
    return nn.Sequential(
        nn.Linear(rel_scenario.noutputs + ntrvs, 32),
        nn.Linear(32, ntrvs * 2),
        nn.Tanh()
    )

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self._cnn = nn.Sequential(
            nn.Conv2d(3, 6, 4, stride=2),
            nn.ELU(),
            nn.Conv2d(6, 6, 4, stride=2),
        )

        outsize = self._cnn(pt.zeros((1, 3, 64, 64))).numel()
        self._post_cnn = nn.Sequential(
            nn.Linear(outsize, 4),
            nn.ELU(),
            nn.Linear(4, 4)
        )

    def forward(self, img):
        return self._post_cnn(self._cnn(img).flatten())




cnn = CNN().to(device)

pi_net = PiNetShared(make_pi_sequence)
q_net = QNetShared(make_q_sequence)

loaded_models = pt.load('models/RelativeBall_lr_0.001_tradeoff_-1_epoch_350')
pi_net.load_state_dict(loaded_models['pi_net_state_dict'])
q_net.load_state_dict(loaded_models['q_net_state_dict'])

states, outputs, trvs, inputs, costs = policies.rollout(pi_net, q_net, ntrvs, rel_scenario, horizon, train_rollout_size)

states = states.to(device)
outputs = outputs.to(device)
trvs = trvs.to(device)
inputs = inputs.to(device)
costs = costs.to(device)
images = relatives_to_imgs(outputs, ball_scenario).to(device)

losses = pt.zeros((outputs.shape[1], minibatch_size), device=device)

opt = pt.optim.Adam(cnn.parameters(), lr=lr)
writer = SummaryWriter(f'runs/cnn', flush_secs=1)

for epoch in range(epochs):
    start_epoch_event = pt.cuda.Event(enable_timing=True)
    end_epoch_event = pt.cuda.Event(enable_timing=True)

    start_epoch_event.record()

    minibatch_idx = np.random.choice(range(train_rollout_size), size=minibatch_size, replace=False)

    for s in range(minibatch_size):
        for t in range(outputs.shape[1]):
            img = images[:, t, minibatch_idx[s]].reshape((1, 3, 64, 64))
            attempt = cnn(img)
            losses[t, s] = pt.norm(attempt - outputs[:, t, minibatch_idx[s]])

    loss = losses.mean()

    opt.zero_grad()
    loss.backward()
    opt.step()

    losses = losses.detach()

    end_epoch_event.record()
    pt.cuda.synchronize()
    elapsed_epoch_time = start_epoch_event.elapsed_time(end_epoch_event) / 1000

    print(f'[{epoch}: {elapsed_epoch_time}]\t\t{loss.detach().item()}')
    writer.add_scalar('Loss/Training', loss, epoch)


























print('Done')
