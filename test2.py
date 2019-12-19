#!/usr/bin/env python
import policies
import scenarios
from networks import PiNetTV, QNetTV, PiNet
import torch as pt
import torch.nn as nn
import numpy as np
import sys

np.random.seed(0)
pt.manual_seed(0)

def sample_initial_dist():
    return np.random.uniform(0, 5)#np.random.normal(2.5, 0.1)

def sample_sensor_noise(cov):
    return np.random.normal(0, np.sqrt(cov))

scenario = scenarios.LavaScenario(sample_initial_dist, lambda: sample_sensor_noise(0.001))
ntrvs = 5
horizon = 5
tradeoff = int(sys.argv[1])
batch_size = 500
epochs = 1000
lr = 0.0001

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
        nn.Linear(ntrvs, 64),
        nn.ELU(),
        nn.Linear(64, 64),
        nn.ELU(),
        nn.Linear(64, scenario.ninputs * 2)
    )

def make_q_sequence(t: int):
    return nn.Sequential(
        nn.Linear(scenario.noutputs + ntrvs, 64),
        nn.ELU(),
        nn.Linear(64, 64),
        nn.ELU(),
        nn.Linear(64, ntrvs * 2)
    )


pi_net = PiNetTV(make_pi_sequence, horizon)
q_net = QNetTV(make_q_sequence, horizon)

print(f'Tradeoff: {tradeoff}')

policies.train_mine_policy(scenario, horizon, batch_size, epochs,
                      ntrvs, Mine, {'epochs' : 100},
                      q_net, pi_net, tradeoff,
                      lr, f'tradeoff5_{tradeoff}')

#policy = policies.MINEPolicy3(scenario, horizon, 500, ntrvs, Mine, 100, q_net, pi_net, tradeoff)
#policy.train(nsamples=500, training_iterations=300, qlr=0.001, pilr=0.0001, tensorboard=True)
