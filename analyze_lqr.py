#!/usr/bin/env python

import numpy as np
import torch as pt
import sys
import torch.nn as nn

from policies import rollout
import scenarios
from networks import PiNetTV, QNetTV

import seaborn as sns
import matplotlib.pyplot as plt

#np.random.seed(0)
#pt.manual_seed(0)

np.random.seed(0)
pt.manual_seed(0)

def sample_initial_dist():
    return np.random.uniform(0, 5)#np.random.normal(2.5, 0.1)

def sample_sensor_noise(cov):
    return 0#np.random.normal(0, np.sqrt(cov))

scenario = scenarios.LQRScenario(sample_initial_dist, lambda: sample_sensor_noise(0.001))
ntrvs = 5
horizon = 2
rollouts = 100
batch_size = 500

def make_pi_sequence(t: int):
    return nn.Sequential(
        nn.Linear(scenario.noutputs, 64),
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
states, outputs, trvs, inputs, costs = rollout(scenario, horizon, rollouts, ntrvs, q_net, pi_net)

print(costs[:, :, 0].sum(axis=0).mean())

for i in range(states.shape[2]):
    print(states[:, :, i])
    print(inputs[:, :, i])
    print(costs[:, :, i])
    print()
    print()
    print()

for t in range(horizon + 1):
    plot_data = pt.stack((states[0, t, :], t * pt.ones(rollouts))).t().numpy()
    plt.scatter(x=plot_data[:, 0], y=plot_data[:, 1])

plt.xlabel('Position [m]')
plt.ylabel('Time [s]')
plt.show()
