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

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.uniform import Uniform

def sample_initial_dist():
    return Uniform(0, 5).sample()#np.random.normal(2.5, 0.1)

def sample_sensor_noise(cov):
    return pt.tensor([0])#np.random.normal(0, np.sqrt(cov))

scenario = scenarios.LavaScenario(sample_initial_dist, lambda: sample_sensor_noise(0.001))
ntrvs = 2
horizon = 5
batch_size = 100

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

loaded_models = pt.load(sys.argv[1])

pi_net.load_state_dict(loaded_models['pi_net_state_dict'])
q_net.load_state_dict(loaded_models['q_net_state_dict'])

states, outputs, trvs, inputs, costs = rollout(pi_net, q_net, ntrvs, scenario, horizon, batch_size)

for t in range(horizon + 1):
    plot_data = pt.stack((states[0, t, :], t * pt.ones(batch_size))).t().numpy()
    plt.scatter(x=plot_data[:, 0], y=plot_data[:, 1])

plt.xlabel('Position [m]')
plt.ylabel('Time [s]')
plt.show()
