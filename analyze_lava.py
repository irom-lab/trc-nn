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

np.random.seed(0)
pt.manual_seed(0)

def sample_initial_dist():
    return Uniform(0, 5).sample()#np.random.normal(2.5, 0.1)

def sample_sensor_noise(cov):
    return MultivariateNormal(pt.zeros(2), pt.eye(2) * cov).sample()

test_covs = [0.001, 0.01, 0.1, 1]

scenario = scenarios.LavaScenario(sample_initial_dist, lambda: sample_sensor_noise(test_cov))
ntrvs = 2
horizon = 5
batch_size = 1000

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


fig, axs = plt.subplots(1, 4, sharey=True, tight_layout=True)
filenames = ['models/Lava_lr_0.0008_tradeoff_100_epoch_299',
             'models/Lava_lr_0.0008_tradeoff_0_epoch_299']

for i in range(len(test_covs)):
    test_cov = test_covs[i]

    for filename in filenames:
        pi_net = PiNetTV(make_pi_sequence, horizon)
        q_net = QNetTV(make_q_sequence, horizon)

        loaded_models = pt.load(filename)

        pi_net.load_state_dict(loaded_models['pi_net_state_dict'])
        q_net.load_state_dict(loaded_models['q_net_state_dict'])

        states, outputs, trvs, inputs, costs = rollout(pi_net, q_net, ntrvs, scenario, horizon, batch_size)

        total_costs = costs.sum(axis=0).detach().numpy()

        print(f'Mean: {total_costs.mean()},\t Std: {total_costs.std()}')
        axs[i].hist(total_costs, bins=30, edgecolor='black',  range=(20, 200), label=f'$\\beta$ = {filename.split("_")[5]}')
        axs[i].set_title(f'Sensor Cov: {test_cov}')
        axs[i].set_xlabel('Costs')

plt.legend()
plt.show()
