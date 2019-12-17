#!/usr/bin/env python
import policies
import scenarios
from networks import PiNetTV, QNetTV
import torch as pt
import torch.nn as nn
import numpy as np
import sys

np.random.seed(0)
pt.manual_seed(0)
ntrvs = 5
tradeoff = int(sys.argv[1])

def sample_initial_dist():
    return np.random.uniform(0, 5)#np.random.normal(2.5, 0.1)

def sample_sensor_noise(cov):
    return np.random.normal(0, np.sqrt(cov))

scenario = scenarios.LavaScenario(sample_initial_dist, lambda: sample_sensor_noise(0.001))
horizon = 5

q_net_list = [nn.Linear(scenario.noutputs + ntrvs, 64),
              nn.ELU(),
              nn.Linear(64, 64),
              nn.ELU(),
              nn.Linear(64, ntrvs * 2)]

pi_net_list = [nn.Linear(ntrvs, 64),
              nn.ELU(),
              nn.Linear(64, 64),
              nn.ELU(),
              nn.Linear(64, scenario.ninputs * 2)]

print(f'Tradeoff: {tradeoff}')

#policy = policies.MINEPolicy2(scenario, 5, 500, ntrvs, [64, 64], 0.1 * np.eye(ntrvs), [64, 64], 0.1 * np.eye(scenario.ninputs), tradeoff, {'hidden_size' : 256, 'epochs' : 100})
#policy.train(nsamples=500, training_iterations=300, qlr=0.001, pilr=0.0001)

policy = policies.MINEPolicy3(scenario, horizon, batch_size=500,
             ntrvs=5, q_net=QNetTV(q_net_list, horizon),
             pi_net=PiNetTV(pi_net_list, horizon), epochs=300, lr=0.0001,
             tradeoff=tradeoff, mine_params={'hidden_size' : 256, 'epochs' : 100})

policy.train()
