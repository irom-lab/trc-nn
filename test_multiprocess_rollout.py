#!/usr/bin/env python

import numpy as np
import torch as pt
import sys
import torch.nn as nn

from policies import rollout, multiprocess_rollout, rollout_trajectory
import scenarios
from networks import PiNetTV, QNetTV

import seaborn as sns
import matplotlib.pyplot as plt

import ray

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
batch_size = 10

def q_net(output, prev_trv, t):
    return output, pt.rand(ntrvs)

def pi_net(trv, t):
    return -pt.tensor([trv[0] + trv[1]])

ray.init()
rollout = multiprocess_rollout(pi_net, q_net, ntrvs, scenario, horizon, batch_size)

print(rollout[0][:, :, 0])
print(rollout[1][:, :, 0])
print(rollout[2][:, :, 0])
print(rollout[3][:, :, 0])
print(rollout[4][:, :, 0])
print(rollout[5][:, 0])

# results = [rollout_trajectory.remote(pi_net, q_net, ntrvs, scenario, horizon, batch_size) for s in range(batch_size)]
#
# for r in results:
#     print(ray.get(r))
