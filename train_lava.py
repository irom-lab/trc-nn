#!/usr/bin/env python
import policies
import scenarios
import torch as pt
import numpy as np
import yaml
import sys

np.random.seed(0)
pt.manual_seed(0)

#pt.autograd.set_detect_anomaly(True)

def sample_initial_dist(cov):
    return np.random.normal(3, np.sqrt(cov))

def sample_sensor_noise(cov):
    return np.random.normal(0, np.sqrt(cov))

with open(sys.argv[1]) as file:
    yaml_def = yaml.load(file, Loader=yaml.FullLoader)

    scenario = scenarios.LavaScenario(lambda: sample_initial_dist(yaml_def['scenario']['init_cov']),
                                      lambda: sample_sensor_noise(yaml_def['scenario']['sensor_cov']))

    policy = policies.RMINEPolicy(**yaml_def['policy'], scenario=scenario)
    policy.train(log=True)
