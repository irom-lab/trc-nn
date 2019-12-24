import numpy as np
import torch as pt
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as multi
from torch.utils.tensorboard import SummaryWriter
import time

from scenarios import Scenario
from networks import train_mine_network

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict

from itertools import chain


def train_mine_policy(scenario: Scenario, horizon: int, batch_size: int,
                      epochs: int, ntrvs: int, mine_class: nn.Module, mine_params,
                      q_net: nn.Module, pi_net: nn.Module, tradeoff: float,
                      lr: float, tag: str=None, save_every: int=100):
    opt = pt.optim.Adam(list(pi_net.parameters()) + list(q_net.parameters()), lr=lr)
    mine = [mine_class() for t in range(horizon)]
    last_time = time.time()
    mi = pt.zeros(horizon)

    if tag is not None:
        writer = SummaryWriter(f'runs/{tag}', flush_secs=1)

    for epoch in range(epochs):
        if epoch % save_every == 0 or epoch == epochs - 1:
            print('Saving Model...')
            pt.save({'pi_net_state_dict' : pi_net.state_dict(),
                     'q_net_state_dict' : q_net.state_dict()}, f'models/{tag}_epoch_{epoch}')

        pi_log_probs = pt.zeros((horizon, batch_size))
        q_log_probs = pt.zeros((horizon, batch_size))

        states, outputs, trvs, inputs, costs = rollout(pi_net, q_net, ntrvs, scenario, horizon, batch_size)
        value = costs.sum(axis=0).mean()

        if tradeoff > -1:
            states_cuda = states.detach()
            trvs_cuda = trvs.detach()

            for t in range(horizon):
                train_mine_network(mine[t], (states_cuda[:, t, :], trvs_cuda[:, t, :]), epochs=mine_params['epochs'])

                num_datapts = states.shape[2]
                batch_size = num_datapts

                joint_batch_idx = np.random.choice(range(num_datapts), size=num_datapts, replace=False)
                marginal_batch_idx1 = np.random.choice(range(num_datapts), size=num_datapts, replace=False)
                marginal_batch_idx2 = np.random.choice(range(num_datapts), size=num_datapts, replace=False)

                joint_batch = pt.cat((states[:, t, joint_batch_idx], trvs[:, t, joint_batch_idx]), axis=0).t()
                marginal_batch = pt.cat((states[:, t, marginal_batch_idx1], trvs[:, t, marginal_batch_idx2]), axis=0).t()

                j_T = mine[t](joint_batch)
                m_T = mine[t](marginal_batch)

                mi[t] = j_T.mean() - pt.log(pt.mean(pt.exp(m_T)))


        mi_sum = mi.sum()

        new_time = time.time()
        print(f'[{epoch}: {new_time - last_time:.3f}]\t\tAvg. Cost: {value:.3f}\t\tEst. MI: {mi_sum.item():.5f}\t\tTotal: {value + tradeoff * mi_sum.item():.3f}')
        last_time = new_time

        for s in range(batch_size):
            trv = pt.zeros(ntrvs)

            for t in range(horizon):
                q_log_probs[t, s] = q_net.log_prob(trvs[:, t, s].detach(), outputs[:, t, s], trv.detach(), t)
                pi_log_probs[t, s] = pi_net.log_prob(inputs[:, t, s].detach(), trvs[:, t, s].detach(), t)
                trv = trvs[:, t, s]

        baseline = costs.sum(axis=0).mean()

        opt.zero_grad()
        loss = pt.mul(pi_log_probs.sum(axis=0), costs.sum(axis=0) - baseline).mean() + \
               pt.mul(q_log_probs.sum(axis=0), costs.sum(axis=0) - baseline).mean() + \
               tradeoff * mi_sum
        loss.backward()
        opt.step()

        if tag is not None:
            writer.add_scalar('Loss/Total', value + tradeoff * mi.sum().item(), epoch)
            writer.add_scalar('Loss/MI', mi_sum, epoch)
            writer.add_scalar('Loss/Cost', value, epoch)

        mi = mi.detach()

        if epoch == epochs - 1:
            return states, outputs, trvs, inputs, costs


def rollout(pi_net, q_net, ntrvs, scenario, horizon, batch_size):
    states = []
    outputs = []
    trvs = []
    inputs = []
    costs = []

    log_probs = pt.zeros((horizon, batch_size))

    for s in range(batch_size):
        traj_states = [scenario.sample_initial_dist().reshape((-1, 1))]
        traj_outputs = []
        traj_trvs = []
        traj_inputs = []
        traj_costs = []

        trv = pt.zeros(ntrvs, requires_grad=True).reshape((-1, 1))

        for t in range(horizon):
            traj_outputs.append(scenario.sensor(traj_states[-1].flatten(), t).reshape((-1, 1)))

            traj_outputs[t].requires_grad_(True)
            trv = q_net(traj_outputs[t].flatten(), trv.flatten(), t).reshape((-1, 1))
            traj_trvs.append(trv)

            traj_inputs.append(pi_net(traj_trvs[t].flatten(), t).reshape((-1, 1)))
            traj_costs.append(scenario.cost(traj_states[t].flatten(), traj_inputs[t].flatten(), t).reshape((-1, 1)))
            traj_states.append(scenario.dynamics(traj_states[t].flatten(), traj_inputs[t], t).reshape((-1, 1)).detach())

        traj_costs.append(scenario.terminal_cost(traj_states[-1].flatten()).reshape((-1, 1)))

        states.append(pt.cat(traj_states, axis=1))
        outputs.append(pt.cat(traj_outputs, axis=1))
        trvs.append(pt.cat(traj_trvs, axis=1))
        inputs.append(pt.cat(traj_inputs, axis=1))
        costs.append(pt.cat(traj_costs, axis=1))


    states = pt.stack(states, axis=2)
    outputs = pt.stack(outputs, axis=2)
    trvs = pt.stack(trvs, axis=2)
    inputs = pt.stack(inputs, axis=2)
    costs = pt.stack(costs, axis=2)[0, :, :].detach()



    return states, outputs, trvs, inputs, costs
