import numpy as np
import torch as pt
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as multi
import time
import ray
import pybullet as pb

from torch.utils.tensorboard import SummaryWriter
from scenarios import Scenario
from networks import train_mine_network
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Union
from itertools import chain

import IPython as ipy


def train_mine_policy(scenario: Scenario, horizon: int, batch_size: int,
                      epochs: int, ntrvs: int, mine_class: nn.Module, mine_params,
                      q_net: nn.Module, pi_net: nn.Module, tradeoff: float,
                      lr: float, tag: str=None, save_every: int=100,
                      log_video_every: Union[int, None]=None,
                      minibatch_size=0,
                      opt_iters=1,
                      lowest_mi=np.inf,
                      cutoff=np.inf,
                      multiprocess=False,
                      pybullet=False,
                      device=pt.device('cpu')):
    q_net.to(device=device)
    pi_net.to(device=device)
    opt = pt.optim.Adam(list(pi_net.parameters()) + list(q_net.parameters()), lr=lr)
    mine = [mine_class().to(device=device) for t in range(horizon)]
    last_time = time.time()
    mi = pt.zeros(horizon).to(device=device)
    mine_counter = 0

    scenario.device = pt.device('cpu')

    prev_best_value = np.inf
    current_value = np.inf

    if minibatch_size == 0:
        minibatch_size = batch_size


    if tag is not None:
        writer = SummaryWriter(f'runs/{tag}', flush_secs=1)

    for epoch in range(epochs):
        #if epoch % save_every == 0 or epoch == epochs - 1:
        start_epoch_event = pt.cuda.Event(enable_timing=True)
        end_epoch_event = pt.cuda.Event(enable_timing=True)
        end_rollout_event = pt.cuda.Event(enable_timing=True)

        start_epoch_event.record()

        pi_log_probs = pt.zeros((horizon, minibatch_size), device=device)
        q_log_probs = pt.zeros((horizon, minibatch_size), device=device)

        q_net.cpu()
        pi_net.cpu()

        if multiprocess:
            states, outputs, samples, trvs, inputs, costs = multiprocess_rollout(pi_net, q_net, ntrvs, scenario, horizon, batch_size, pt.device('cpu'), pybullet)
        else:
            states, outputs, samples, trvs, inputs, costs = rollout(pi_net, q_net, ntrvs, scenario, horizon, batch_size, pt.device('cpu'), pybullet)

        print('Input Mean:')
        print(inputs.mean(axis=2))
        print('Input Std:')
        print(inputs.std(axis=2))
        end_rollout_event.record()
        pt.cuda.synchronize()
        elapsed_rollout_time = start_epoch_event.elapsed_time(end_rollout_event) / 1000

        print(f'Rollout Time: {elapsed_rollout_time:.3f}')

        states = states.to(device)
        outputs = outputs.to(device)
        samples = samples.to(device)
        trvs = trvs.to(device)
        inputs = inputs.to(device)
        costs = costs.to(device)

        q_net.to(device)
        pi_net.to(device)

        for s in range(batch_size):
            trv = pt.zeros(ntrvs, device=device)

            for t in range(horizon):
                trvs[:, t, s] = q_net(outputs[:, t, s], trv, t, samples[:, t, s])[0]
                trv = trvs[:, t, s]

        value = costs.sum(axis=0).mean().item()

        if tradeoff > -1:
            states_mi = states.detach().cuda()
            trvs_mi = trvs.detach().cuda()

            for t in range(horizon):
                mine[t].cuda()
                if epoch == 0:
                    # GOOD: ma_rate=0.01, lr=1e-4
                    mi_values = train_mine_network(mine[t], (states_mi[:, t, :], trvs_mi[:, t, :]), epochs=500*mine_params['epochs'], unbiased=False, lr=mine_params['lr'])
                else:
                    mi_values = train_mine_network(mine[t], (states_mi[:, t, :], trvs_mi[:, t, :]), epochs=100*mine_params['epochs'], unbiased=False, lr=mine_params['lr'])

                for v in mi_values:
                    writer.add_scalar('Loss/MINE', v, mine_counter)
                    mine_counter += 1

            for t in range(horizon):
                num_datapts = states.shape[2]
                batch_size = num_datapts

                joint_batch_idx = np.random.choice(range(num_datapts), size=num_datapts, replace=False)
                marginal_batch_idx1 = np.random.choice(range(num_datapts), size=num_datapts, replace=False)
                marginal_batch_idx2 = np.random.choice(range(num_datapts), size=num_datapts, replace=False)

                joint_batch = pt.cat((states[:, t, joint_batch_idx], trvs[:, t, joint_batch_idx]), axis=0).t()
                marginal_batch = pt.cat((states[:, t, marginal_batch_idx1], trvs[:, t, marginal_batch_idx2]), axis=0).t()

                j_T = mine[t](joint_batch)
                m_T = mine[t](marginal_batch)

                #mi[t] = j_T.mean() - pt.log(pt.mean(pt.exp(m_T)))
                mi[t] = j_T.mean() - pt.logsumexp(m_T.flatten(), 0) + np.log(batch_size)

                pt.save({f'{t}' : mine[t].state_dict()}, f'models/mine_{epoch}_{t}')

                if np.isnan(mi[t].cpu().detach().numpy()):
                    print(j_T.mean())
                    print(m_T)
                    print(pt.mean(pt.exp(m_T)))
                    print(pt.log(pt.mean(pt.exp(m_T))))
                    ipy.embed()

        mi_sum = mi.sum()
        baseline = costs.sum(axis=0).mean()

        current_value = value + tradeoff * mi_sum.detach()

        if value < cutoff and mi_sum < lowest_mi:
            print('Saving Model...')
            lowest_mi = mi_sum.item()
            pt.save({'pi_net_state_dict' : pi_net.state_dict(),
                     'q_net_state_dict' : q_net.state_dict()}, f'models/{tag}_epoch_{epoch}_mi_{lowest_mi:.3f}')
        elif epoch % save_every == 0 or epoch == epochs - 1:
            print('Saving Model...')
            pt.save({'pi_net_state_dict' : pi_net.state_dict(),
                     'q_net_state_dict' : q_net.state_dict()}, f'models/{tag}_epoch_{epoch}_mi_{lowest_mi:.3f}')
        else:
            print(f'Current Best: {prev_best_value}')

        for iter in range(opt_iters):
            print(f'Computing Iteration {iter}')
            minibatch_idx = np.random.choice(range(batch_size), size=minibatch_size, replace=False)

            outputs_minibatch = outputs[:, :, minibatch_idx]
            trvs_minibatch = trvs[:, :, minibatch_idx]
            inputs_minibatch = inputs[:, :, minibatch_idx]
            costs_minibatch = costs[:, minibatch_idx]

            for s in range(minibatch_size):
                trv = pt.zeros(ntrvs, device=device)

                for t in range(horizon):
                    q_log_probs[t, s] = q_net.log_prob(trvs[:, t, s].detach(), outputs_minibatch[:, t, s], trv.detach(), t)
                    pi_log_probs[t, s] = pi_net.log_prob(inputs_minibatch[:, t, s].detach(), trvs_minibatch[:, t, s].detach(), t)
                    trv = trvs_minibatch[:, t, s]

            opt.zero_grad()
            loss = pt.mul(pi_log_probs.sum(axis=0), costs_minibatch.sum(axis=0) - baseline).mean() + \
                   pt.mul(q_log_probs.sum(axis=0), costs_minibatch.sum(axis=0) - baseline).mean() + \
                   tradeoff * mi_sum
            loss.backward()
            opt.step()

            pi_log_probs = pi_log_probs.detach()
            q_log_probs = pi_log_probs.detach()


        if tag is not None:
            writer.add_scalar('Loss/Total', value + tradeoff * mi.sum().item(), epoch)
            writer.add_scalar('Loss/MI', mi_sum, epoch)
            writer.add_scalar('Loss/Cost', value, epoch)
            writer.add_histogram('Loss/Cost Dist', costs.sum(axis=0), epoch)

            if log_video_every is not None and epoch % log_video_every == 0:
                print('Saving Video...')

                for s in range(batch_size):
                    img = outputs[:, t, s].view(128, 128)
                    writer.add_image(f'Loss/Image {s}', img, epoch, dataformats='HW')

                # best_traj_idx = pt.argmin(costs.sum(axis=0))
                # worst_traj_idx = pt.argmax(costs.sum(axis=0))
                #
                # best_traj_vid = pt.stack([pt.stack([outputs[:, t, best_traj_idx].view(1, 128, 128) for t in range(horizon)])])
                # worst_traj_vid = pt.stack([pt.stack([outputs[:, t, worst_traj_idx].view(1, 128, 128) for t in range(horizon)])])
                #
                # writer.add_video('Loss/Worst Traj', worst_traj_vid, epoch)
                # writer.add_video('Loss/Best Traj', best_traj_vid, epoch)


        mi = mi.detach()
        end_epoch_event.record()
        pt.cuda.synchronize()
        elapsed_epoch_time = start_epoch_event.elapsed_time(end_epoch_event) / 1000

        print(f'[{tradeoff}.{epoch}: {elapsed_epoch_time:.3f}]\t\tAvg. Cost: {value:.3f}\t\tEst. MI: {mi_sum.item():.5f}\t\tTotal: {value + tradeoff * mi_sum.item():.3f}\t\t Lowest MI: {lowest_mi:.3f}')

        if epoch == epochs - 1:
            return lowest_mi

@ray.remote(num_return_vals=6)
def rollout_trajectory(pi_net, q_net, ntrvs, scenario, horizon, batch_size, device=pt.device('cpu'), pybullet=False):
    if pybullet:
        pb.connect(pb.DIRECT)

    traj_states = [scenario.sample_initial_dist().reshape((-1, 1))]
    traj_outputs = []
    traj_samples = []
    traj_trvs = []
    traj_inputs = []
    traj_costs = []

    trv = pt.zeros(ntrvs, requires_grad=True, device=device).reshape((-1, 1))

    for t in range(horizon):
        traj_outputs.append(scenario.sensor(traj_states[t].flatten(), t).reshape((-1, 1)))

        traj_outputs[t].requires_grad_(True)
        trv, sample = q_net(traj_outputs[t].flatten(), trv.flatten(), t)
        traj_trvs.append(trv.reshape((-1, 1)))
        traj_samples.append(sample.reshape((-1, 1)))

        traj_inputs.append(pi_net(traj_trvs[t].flatten(), t).reshape((-1, 1)))
        traj_costs.append(scenario.cost(traj_states[t].flatten(), traj_inputs[t].flatten(), t).reshape((-1, 1)))
        traj_states.append(scenario.dynamics(traj_states[t].flatten(), traj_inputs[t].flatten(), t).reshape((-1, 1)).detach())

    traj_costs.append(scenario.terminal_cost(traj_states[-1].flatten()).reshape((-1, 1)))

    if pybullet:
        pb.disconnect()

    return pt.cat(traj_states, axis=1), pt.cat(traj_outputs, axis=1), pt.cat(traj_samples, axis=1), pt.cat(traj_trvs, axis=1), pt.cat(traj_inputs, axis=1), pt.cat(traj_costs, axis=1)

def multiprocess_rollout(pi_net, q_net, ntrvs, scenario, horizon, batch_size, device=pt.device('cpu'), pybullet=False):
    states = []
    outputs = []
    samples = []
    trvs = []
    inputs = []
    costs = []

    results = [rollout_trajectory.remote(pi_net, q_net, ntrvs, scenario, horizon, batch_size, device, pybullet) for i in range(batch_size)]

    for r in results:
        result = ray.get(r)
        states.append(result[0])
        outputs.append(result[1])
        samples.append(result[2])
        trvs.append(result[3])
        inputs.append(result[4])
        costs.append(result[5])

    states = pt.stack(states, axis=2).detach()
    outputs = pt.stack(outputs, axis=2).detach()
    samples = pt.stack(samples, axis=2).detach()
    trvs = pt.stack(trvs, axis=2).detach()
    inputs = pt.stack(inputs, axis=2).detach()
    costs = pt.stack(costs, axis=2)[0, :, :].detach()

    return states, outputs, samples, trvs, inputs, costs


def rollout(pi_net, q_net, ntrvs, scenario, horizon, batch_size, device=pt.device('cpu'), pybullet=False):
    states = []
    outputs = []
    samples = []
    trvs = []
    inputs = []
    costs = []

    for s in range(batch_size):
        traj_states = [scenario.sample_initial_dist().reshape((-1, 1))]
        traj_outputs = []
        traj_samples = []
        traj_trvs = []
        traj_inputs = []
        traj_costs = []

        trv = pt.zeros(ntrvs, requires_grad=True, device=device).reshape((-1, 1))

        for t in range(horizon):
            traj_outputs.append(scenario.sensor(traj_states[t].flatten(), t).reshape((-1, 1)))

            traj_outputs[t].requires_grad_(True)
            trv, sample = q_net(traj_outputs[t].flatten(), trv.flatten(), t)
            traj_trvs.append(trv.reshape((-1, 1)))
            traj_samples.append(sample.reshape((-1, 1)))

            traj_inputs.append(pi_net(traj_trvs[t].flatten(), t).reshape((-1, 1)))
            traj_costs.append(scenario.cost(traj_states[t].flatten(), traj_inputs[t].flatten(), t).reshape((-1, 1)))
            traj_states.append(scenario.dynamics(traj_states[t].flatten(), traj_inputs[t].flatten(), t).reshape((-1, 1)).detach())

        traj_costs.append(scenario.terminal_cost(traj_states[-1].flatten()).reshape((-1, 1)))

        states.append(pt.cat(traj_states, axis=1))
        outputs.append(pt.cat(traj_outputs, axis=1))
        samples.append(pt.cat(traj_samples, axis=1))
        trvs.append(pt.cat(traj_trvs, axis=1))
        inputs.append(pt.cat(traj_inputs, axis=1))
        costs.append(pt.cat(traj_costs, axis=1))


    states = pt.stack(states, axis=2).detach()
    outputs = pt.stack(outputs, axis=2).detach()
    samples = pt.stack(samples, axis=2).detach()
    trvs = pt.stack(trvs, axis=2).detach()
    inputs = pt.stack(inputs, axis=2).detach()
    costs = pt.stack(costs, axis=2)[0, :, :].detach()



    return states, outputs, samples, trvs, inputs, costs
