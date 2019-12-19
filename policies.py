import numpy as np
import torch as pt
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as multi
from torch.utils.tensorboard import SummaryWriter
import time

from scenarios import Scenario
from networks import PiNet, QNet, Mine, GausNet, QNet2, PiNet2, train_mine_network

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict

from itertools import chain


class Policy:
    def __init__(self, scenario: Scenario, horizon: int):
        self._scenario = scenario
        self._horizon = horizon

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def rollout(self, nsamples: int) -> Tuple[np.ndarray, np.ndarray, float]:
        pass

class PGPolicy(Policy):
    def __init__(self, scenario: Scenario, horizon: int, nsamples: int,
                 pi_net_sizes: List[int], pi_net_cov: np.ndarray):
        super().__init__(scenario, horizon)

        self._nsamples = nsamples

        self._states = pt.zeros((scenario.nstates, horizon + 1, nsamples))
        self._inputs = pt.zeros((scenario.ninputs, horizon, nsamples))
        self._outputs = pt.zeros((scenario.noutputs, horizon, nsamples))
        self._costs = pt.zeros((horizon + 1, nsamples))

        self._pi_net = [PiNet([scenario.noutputs] + pi_net_sizes + [scenario.ninputs], pt.from_numpy(pi_net_cov).float()) for t in range(horizon)]

    def train(self, training_iterations: int=10, lr=0.01, log: bool=True):
        pi_opt = optim.Adam(chain(*[pi.parameters() for pi in self._pi_net]), lr=lr)
        horizon = self._horizon
        nsamples = self._nsamples
        start_time = time.time()

        pi_log_probs = pt.zeros((self._horizon, nsamples))

        for titer in range(training_iterations):
            self.rollout(nsamples)

            total_cost = self._costs.sum(axis=0).mean()

            for t in range(horizon):
                for s in range(nsamples):
                    pi_log_probs[t, s] = self._pi_net[t].log_prob(self._inputs[:, t, s], self._outputs[:, t, s])

            baseline = self._costs.sum(axis=0).mean()

            pi_opt.zero_grad()
            loss = (pt.mul(pi_log_probs.sum(axis=0), self._costs.sum(axis=0) - baseline)).mean()
            loss.backward()
            pi_opt.step()

            pi_log_probs = pi_log_probs.detach()
            dt = time.time() - start_time

            if log:
                print('[{0}]\t\tAvg. Cost: {1:.3f}\t\t dt: {2:.3f}'.format(titer, total_cost, dt))

            start_time += dt

    def _compute_sample(self, s: int):
        self._states[:, 0, s] = self._scenario.sample_initial_dist()

        for t in range(self._horizon):
            self._outputs[:, t, s] = self._scenario.sensor(self._states[:, t, s], t)
            self._inputs[:, t, s] = self._pi_net[t](self._outputs[:, t, s])
            self._costs[t, s] = self._scenario.cost(self._states[:, t, s], self._inputs[:, t, s], t)

            self._states[:, t + 1, s] = self._scenario.dynamics(self._states[:, t, s], self._inputs[:, t, s], t)

        self._costs[-1, s] = self._scenario.terminal_cost(self._states[:, -1, s])


    def rollout(self, nsamples: int=-1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        horizon = self._horizon
        scenario = self._scenario

        if nsamples < 0:
            nsamples = self._nsamples

        for s in range(nsamples):
            self._compute_sample(s)

        self._states = self._states.detach()
        self._outputs = self._outputs.detach()
        self._inputs = self._inputs.detach()
        self._costs = self._costs.detach()

class MINEPolicy(Policy):
    def __init__(self, scenario: Scenario, horizon: int, nsamples: int, ntrvs: int,
                 q_net_sizes: List[int], q_net_cov: np.ndarray,
                 pi_net_sizes: List[int], pi_net_cov: np.ndarray,
                 tradeoff: float, mine_params):

        super().__init__(scenario, horizon)

        self._ntrvs = ntrvs
        self._tradeoff = tradeoff
        self._q_net = [QNet([scenario.noutputs] + q_net_sizes + [ntrvs], pt.from_numpy(q_net_cov).float(), False)] + [QNet([scenario.noutputs + ntrvs] + q_net_sizes + [ntrvs], pt.from_numpy(q_net_cov).float()) for t in range(horizon - 1)]
        self._pi_net = [PiNet([ntrvs] + pi_net_sizes + [scenario.ninputs], pt.from_numpy(pi_net_cov).float()) for t in range(horizon)]
        self._mine = [Mine(scenario.nstates + ntrvs, mine_params['hidden_size']) for t in range(horizon)]

    def train(self, training_iterations: int=10, qlr=0.0001, pilr=0.0001, nsamples: int=500, log: bool=True):
        #pi_opt = optim.Adam(chain(*[pi.parameters() for pi in self._pi_net]), lr=pilr)
        #q_opt = optim.Adam(chain(*[q.parameters() for q in self._q_net]), lr=qlr)
        opt = optim.Adam(chain(*([pi.parameters() for pi in self._pi_net] + [q.parameters() for q in self._q_net])), lr=pilr)
        horizon = self._horizon

        pi_log_probs = pt.zeros((self._horizon, nsamples))
        q_log_probs = pt.zeros((self._horizon, nsamples))
        mi = pt.zeros(horizon)

        for titer in range(training_iterations):
            states, outputs, trvs, inputs, costs = self.rollout(nsamples)
            states.requires_grad_(True)
            outputs.requires_grad_(True)
            inputs.requires_grad_(True)
            costs.requires_grad_(True)
            trvs.requires_grad_(True)

            total_cost = costs.sum(axis=0).sum() / nsamples

            if self._tradeoff > -1:
                states_cuda = states.detach()
                trvs_cuda = trvs.detach()

                for t in range(horizon):
                    mi[t] = self._mine[t].train((states_cuda[:, t, :], trvs_cuda[:, t, :]), iters=500, value_filter_window=2)[-1]

                    num_datapts = states.shape[2]
                    batch_size = num_datapts

                    joint_batch_idx = np.random.choice(range(num_datapts), size=num_datapts, replace=False)
                    marginal_batch_idx1 = np.random.choice(range(num_datapts), size=num_datapts, replace=False)
                    marginal_batch_idx2 = np.random.choice(range(num_datapts), size=num_datapts, replace=False)

                    joint_batch = pt.cat((states[:, t, joint_batch_idx], trvs[:, t, joint_batch_idx]), axis=0).t()
                    marginal_batch = pt.cat((states[:, t, marginal_batch_idx1], trvs[:, t, marginal_batch_idx2]), axis=0).t()

                    j_T = self._mine[t](joint_batch)
                    m_T = self._mine[t](marginal_batch)

                    mi[t] = j_T.mean() - pt.log(pt.mean(pt.exp(m_T)))

            for t in range(horizon):
                for s in range(nsamples):
                    pi_log_probs[t, s] = self._pi_net[t].log_prob(inputs[:, t, s], trvs[:, t, s])

                    if self._q_net[t]._x_tilde_prev:
                       q_log_probs[t, s] = self._q_net[t].log_prob(trvs[:, t, s], outputs[:, t, s], trvs[:, t - 1, s])
                    else:
                       q_log_probs[t, s] = self._q_net[t].log_prob(trvs[:, t, s], outputs[:, t, s])

            if log:
                print('[{0}]\t\tAvg. Cost: {1:.3f}\t\tEst. MI: {2:.3f}\t\tTotal: {3:.3f}'.format(titer, total_cost, mi.sum().item(), total_cost + self._tradeoff * mi.sum().item()))

            opt.zero_grad()

            baseline = costs.sum(axis=0).mean()
            mi_sum = mi.sum()

            loss = ((pt.mul(pi_log_probs.sum(axis=0), costs.sum(axis=0) - baseline)).sum() / nsamples + pt.mul(q_log_probs.sum(axis=0), costs.sum(axis=0) - baseline).sum() / nsamples) + self._tradeoff * mi_sum

            loss.backward()

            opt.step()

            pi_log_probs = pi_log_probs.detach()
            mi = mi.detach()
            q_log_probs = q_log_probs.detach()

    def rollout(self, nsamples):
        horizon = self._horizon
        scenario = self._scenario

        states = [] * nsamples
        trvs = [] * nsamples
        inputs = [] * nsamples
        outputs = [] * nsamples
        costs = [] * nsamples

        for s in range(nsamples):
            traj_states = [scenario.sample_initial_dist().reshape(1, -1)]
            traj_outputs = []
            traj_trvs = []
            traj_inputs = []
            traj_costs = []

            for t in range(horizon):
                traj_outputs.append(scenario.sensor(traj_states[t].flatten(), t).detach().reshape((1, -1)))

                if self._q_net[t]._x_tilde_prev:
                    traj_trvs.append(self._q_net[t](traj_outputs[t].flatten(), traj_trvs[t - 1].flatten().detach()).reshape((1, -1)))
                else:
                    traj_trvs.append(self._q_net[t](traj_outputs[t].flatten()).reshape((1, -1)))

                traj_inputs.append(self._pi_net[t](traj_trvs[t].flatten()).reshape((1, -1)))

                traj_costs.append(scenario.cost(traj_states[t].flatten(), traj_inputs[t].flatten(), t).reshape((1, -1)))
                traj_states.append(scenario.dynamics(traj_states[t].flatten(), traj_inputs[t].flatten(), t).detach().reshape((1, -1)))

            traj_costs.append(scenario.terminal_cost(traj_states[-1].flatten()).reshape((1, -1)))

            states.append(pt.cat(traj_states, 0).t())
            outputs.append(pt.cat(traj_outputs, 0).t())
            trvs.append(pt.cat(traj_trvs, 0).t())
            inputs.append(pt.cat(traj_inputs, 0).t())
            costs.append(pt.cat(traj_costs, 0).t())

        states = pt.stack(states, axis=2)
        outputs = pt.stack(outputs, axis=2)
        trvs = pt.stack(trvs, axis=2)
        inputs = pt.stack(inputs, axis=2)
        costs = pt.stack(costs, axis=2)

        return states, outputs, trvs, inputs.detach(), costs.detach()

class MINEPolicy2(Policy):
    def __init__(self, scenario: Scenario, horizon: int, nsamples: int, ntrvs: int,
                 q_net_sizes: List[int], q_net_cov: np.ndarray,
                 pi_net_sizes: List[int], pi_net_cov: np.ndarray,
                 tradeoff: float, mine_params):

        super().__init__(scenario, horizon)

        self._ntrvs = ntrvs
        self._tradeoff = tradeoff
        self._pi_net = [PiNet2([ntrvs] + pi_net_sizes + [scenario.ninputs]) for t in range(horizon)]
        self._q_net = [QNet2([scenario.noutputs] + q_net_sizes + [ntrvs], False)] + [QNet2([scenario.noutputs + ntrvs] + q_net_sizes + [ntrvs]) for t in range(horizon - 1)]
        self._mine = [Mine(scenario.nstates + ntrvs, mine_params['hidden_size']) for t in range(horizon)]
        self._mine_params = mine_params

        for pi in self._pi_net:
            print(pi)

    def train(self, training_iterations: int=10, qlr=0.0001, pilr=0.0001, nsamples: int=500, log: bool=True):
        #pi_opt = optim.Adam(chain(*[pi.parameters() for pi in self._pi_net]), lr=pilr)
        #q_opt = optim.Adam(chain(*[q.parameters() for q in self._q_net]), lr=qlr)
        opt = optim.Adam(chain(*([pi.parameters() for pi in self._pi_net] + [q.parameters() for q in self._q_net])), lr=pilr)
        horizon = self._horizon

        pi_log_probs = pt.zeros((self._horizon, nsamples))
        q_log_probs = pt.zeros((self._horizon, nsamples))
        mi = pt.zeros(horizon)

        last_time = time.time()

        writer = SummaryWriter(f'runs/tradeoff_{int(self._tradeoff)}', flush_secs=1)

        for titer in range(training_iterations):
            states, outputs, trvs, inputs, costs = self.rollout(nsamples)
            costs = costs[0, :, :]

            total_cost = costs.sum(axis=0).mean()

            if self._tradeoff > -1:
                states_cuda = states.detach()
                trvs_cuda = trvs.detach()

                for t in range(horizon):
                    self._mine[t].train((states_cuda[:, t, :], trvs_cuda[:, t, :]), iters=self._mine_params['epochs'], value_filter_window=2)[-1]

                    num_datapts = states.shape[2]
                    batch_size = num_datapts

                    joint_batch_idx = np.random.choice(range(num_datapts), size=num_datapts, replace=False)
                    marginal_batch_idx1 = np.random.choice(range(num_datapts), size=num_datapts, replace=False)
                    marginal_batch_idx2 = np.random.choice(range(num_datapts), size=num_datapts, replace=False)

                    joint_batch = pt.cat((states[:, t, joint_batch_idx], trvs[:, t, joint_batch_idx]), axis=0).t()
                    marginal_batch = pt.cat((states[:, t, marginal_batch_idx1], trvs[:, t, marginal_batch_idx2]), axis=0).t()

                    j_T = self._mine[t](joint_batch)
                    m_T = self._mine[t](marginal_batch)

                    mi[t] = j_T.mean() - pt.log(pt.mean(pt.exp(m_T)))

            for t in range(horizon):
                for s in range(nsamples):
                    pi_log_probs[t, s] = self._pi_net[t].log_prob(inputs[:, t, s], trvs[:, t, s])

                    if self._q_net[t]._x_tilde_prev:
                       q_log_probs[t, s] = self._q_net[t].log_prob(trvs[:, t, s], outputs[:, t, s], trvs[:, t - 1, s])
                    else:
                       q_log_probs[t, s] = self._q_net[t].log_prob(trvs[:, t, s], outputs[:, t, s])

            new_time = time.time()


            if log:
                print('[{0}: {1:.3f}]\t\tAvg. Cost: {2:.3f}\t\tEst. MI: {3:.3f}\t\tTotal: {4:.3f}'.format(titer, new_time - last_time, total_cost, mi.sum().item(), total_cost + self._tradeoff * mi.sum().item()))

            last_time = new_time

            opt.zero_grad()

            baseline = costs.sum(axis=0).mean()
            mi_sum = mi.sum()

            loss = ((pt.mul(pi_log_probs.sum(axis=0), costs.sum(axis=0) - baseline)).mean() + pt.mul(q_log_probs.sum(axis=0), costs.sum(axis=0) - baseline).mean()) + self._tradeoff * mi_sum

            loss.backward()

            opt.step()

            writer.add_scalar('Loss/Total', total_cost + self._tradeoff * mi.sum().item(), titer)
            writer.add_scalar('Loss/MI', mi_sum, titer)
            writer.add_scalar('Loss/Cost', total_cost, titer)

            pi_log_probs = pi_log_probs.detach()
            mi = mi.detach()
            q_log_probs = q_log_probs.detach()

    def rollout(self, nsamples):
        horizon = self._horizon
        scenario = self._scenario

        states = [] * nsamples
        trvs = [] * nsamples
        inputs = [] * nsamples
        outputs = [] * nsamples
        costs = [] * nsamples

        for s in range(nsamples):
            traj_states = [scenario.sample_initial_dist().reshape(1, -1)]
            traj_outputs = []
            traj_trvs = []
            traj_inputs = []
            traj_costs = []

            for t in range(horizon):
                traj_outputs.append(scenario.sensor(traj_states[t].flatten(), t).detach().reshape((1, -1)))

                if self._q_net[t]._x_tilde_prev:
                    traj_trvs.append(self._q_net[t](traj_outputs[t].flatten(), traj_trvs[t - 1].flatten()).reshape((1, -1)))
                else:
                    traj_trvs.append(self._q_net[t](traj_outputs[t].flatten()).reshape((1, -1)))

                traj_inputs.append(self._pi_net[t](traj_trvs[t].flatten()).reshape((1, -1)))

                traj_costs.append(scenario.cost(traj_states[t].flatten(), traj_inputs[t].flatten(), t).reshape((1, -1)))
                traj_states.append(scenario.dynamics(traj_states[t].flatten(), traj_inputs[t].flatten(), t).detach().reshape((1, -1)))

            traj_costs.append(scenario.terminal_cost(traj_states[-1].flatten()).reshape((1, -1)))

            states.append(pt.cat(traj_states, 0).t())
            outputs.append(pt.cat(traj_outputs, 0).t())
            trvs.append(pt.cat(traj_trvs, 0).t())
            inputs.append(pt.cat(traj_inputs, 0).t())
            costs.append(pt.cat(traj_costs, 0).t())

        states = pt.stack(states, axis=2)
        outputs = pt.stack(outputs, axis=2)
        trvs = pt.stack(trvs, axis=2)
        inputs = pt.stack(inputs, axis=2)
        costs = pt.stack(costs, axis=2)

        return states, outputs, trvs, inputs.detach(), costs.detach()



def train_mine_policy(scenario: Scenario, horizon: int, batch_size: int,
                      epochs: int, ntrvs: int, mine_class: nn.Module, mine_params,
                      q_net: nn.Module, pi_net: nn.Module, tradeoff: float,
                      lr: float, tag: str=None, save_every: int=100):
    opt = optim.Adam(list(pi_net.parameters()) + list(q_net.parameters()), lr=lr)

    pi_log_probs = pt.zeros((horizon, batch_size))
    q_log_probs = pt.zeros((horizon, batch_size))
    mi = pt.zeros(horizon)

    mine = [mine_class() for t in range(horizon)]

    last_time = time.time()

    if tag is not None:
        writer = SummaryWriter(f'runs/{tag}', flush_secs=1)

    for epoch in range(epochs):
        if epoch % save_every == 0:
            print('Saving Model...')
            pt.save({'pi_net_state_dict' : pi_net.state_dict(),
                     'q_net_state_dict' : q_net.state_dict()}, f'models/{tag}_epoch_{epoch}')

        states, outputs, trvs, inputs, costs = rollout(scenario, horizon, batch_size, ntrvs, q_net, pi_net)
        costs = costs[0, :, :]

        total_cost = costs.sum(axis=0).mean()

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

        for t in range(horizon):
            for s in range(batch_size):
                pi_log_probs[t, s] = pi_net.log_prob(inputs[:, t, s], trvs[:, t, s], t)

                if t > 0:
                   q_log_probs[t, s] = q_net.log_prob(trvs[:, t, s], outputs[:, t, s], trvs[:, t - 1, s], t)
                else:
                   q_log_probs[t, s] = q_net.log_prob(trvs[:, t, s], outputs[:, t, s], pt.zeros(ntrvs), t)

        new_time = time.time()
        print('[{0}: {1:.3f}]\t\tAvg. Cost: {2:.3f}\t\tEst. MI: {3:.3f}\t\tTotal: {4:.3f}'.format(epoch, new_time - last_time, total_cost, mi.sum().item(), total_cost + tradeoff * mi.sum().item()))
        last_time = new_time

        opt.zero_grad()

        baseline = costs.sum(axis=0).mean()
        mi_sum = mi.sum()
        loss = ((pt.mul(pi_log_probs.sum(axis=0), costs.sum(axis=0) - baseline)).mean() + pt.mul(q_log_probs.sum(axis=0), costs.sum(axis=0) - baseline).mean()) + tradeoff * mi_sum

        loss.backward()
        opt.step()

        if tag is not None:
            writer.add_scalar('Loss/Total', total_cost + tradeoff * mi.sum().item(), epoch)
            writer.add_scalar('Loss/MI', mi_sum, epoch)
            writer.add_scalar('Loss/Cost', total_cost, epoch)

        pi_log_probs = pi_log_probs.detach()
        mi = mi.detach()
        q_log_probs = q_log_probs.detach()


def rollout(scenario: Scenario, horizon: int, batch_size: int, ntrvs: int, q_net, pi_net):
    horizon = horizon
    scenario = scenario

    states = [] * batch_size
    trvs = [] * batch_size
    inputs = [] * batch_size
    outputs = [] * batch_size
    costs = [] * batch_size

    for s in range(batch_size):
        traj_states = [scenario.sample_initial_dist().reshape(1, -1)]
        traj_outputs = []
        traj_trvs = []
        traj_inputs = []
        traj_costs = []

        for t in range(horizon):
            traj_outputs.append(scenario.sensor(traj_states[t].flatten(), t).detach().reshape((1, -1)))

            if t > 0:
                traj_trvs.append(q_net(traj_outputs[t].flatten(), traj_trvs[t - 1].flatten(), t).reshape((1, -1)))
            else:
                traj_trvs.append(q_net(traj_outputs[t].flatten(), pt.zeros(ntrvs), t).reshape((1, -1)))

            traj_inputs.append(pi_net(traj_trvs[t].flatten(), t).reshape((1, -1)))

            traj_costs.append(scenario.cost(traj_states[t].flatten(), traj_inputs[t].flatten(), t).reshape((1, -1)))
            traj_states.append(scenario.dynamics(traj_states[t].flatten(), traj_inputs[t].flatten(), t).detach().reshape((1, -1)))

        traj_costs.append(scenario.terminal_cost(traj_states[-1].flatten()).reshape((1, -1)))

        states.append(pt.cat(traj_states, 0).t())
        outputs.append(pt.cat(traj_outputs, 0).t())
        trvs.append(pt.cat(traj_trvs, 0).t())
        inputs.append(pt.cat(traj_inputs, 0).t())
        costs.append(pt.cat(traj_costs, 0).t())

    states = pt.stack(states, axis=2)
    outputs = pt.stack(outputs, axis=2)
    trvs = pt.stack(trvs, axis=2)
    inputs = pt.stack(inputs, axis=2)
    costs = pt.stack(costs, axis=2)

    return states, outputs, trvs, inputs.detach(), costs.detach()


class MINEPolicy3(Policy):
    def __init__(self, scenario: Scenario, horizon: int, nsamples: int, ntrvs: int,
                 mine_class, mine_epochs,
                 q_net,
                 pi_net,
                 tradeoff: float):

        super().__init__(scenario, horizon)

        self._ntrvs = ntrvs
        self._tradeoff = tradeoff
        self._pi_net = pi_net#[pi_net() for t in range(horizon)]
        self._q_net = q_net #[QNet2([scenario.noutputs] + q_net_sizes + [ntrvs], False)] + [QNet2([scenario.noutputs + ntrvs] + q_net_sizes + [ntrvs]) for t in range(horizon - 1)]
        self._mine = [mine_class() for t in range(horizon)]
        self._mine_epochs = mine_epochs

    def train(self, training_iterations: int=10, qlr=0.0001, pilr=0.0001, nsamples: int=500, log: bool=True, tensorboard: bool=True):
        #pi_opt = optim.Adam(chain(*[pi.parameters() for pi in self._pi_net]), lr=pilr)
        #q_opt = optim.Adam(chain(*[q.parameters() for q in self._q_net]), lr=qlr)
        #opt = optim.Adam(list(chain(*[pi.parameters() for pi in self._pi_net])) + list(chain(*[q.parameters() for q in self._q_net])), lr=pilr)
        #opt = optim.Adam(list(self._pi_net.parameters()) + list(chain(*[q.parameters() for q in self._q_net])), lr=pilr)
        opt = optim.Adam(list(self._pi_net.parameters()) + list(self._q_net.parameters()), lr=pilr)
        horizon = self._horizon

        pi_log_probs = pt.zeros((self._horizon, nsamples))
        q_log_probs = pt.zeros((self._horizon, nsamples))
        mi = pt.zeros(horizon)

        last_time = time.time()

        if tensorboard:
            writer = SummaryWriter(f'runs/tradeoff4_{int(self._tradeoff)}', flush_secs=1)

        for titer in range(training_iterations):
            states, outputs, trvs, inputs, costs = self.rollout(nsamples)
            costs = costs[0, :, :]

            total_cost = costs.sum(axis=0).mean()

            if self._tradeoff > -1:
                states_cuda = states.detach()
                trvs_cuda = trvs.detach()

                for t in range(horizon):
                    train_mine_network(self._mine[t], (states_cuda[:, t, :], trvs_cuda[:, t, :]), epochs=self._mine_epochs)

                    num_datapts = states.shape[2]
                    batch_size = num_datapts

                    joint_batch_idx = np.random.choice(range(num_datapts), size=num_datapts, replace=False)
                    marginal_batch_idx1 = np.random.choice(range(num_datapts), size=num_datapts, replace=False)
                    marginal_batch_idx2 = np.random.choice(range(num_datapts), size=num_datapts, replace=False)

                    joint_batch = pt.cat((states[:, t, joint_batch_idx], trvs[:, t, joint_batch_idx]), axis=0).t()
                    marginal_batch = pt.cat((states[:, t, marginal_batch_idx1], trvs[:, t, marginal_batch_idx2]), axis=0).t()

                    j_T = self._mine[t](joint_batch)
                    m_T = self._mine[t](marginal_batch)

                    mi[t] = j_T.mean() - pt.log(pt.mean(pt.exp(m_T)))

            for t in range(horizon):
                for s in range(nsamples):
                    pi_log_probs[t, s] = self._pi_net.log_prob(inputs[:, t, s], trvs[:, t, s], t)

                    if t > 0:
                       q_log_probs[t, s] = self._q_net.log_prob(trvs[:, t, s], outputs[:, t, s], trvs[:, t - 1, s], t)
                    else:
                       q_log_probs[t, s] = self._q_net.log_prob(trvs[:, t, s], outputs[:, t, s], pt.zeros(self._ntrvs), t)

            new_time = time.time()


            if log:
                print('[{0}: {1:.3f}]\t\tAvg. Cost: {2:.3f}\t\tEst. MI: {3:.3f}\t\tTotal: {4:.3f}'.format(titer, new_time - last_time, total_cost, mi.sum().item(), total_cost + self._tradeoff * mi.sum().item()))

            last_time = new_time

            opt.zero_grad()

            baseline = costs.sum(axis=0).mean()
            mi_sum = mi.sum()

            loss = ((pt.mul(pi_log_probs.sum(axis=0), costs.sum(axis=0) - baseline)).mean() + pt.mul(q_log_probs.sum(axis=0), costs.sum(axis=0) - baseline).mean()) + self._tradeoff * mi_sum

            loss.backward()

            opt.step()

            if tensorboard:
                writer.add_scalar('Loss/Total', total_cost + self._tradeoff * mi.sum().item(), titer)
                writer.add_scalar('Loss/MI', mi_sum, titer)
                writer.add_scalar('Loss/Cost', total_cost, titer)

            pi_log_probs = pi_log_probs.detach()
            mi = mi.detach()
            q_log_probs = q_log_probs.detach()

    def rollout(self, nsamples):
        horizon = self._horizon
        scenario = self._scenario

        states = [] * nsamples
        trvs = [] * nsamples
        inputs = [] * nsamples
        outputs = [] * nsamples
        costs = [] * nsamples

        for s in range(nsamples):
            traj_states = [scenario.sample_initial_dist().reshape(1, -1)]
            traj_outputs = []
            traj_trvs = []
            traj_inputs = []
            traj_costs = []

            for t in range(horizon):
                traj_outputs.append(scenario.sensor(traj_states[t].flatten(), t).detach().reshape((1, -1)))

                if t > 0:
                    traj_trvs.append(self._q_net(traj_outputs[t].flatten(), traj_trvs[t - 1].flatten(), t).reshape((1, -1)))
                else:
                    traj_trvs.append(self._q_net(traj_outputs[t].flatten(), pt.zeros(self._ntrvs), t).reshape((1, -1)))

                traj_inputs.append(self._pi_net(traj_trvs[t].flatten(), t).reshape((1, -1)))

                traj_costs.append(scenario.cost(traj_states[t].flatten(), traj_inputs[t].flatten(), t).reshape((1, -1)))
                traj_states.append(scenario.dynamics(traj_states[t].flatten(), traj_inputs[t].flatten(), t).detach().reshape((1, -1)))

            traj_costs.append(scenario.terminal_cost(traj_states[-1].flatten()).reshape((1, -1)))

            states.append(pt.cat(traj_states, 0).t())
            outputs.append(pt.cat(traj_outputs, 0).t())
            trvs.append(pt.cat(traj_trvs, 0).t())
            inputs.append(pt.cat(traj_inputs, 0).t())
            costs.append(pt.cat(traj_costs, 0).t())

        states = pt.stack(states, axis=2)
        outputs = pt.stack(outputs, axis=2)
        trvs = pt.stack(trvs, axis=2)
        inputs = pt.stack(inputs, axis=2)
        costs = pt.stack(costs, axis=2)

        return states, outputs, trvs, inputs.detach(), costs.detach()


class RMINEPolicy(Policy):
        def __init__(self, scenario: Scenario, horizon: int, ntrvs: int, tradeoff: float, q_layers: List[int], pi_layers: List[int], gpu: bool, nsamples: int, epochs: int, lr: float, mine):
            self._scenario = scenario
            self._horizon = horizon
            self._ntrvs = ntrvs
            self._gpu = gpu
            self._nsamples = nsamples
            self._epochs = epochs
            self._lr = lr
            self._mine_params = mine
            self._tradeoff = tradeoff

            if ntrvs > 0:
                self._q_net = GausNet(scenario.noutputs + ntrvs, ntrvs, q_layers)
                self._mine = [Mine(scenario.nstates + ntrvs, mine['hidden_size']) for t in range(horizon)]
                self._pi_net = GausNet(ntrvs, scenario.ninputs, pi_layers)
            else:
                self._q_net = None
                self._mine = None
                self._pi_net = GausNet(scenario.noutputs, scenario.ninputs, pi_layers)


        def train(self, log: bool):
            horizon = self._horizon
            nsamples = self._nsamples

            if self._ntrvs > 0:
                print('Optimizing both')
                opt = optim.Adam(chain(self._pi_net.parameters(), self._q_net.parameters()), lr=self._lr)
            else:
                opt = optim.Adam(self._pi_net.parameters(), lr=self._lr)

            pi_log_probs = pt.zeros((self._horizon, nsamples))
            q_log_probs = pt.zeros((self._horizon, nsamples))
            mi = pt.zeros(horizon)

            ctg = pt.zeros((self._horizon, self._nsamples))

            for epoch in range(self._epochs):
                opt.zero_grad()
                states, outputs, trvs, inputs, costs = self.rollout(nsamples)

                costs = costs[0, :, :]

                total_cost = costs.sum(axis=0).mean()

                if self._ntrvs > 0 and self._tradeoff > 0:
                    states_cuda = states.detach()
                    trvs_cuda = trvs.detach()
                    print('Computing MINE Estimate')

                    for t in range(horizon):
                        mi[t] = self._mine[t].train((states_cuda[:, t, :], trvs_cuda[:, t, :]), iters=self._mine_params['epochs'], value_filter_window=2)[-1]

                        num_datapts = states.shape[2]
                        batch_size = num_datapts

                        joint_batch_idx = np.random.choice(range(num_datapts), size=num_datapts, replace=False)
                        marginal_batch_idx1 = np.random.choice(range(num_datapts), size=num_datapts, replace=False)
                        marginal_batch_idx2 = np.random.choice(range(num_datapts), size=num_datapts, replace=False)

                        joint_batch = pt.cat((states[:, t, joint_batch_idx], trvs[:, t, joint_batch_idx]), axis=0).t()
                        marginal_batch = pt.cat((states[:, t, marginal_batch_idx1], trvs[:, t, marginal_batch_idx2]), axis=0).t()

                        j_T = self._mine[t](joint_batch)
                        m_T = self._mine[t](marginal_batch)

                        mi[t] = j_T.mean() - pt.log(pt.mean(pt.exp(m_T)))

                for t in range(horizon):
                    for s in range(nsamples):
                        ctg[t, s] = costs[t:, s].sum()

                        if self._ntrvs > 0:
                            pi_log_probs[t, s] = self._pi_net.log_prob(inputs[:, t, s], trvs[:, t, s])

                            if t == 0:
                                prev_trv = pt.zeros(self._ntrvs)
                            else:
                                prev_trv = trvs[:, t - 1, s]

                            q_log_probs[t, s] = self._q_net.log_prob(trvs[:, t, s], pt.cat((outputs[:, t, s], prev_trv)))
                        else:
                            pi_log_probs[t, s] = self._pi_net.log_prob(inputs[:, t, s], outputs[:, t, s])


                mi_sum = mi.sum()

                if log:
                    print('[{0}]\t\tAvg. Cost: {1:.3f}\t\tEst. MI: {2:.3f}\t\tTotal: {3:.3f}'.format(epoch, total_cost, mi.sum().item(), total_cost + self._tradeoff * mi_sum.item()))
                    #print('[{0}]\t\tAvg. Cost: {1:.3f}\t\tEst. MI: {2:.3f}\t\tTotal: {3:.3f}'.format(epoch, total_cost, pt.norm(trvs, dim=0).sum(), total_cost + self._tradeoff * mi_sum.item()))

                baseline = costs.sum(axis=0).mean()
                loss = ((pt.mul(pi_log_probs.sum(axis=0), ctg - baseline)).mean() + (pt.mul(q_log_probs.sum(axis=0), ctg - baseline)).mean()) + self._tradeoff * mi_sum

                loss.backward()

                opt.step()

                pi_log_probs = pi_log_probs.detach()
                mi = mi.detach()
                q_log_probs = q_log_probs.detach()
                ctg = ctg.detach()


        def rollout(self, nsamples: int) -> Tuple[np.ndarray, np.ndarray, float]:
            horizon = self._horizon
            scenario = self._scenario

            states = [] * nsamples
            trvs = [] * nsamples
            inputs = [] * nsamples
            outputs = [] * nsamples
            costs = [] * nsamples

            for s in range(nsamples):
                traj_states = [scenario.sample_initial_dist().reshape(1, -1)]
                traj_outputs = []
                traj_trvs = []
                traj_inputs = []
                traj_costs = []

                for t in range(horizon):
                    traj_outputs.append(scenario.sensor(traj_states[t].flatten(), t).reshape((1, -1)))

                    if self._ntrvs > 0:
                        if t > 0:
                            q_net_input = pt.cat((traj_outputs[t].flatten(), prev_trv.detach()))
                        else:
                            q_net_input = pt.cat((traj_outputs[t].flatten(), pt.zeros(self._ntrvs)))

                        prev_trv = self._q_net(q_net_input)
                        traj_trvs.append(prev_trv.reshape((1, -1)))
                    else:
                        prev_trv = traj_outputs[t]

                    traj_inputs.append(self._pi_net(prev_trv).reshape((1, -1)))

                    traj_costs.append(scenario.cost(traj_states[t].flatten(), traj_inputs[t].flatten(), t).reshape((1, -1)))
                    traj_states.append(scenario.dynamics(traj_states[t].flatten(), traj_inputs[t].flatten(), t).detach().reshape((1, -1)))

                traj_costs.append(scenario.terminal_cost(traj_states[-1].flatten()).reshape((1, -1)))

                states.append(pt.cat(traj_states, 0).t())
                outputs.append(pt.cat(traj_outputs, 0).t())

                if self._ntrvs > 0:
                    trvs.append(pt.cat(traj_trvs, 0).t())

                inputs.append(pt.cat(traj_inputs, 0).t())
                costs.append(pt.cat(traj_costs, 0).t())

            states = pt.stack(states, axis=2)
            outputs = pt.stack(outputs, axis=2)

            if self._ntrvs > 0:
                trvs = pt.stack(trvs, axis=2)
            else:
                trvs = pt.empty((self._ntrvs, horizon, nsamples))

            inputs = pt.stack(inputs, axis=2)
            costs = pt.stack(costs, axis=2)


            return states, outputs, trvs, inputs.detach(), costs.detach()
