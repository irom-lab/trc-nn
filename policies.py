import numpy as np
import torch as pt
import torch.optim as optim
import torch.multiprocessing as multi

from scenarios import Scenario
from networks import PiNet, QNet, Mine

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

        pi_log_probs = pt.zeros((self._horizon, nsamples))

        for titer in range(training_iterations):
            self.rollout(nsamples)

            total_cost = self._costs.sum(axis=0).mean()

            if log:
                print('[{0}]\t\tAvg. Cost: {1}'.format(titer, total_cost))

            for t in range(horizon):
                for s in range(nsamples):
                    pi_log_probs[t, s] = self._pi_net[t].log_prob(self._inputs[:, t, s], self._outputs[:, t, s])

            baseline = self._costs.sum(axis=0).mean()

            pi_opt.zero_grad()
            loss = (pt.mul(pi_log_probs.sum(axis=0), self._costs.sum(axis=0) - baseline)).mean()
            loss.backward()
            pi_opt.step()

            pi_log_probs = pi_log_probs.detach()

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
    def __init__(self, scenario: Scenario, horizon: int, ntrvs: int,
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
        pi_opt = optim.Adam(chain(*[pi.parameters() for pi in self._pi_net]), lr=pilr)
        q_opt = optim.Adam(chain(*[q.parameters() for q in self._q_net]), lr=qlr)
        horizon = self._horizon

        pi_log_probs = pt.zeros((self._horizon, nsamples))
        q_log_probs = pt.zeros((self._horizon, nsamples))
        mi = pt.zeros(horizon)

        for titer in range(training_iterations):
            res = self.rollout(nsamples)

            states = pt.from_numpy(res[0]).float()
            outputs = pt.from_numpy(res[1]).float()
            trvs = pt.from_numpy(res[2]).float()
            inputs = pt.from_numpy(res[3]).float()
            costs = pt.from_numpy(res[4]).float()

            total_cost = costs.sum(axis=0).sum() / nsamples

            if self._tradeoff > 0:
                states_cuda = states.cuda()
                trvs_cuda = trvs.cuda()

                for t in range(horizon):
                    self._mine[t].cuda()
                    mi[t] = self._mine[t].train((states_cuda[:, t, :], trvs_cuda[:, t, :]))[-1]
                    self._mine[t].cpu()

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
                print('[{0}]\t\tAvg. Cost: {1:.3f}\t\tEst. MI: {2:.3f}\t\tTotal: {3:.3f}'.format(titer, total_cost, self._tradeoff * float(mi.sum().item()), total_cost + float(mi.sum().item())))

            pi_opt.zero_grad()
            q_opt.zero_grad()

            baseline = costs.sum(axis=0).mean()

            pi_loss = (pt.mul(pi_log_probs.sum(axis=0), costs.sum(axis=0) - baseline)).sum() / nsamples
            q_loss = self._tradeoff * (pt.mul(q_log_probs.sum(axis=0), costs.sum(axis=0) - baseline)).sum() / nsamples +  mi.sum()

            pi_loss.backward()
            q_loss.backward()

            pi_opt.step()
            q_opt.step()

            pi_log_probs = pi_log_probs.detach()
            q_log_probs = q_log_probs.detach()
            mi = mi.detach()

    def rollout(self, nsamples):
        horizon = self._horizon
        scenario = self._scenario

        with pt.no_grad():
            states = np.zeros((scenario.nstates, horizon + 1, nsamples))
            trvs = np.zeros((self._ntrvs, horizon, nsamples))
            inputs = np.zeros((scenario.ninputs, horizon, nsamples))
            outputs = np.zeros((scenario.noutputs, horizon, nsamples))
            costs = np.zeros((horizon + 1, nsamples))

            for s in range(nsamples):
                t = 0
                states[:, 0, s] = scenario.sample_initial_dist()

                for t in range(horizon):
                    outputs[:, t, s] = scenario.sensor(states[:, t, s], t)

                    if self._q_net[t]._x_tilde_prev:
                        trvs[:, t, s] = self._q_net[t](pt.from_numpy(outputs[:, t, s]).float(), pt.from_numpy(trvs[:, t - 1, s]).float()).detach().numpy().astype('double')
                    else:
                        trvs[:, t, s] = self._q_net[t](pt.from_numpy(outputs[:, t, s]).float()).detach().numpy().astype('double')

                    inputs[:, t, s] = self._pi_net[t](pt.from_numpy(trvs[:, t, s]).float()).detach().numpy().astype('double')

                    costs[t, s] = scenario.cost(states[:, t, s], inputs[:, t, s], t)

                    states[:, t + 1, s] = scenario.dynamics(states[:, t, s], inputs[:, t, s], t)

                costs[-1, s] += scenario.terminal_cost(states[:, -1, s])

            return states, outputs, trvs, inputs, costs
