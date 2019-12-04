import torch as pt
import torch.nn as nn
import torch.nn.functional as fn
import torch.optim as optim

import numpy as np

from typing import List, Tuple, Union

from torch.distributions.multivariate_normal import MultivariateNormal

def moving_average(series: np.ndarray, window_size: int=100):
    return [np.mean(series[i:(i + window_size)]) for i in range(0, len(series) - window_size)]

def log_prob_mvn(x: pt.Tensor, mean: pt.Tensor, cov: pt.Tensor):
    k = mean.shape[0]
    center = (x - mean).reshape((-1, 1))

    return (pt.exp(-0.5 * pt.mm(center.t(), pt.mm(cov, center))) / pt.sqrt((2 * np.pi) ** k * pt.det(cov))).log()

class Mine(nn.Module):
    def __init__(self, input_size=2, hidden_size=100):
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

        nn.init.normal_(self.fc1.weight,std=0.02)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.normal_(self.fc2.weight,std=0.02)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.normal_(self.fc3.weight,std=0.02)
        nn.init.constant_(self.fc3.bias, 0)

        self._best_value = -np.inf

    def forward(self, input):
        output = fn.elu(self.fc1(input))
        output = fn.elu(self.fc2(output))
        output = self.fc3(output)

        return output

    @property
    def best_value(self):
        return self._best_value

    def train(self, data: pt.Tensor, batch_size: int=None,
              iters: int=int(5e2), log: bool=False, lr: float=1e-3,
              unbiased: bool=True, ma_rate: float=0.01,
              value_filter_window: int=10):

        assert(data[0].shape[1] == data[1].shape[1])

        num_datapts = data[0].shape[1]

        if batch_size is None:
            batch_size = int(num_datapts / 10.0)

        x_data, z_data = data
        opt = optim.Adam(self.parameters(), lr=lr)

        self._best_value = -1

        moving_avg_eT = 1

        values = np.zeros(iters)

        for i in range(iters):
            joint_batch_idx = np.random.choice(range(num_datapts), size=batch_size, replace=False)
            marginal_batch_idx1 = np.random.choice(range(num_datapts), size=batch_size, replace=False)
            marginal_batch_idx2 = np.random.choice(range(num_datapts), size=batch_size, replace=False)

            joint_batch = pt.cat((x_data[:, joint_batch_idx], z_data[:, joint_batch_idx]), axis=0).t()
            marginal_batch = pt.cat((x_data[:, marginal_batch_idx1], z_data[:, marginal_batch_idx2]), axis=0).t()

            j_T = self(joint_batch)
            m_T = self(marginal_batch)

            if unbiased:
                mean_T = j_T.mean()
                eT = pt.mean(pt.exp(m_T))

                moving_avg_eT = ((1 - ma_rate) * moving_avg_eT + ma_rate * eT).detach()

                loss = -(mean_T - (1 / moving_avg_eT) * eT)
                value = float(mean_T - pt.log(eT))
            else:
                loss = -(j_T.mean() - pt.log(pt.mean(pt.exp(m_T))))
                value = float(-loss)

            values[i] = value

            if value > self._best_value:
                self._best_value = value

            if log:
                print('[{0}]\t\t{1}'.format(i, value))

            opt.zero_grad()
            loss.backward()
            opt.step()

        return moving_average(values, value_filter_window)

class Klne(nn.Module):
    def __init__(self, input_size=2, hidden_size=100):
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

        nn.init.normal_(self.fc1.weight,std=0.02)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.normal_(self.fc2.weight,std=0.02)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.normal_(self.fc3.weight,std=0.02)
        nn.init.constant_(self.fc3.bias, 0)

        self._best_value = -np.inf

    def forward(self, input):
        output = fn.elu(self.fc1(input))
        output = fn.elu(self.fc2(output))
        output = self.fc3(output)

        return output

    @property
    def best_value(self):
        return self._best_value

    def train(self, data: pt.Tensor, batch_size: int=None,
              iters: int=int(5e2), log: bool=False, lr: float=1e-3,
              unbiased: bool=True, ma_rate: float=0.01,
              value_filter_window: int=10):

        assert(data[0].shape[1] == data[1].shape[1])

        num_datapts = data[0].shape[1]

        if batch_size is None:
            batch_size = int(num_datapts / 10.0)

        x_data, z_data = data
        opt = optim.Adam(self.parameters(), lr=lr)

        self._best_value = -1

        moving_avg_eT = 1

        values = np.zeros(iters)

        for i in range(iters):
            marginal_batch_idx1 = np.random.choice(range(num_datapts), size=batch_size, replace=False)
            marginal_batch_idx2 = np.random.choice(range(num_datapts), size=batch_size, replace=False)

            p_batch = x_data[:, marginal_batch_idx1].t()
            q_batch = z_data[:, marginal_batch_idx2].t()

            p_T = self(p_batch)
            q_T = self(q_batch)

            if unbiased:
                mean_T = p_T.mean()
                eT = pt.mean(pt.exp(q_T))

                moving_avg_eT = ((1 - ma_rate) * moving_avg_eT + ma_rate * eT).detach()

                loss = -(mean_T - (1 / moving_avg_eT) * eT)
                value = float(mean_T - pt.log(eT))
            else:
                loss = -(p_T.mean() - pt.log(pt.mean(pt.exp(q_T))))
                value = float(-loss)

            values[i] = value

            if value > self._best_value:
                self._best_value = value

            if log:
                print('[{0}]\t\t{1}'.format(i, value))

            opt.zero_grad()
            loss.backward()
            opt.step()

        return moving_average(values, value_filter_window)

class FFNet(nn.Module):
    def __init__(self, sizes: List[int]):
        super().__init__()
        self._linears = [nn.Linear(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 2)]
        self._final_layer = nn.Linear(sizes[-2], sizes[-1])

    def forward(self, x):
        for l in self._linears:
            x = fn.elu(l(x))

        return self._final_layer(x)

class PiNet(FFNet):
    def __init__(self, sizes: List[int], cov: pt.Tensor):
        super().__init__(sizes)

        self._cov = nn.Parameter(cov)
        self._cov.requires_grad = True

        self._mvn = MultivariateNormal(pt.zeros(sizes[-1]), cov)

    def forward(self, x):
        return super().forward(x) + self._mvn.sample()

    def log_prob(self, x_tilde, x):
        mean = super().forward(x)
        return log_prob_mvn(x_tilde, mean, self._cov)

class QNet(FFNet):
    def __init__(self, sizes: List[int], cov: pt.Tensor, x_tilde_prev: bool=True):
        super().__init__(sizes)

        self._cov = cov
        self._mvn = MultivariateNormal(pt.zeros(sizes[-1]), cov)
        self._x_tilde_prev = x_tilde_prev

        self._cov = nn.Parameter(cov)
        self._cov.requires_grad = True

    def forward(self, y: pt.tensor, x_tilde_prev: Union[pt.tensor, None]=None):
        if x_tilde_prev is None and self._x_tilde_prev:
            raise RuntimeError('No x_tilde_prev value given despite specifying this network requires one as an input')

        if self._x_tilde_prev:
            input = pt.cat([y, x_tilde_prev], 0)
        else:
            input = y

        return super().forward(input) + self._mvn.sample()

    def log_prob(self, x_tilde: pt.tensor, y: pt.tensor, x_tilde_prev: Union[pt.tensor, None]=None):
        if x_tilde_prev is None and self._x_tilde_prev:
            raise RuntimeError('No x_tilde_prev value given despite specifying this network requires one as an input')

        if self._x_tilde_prev:
            input = pt.cat([y, x_tilde_prev], 0)
        else:
            input = y

        mean = super().forward(input)
        return log_prob_mvn(x_tilde, mean, self._cov)
