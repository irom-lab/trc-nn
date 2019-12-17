import torch as pt
import torch.nn as nn
import torch.nn.functional as fn
import torch.optim as optim

import numpy as np

from typing import List, Tuple, Union

from torch.distributions.multivariate_normal import MultivariateNormal

from abc import abstractmethod

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

class FFNet(nn.Module):
    def __init__(self, sizes: List[int]):
        super().__init__()
        self._linears = [nn.Linear(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 2)]
        self._final_layer = nn.Linear(sizes[-2], sizes[-1])
        self._sizes = sizes.copy()

    def forward(self, x):
        for l in self._linears:
            x = fn.elu(l(x))

        return self._final_layer(x)

class QNet(nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, output, prev_trv, t):
        pass

    @abstractmethod
    def log_prob(self, trv, output, prev_trv, t):
        pass

class QNetTV(QNet):
    def __init__(self, module_list: List[nn.Module], horizon: int):
        super().__init__()
        self._module_list = nn.ModuleList([nn.ModuleList(module_list) for t in range(horizon)])

    def forward(self, output, prev_trv, t):
        input = pt.cat([output, prev_trv], 0)
        net = self._module_list[t]

        for module in net:
            input = module(input)

        output = input
        outsize = int(len(output) / 2)

        mean = output[:outsize]
        linear = pt.diag(output[outsize:] + 1e-6)
        sample = pt.distributions.multivariate_normal.MultivariateNormal(pt.zeros(outsize), pt.eye(outsize)).sample()

        return mean + linear.matmul(sample)

    def log_prob(self, trv, output, prev_trv, t):
        input = pt.cat([output, prev_trv], 0)
        net = self._module_list[t]

        for module in net:
            input = module(input)

        output = input
        outsize = int(len(output) / 2)

        mean = output[:outsize]
        linear = pt.diag(output[outsize:] + 1e-6)
        mvn = pt.distributions.multivariate_normal.MultivariateNormal(mean, linear.matmul(linear.t()))

        return mvn.log_prob(trv)

class QNetShared(QNetTV):
    def __init__(self, module_list: List[nn.Module]):
        super().__init__(module_list, 1)

    def forward(self, output, prev_trv, t):
        return super().forward(output, prev_trv, 0)

    def log_prob(self, trv, output, prev_trv, t):
        return super().log_prob(trv, output, prev_trv, 0)


class PiNet(nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, trv, t):
        pass

    @abstractmethod
    def log_prob(self, input, trv, t):
        pass

class PiNetTV(PiNet):
    def __init__(self, module_list: List[nn.Module], horizon: int):
        super().__init__()
        self._module_list = nn.ModuleList([nn.ModuleList(module_list) for t in range(horizon)])

    def forward(self, trv, t):
        net = self._module_list[t]

        for module in net:
            trv = module(trv)

        output = trv

        outsize = int(len(output) / 2)

        mean = output[:outsize]
        linear = pt.diag(output[outsize:] + 1e-6)
        sample = pt.distributions.multivariate_normal.MultivariateNormal(pt.zeros(outsize), pt.eye(outsize)).sample()

        return mean + linear.matmul(sample)

    def log_prob(self, input, trv, t):
        net = self._module_list[t]

        for module in net:
            trv = module(trv)

        output = trv
        outsize = int(len(output) / 2)

        mean = output[:outsize]
        linear = pt.diag(output[outsize:] + 1e-6)
        mvn = pt.distributions.multivariate_normal.MultivariateNormal(mean, linear.matmul(linear.t()))

        return mvn.log_prob(input)

class PiNetShared(PiNetTV):
    def __init__(self, module_list: List[nn.Module], horizon: int):
        super().__init__()
        self._module_list = [nn.ModuleList(module_list)]

    def forward(self, trv, t):
        return super().forward(output, prev_trv, 0)

    def log_prob(self, input, trv, t):
        return super().log_prob(trv, output, prev_trv, 0)

class PiNetOld(FFNet):
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


class PiNet2(FFNet):
    def __init__(self, sizes: List[int], cov):
        self._sizes = sizes.copy()
        self._outsize = sizes[-1]
        self._sizes[-1] = self._sizes[-1] * 2

        super().__init__(self._sizes)

    def forward(self, x):
        output = super().forward(x)

        mean = output[:self._outsize]
        linear = pt.diag(output[self._outsize:])
        sample = pt.distributions.multivariate_normal.MultivariateNormal(pt.zeros(self._outsize), pt.eye(self._outsize)).sample()

        return mean + linear.matmul(sample)

    def log_prob(self, x_tilde, x):
        output = super().forward(x)

        mean = output[:self._outsize]
        linear = pt.diag(output[self._outsize:])

        mvn = pt.distributions.multivariate_normal.MultivariateNormal(mean, linear.matmul(linear.t()))

        return mvn.log_prob(x_tilde)

class GausNet(FFNet):
    def __init__(self, input_size: int, output_size: int, hidden_sizes: List[int]):
        self._input_size = input_size
        self._output_size = output_size
        self._hidden_sizes = hidden_sizes

        super().__init__([input_size] + hidden_sizes + [output_size + output_size])

    def forward(self, input):
        output = super().forward(input)

        affine = output[self._output_size:]
        linear = pt.diag(output[:self._output_size])

        sample = pt.distributions.multivariate_normal.MultivariateNormal(pt.zeros(self._output_size), pt.eye(self._output_size)).sample()

        return affine + pt.matmul(linear, sample)

    def log_prob(self, x, input):
        output = super().forward(input)

        affine = output[self._output_size:]
        linear = pt.diag(output[:self._output_size])

        return pt.distributions.multivariate_normal.MultivariateNormal(affine, pt.matmul(linear, linear.t())).log_prob(x)


class QNetOld(FFNet):
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

class QNet2(FFNet):
    def __init__(self, sizes: List[int], cov, x_tilde_prev: bool=True):
        self._sizes = sizes.copy()
        self._outsize = sizes[-1]
        self._sizes[-1] = self._sizes[-1] * 2

        super().__init__(self._sizes)

        self._x_tilde_prev = x_tilde_prev

    def forward(self, y: pt.tensor, x_tilde_prev: Union[pt.tensor, None]=None):
        if x_tilde_prev is None and self._x_tilde_prev:
            raise RuntimeError('No x_tilde_prev value given despite specifying this network requires one as an input')

        if self._x_tilde_prev:
            input = pt.cat([y, x_tilde_prev], 0)
        else:
            input = y

        output = super().forward(input)

        mean = output[:self._outsize]
        linear = pt.diag(output[self._outsize:])
        sample = pt.distributions.multivariate_normal.MultivariateNormal(pt.zeros(self._outsize), pt.eye(self._outsize)).sample()

        return mean + linear.matmul(sample)

    def log_prob(self, x_tilde: pt.tensor, y: pt.tensor, x_tilde_prev: Union[pt.tensor, None]=None):
        if x_tilde_prev is None and self._x_tilde_prev:
            raise RuntimeError('No x_tilde_prev value given despite specifying this network requires one as an input')

        if self._x_tilde_prev:
            input = pt.cat([y, x_tilde_prev], 0)
        else:
            input = y

        output = super().forward(input)

        mean = output[:self._outsize]
        linear = pt.diag(output[self._outsize:] + 1e-4)

        try:
            mvn = pt.distributions.multivariate_normal.MultivariateNormal(mean, linear.matmul(linear.t()))
        except:
            print(linear)
            print(linear.matmul(linear.t()))

        return mvn.log_prob(x_tilde)

# class QNet2(FFNet):
#     def __init__(self, input_size: int, y_size: int, output_size: int, y_network: FFNet=None, hidden_sizes: List[int]):
#         self._y_network = y_network
#
#         if y_network is not None:
#             self._input_size = input_size + y_network._output_size
#         else:
#             self._input_size = input_size + y_size
#
#         self._output_size = output_size
#         self._hidden_sizes = hidden_sizes
#
#         super().__init__([self._input_size + 1] + hidden_sizes + [output_size + output_size])
#
#     def forward(self, input, t):
#         output = super().forward(pt.cat((input, pt.tensor([t]))))
#
#         affine = output[:self._output_size]
#         linear = pt.diag(output[self._output_size:])
#
#         return affine + pt.matmul(linear, pt.normal(pt.zeros(self._output_size), pt.eye(self._output_size)))
#
#     def log_prob(self, output, input, t):
#         output = super().forward(pt.cat((input, pt.tensor([t]))))
#
#         affine = output[:self._output_size]
#         linear = pt.diag(output[self._output_size:])
#
#         log_prob_mvn(output, affine, pt.matmul(linear, linear.t()))
