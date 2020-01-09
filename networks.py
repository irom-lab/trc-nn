import torch as pt
import torch.nn as nn
import torch.nn.functional as fn
import torch.optim as optim
import numpy as np
import copy

from typing import List, Tuple, Union
from torch.distributions.multivariate_normal import MultivariateNormal
from abc import abstractmethod
from torch.utils.tensorboard import SummaryWriter

from torch.distributions.multivariate_normal import MultivariateNormal

def moving_average(series: np.ndarray, window_size: int=100):
    return [np.mean(series[i:(i + window_size)]) for i in range(0, len(series) - window_size)]

def log_prob_mvn(x: pt.Tensor, mean: pt.Tensor, cov: pt.Tensor):
    k = mean.shape[0]
    center = (x - mean).reshape((-1, 1))

    return (pt.exp(-0.5 * pt.mm(center.t(), pt.mm(cov, center))) / pt.sqrt((2 * np.pi) ** k * pt.det(cov))).log()

def train_mine_network(mine, data: pt.Tensor, batch_size: int=None, epochs: int=int(5e2), log: bool=False, lr: float=1e-3,
                       unbiased: bool=True, ma_rate: float=0.01, tag=None):
          assert(data[0].shape[1] == data[1].shape[1])

          num_datapts = data[0].shape[1]

          if batch_size is None:
              batch_size = int(num_datapts / 10.0)

          x_data, z_data = data
          moving_avg_eT = 1
          values = np.zeros(epochs)
          opt = optim.Adam(mine.parameters(), lr=lr)

          if tag is not None:
              writer = SummaryWriter(tag)

          for epoch in range(epochs):
              joint_batch_idx = np.random.choice(range(num_datapts), size=batch_size, replace=False)
              marginal_batch_idx1 = np.random.choice(range(num_datapts), size=batch_size, replace=False)
              marginal_batch_idx2 = np.random.choice(range(num_datapts), size=batch_size, replace=False)

              joint_batch = pt.cat((x_data[:, joint_batch_idx], z_data[:, joint_batch_idx]), axis=0).t()
              marginal_batch = pt.cat((x_data[:, marginal_batch_idx1], z_data[:, marginal_batch_idx2]), axis=0).t()

              j_T = mine(joint_batch)
              m_T = mine(marginal_batch)

              if unbiased:
                  mean_T = j_T.mean()
                  eT = pt.mean(pt.exp(m_T))

                  moving_avg_eT = ((1 - ma_rate) * moving_avg_eT + ma_rate * eT).detach()

                  loss = -(mean_T - (1 / moving_avg_eT) * eT)
                  value = float(mean_T - pt.log(eT))
              else:
                  loss = -(j_T.mean() - pt.log(pt.mean(pt.exp(m_T))))
                  value = -loss.item()

              values[epoch] = value

              if log:
                  print('[{0}]\t\t{1}'.format(i, value))

              if tag is not None:
                  writer.add_scalar('MINE', value, i)

              opt.zero_grad()
              loss.backward()
              opt.step()

          return values

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
    def __init__(self, make_sequence, horizon: int, make_preprocess_net=None, reshape_to=None):
        super().__init__()
        self._reshape_to = reshape_to

        if make_preprocess_net is not None:
            self._preprocess_net = make_preprocess_net()
        else:
            self._preprocess_net = None

        self._module_list = nn.ModuleList([make_sequence(t) for t in range(horizon)])

    def forward(self, output, prev_trv, t):
        if self._reshape_to is not None:
            preprocessed = self._preprocess_net(output.reshape(self._reshape_to)).flatten()
        else:
            preprocessed = output

        input = pt.cat([preprocessed, prev_trv], 0)
        out = self._module_list[t](input)
        outsize = int(len(out) / 2)

        mean = out[:outsize]
        logcov = out[outsize:]
        linear = pt.diag((pt.exp(logcov) + 1e-3).sqrt())

        ret = mean + linear.matmul(MultivariateNormal(pt.zeros(outsize), pt.eye(outsize)).sample().to(device=mean.device))

        if np.isinf(ret.detach().cpu().numpy()).any():
            print('Yikes')
            print(output)
            print(t)
            print(prev_trv)
            print(out)
            raise RuntimeError()

        return ret

    def log_prob(self, trv, output, prev_trv, t):
        if self._reshape_to is not None:
            preprocessed = self._preprocess_net(output.reshape(self._reshape_to)).flatten()
        else:
            preprocessed = output

        input = pt.cat([preprocessed, prev_trv], 0)
        output = self._module_list[t](input)
        outsize = int(len(output) / 2)

        mean = output[:outsize]
        logcov = output[outsize:]
        cov = pt.diag(pt.exp(logcov) + 1e-3)

        return MultivariateNormal(mean, cov).log_prob(trv)

class QNetShared(QNetTV):
    def __init__(self, make_sequence, make_preprocess_net=None, reshape_to=None):
        super().__init__(make_sequence, 1, make_preprocess_net, reshape_to)

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
    def __init__(self, make_sequence, horizon: int):
        super().__init__()
        self._module_list = nn.ModuleList([make_sequence(t) for t in range(horizon)])

    def forward(self, trv, t):
        output = self._module_list[t](trv)
        outsize = int(len(output) / 2)

        mean = output[:outsize]
        logcov = output[outsize:]
        cov = pt.diag(pt.exp(logcov) + 1e-3)

        try:
            return MultivariateNormal(mean, cov).sample()
        except:
            print(trv)
            print(cov)

    def log_prob(self, input, trv, t):
        output = self._module_list[t](trv)
        outsize = int(len(output) / 2)

        mean = output[:outsize]
        logcov = output[outsize:]
        cov = pt.diag(pt.exp(logcov) + 1e-3)

        return MultivariateNormal(mean, cov).log_prob(input)

class PiNetShared(PiNetTV):
    def __init__(self, make_sequence):
        super().__init__(make_sequence, 1)

    def forward(self, trv, t):
        return super().forward(trv, 0)

    def log_prob(self, input, trv, t):
        return super().log_prob(input, trv, 0)
