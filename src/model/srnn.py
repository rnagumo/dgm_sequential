
"""Stochastic RNN

Sequential Neural Models with Stochastic Layers
http://arxiv.org/abs/1605.07571
"""

import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F

import pixyz.distributions as pxd
import pixyz.losses as pxl

from .iteration_loss import MonitoredIterativeLoss
from .time_expectation import TimeSeriesExpectation
from .base import BaseSequentialModel


class ForwardRNN(pxd.Deterministic):
    def __init__(self, u_dim, d_dim):
        super().__init__(cond_var=["u", "d_prev"], var=["d"])

        self.rnn_cell = nn.GRUCell(u_dim, d_dim)
        self.d0 = nn.Parameter(torch.zeros(1, 1, d_dim))

    def forward(self, u, d_prev):
        d = self.rnn_cell(u, d_prev)
        return {"d": d}


class Prior(pxd.Normal):
    def __init__(self, d_dim, z_dim):
        super().__init__(cond_var=["z_prev", "d"], var=["z"])

        self.fc1 = nn.Linear(d_dim + z_dim, 512)
        self.fc21 = nn.Linear(512, z_dim)
        self.fc22 = nn.Linear(512, z_dim)

    def forward(self, z_prev, d):
        h = F.relu(self.fc1(torch.cat([z_prev, d], dim=-1)))
        scale = self.fc21(h)
        loc = F.softplus(self.fc22(h))
        return {"scale": scale, "loc": loc}


class Generator(pxd.Bernoulli):
    def __init__(self, z_dim, d_dim, x_dim):
        super().__init__(cond_var=["z", "d"], var=["x"])

        self.fc1 = nn.Linear(z_dim + d_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, x_dim)

    def forward(self, z, d):
        h = F.relu(self.fc1(torch.cat([z, d], dim=-1)))
        h = F.relu(self.fc2(h))
        probs = torch.sigmoid(self.fc3(h))
        return {"probs": probs}


class BackwardRNN(pxd.Deterministic):
    def __init__(self, x_dim, d_dim, a_dim):
        super().__init__(cond_var=["x", "d"], var=["a"])

        self.rnn = nn.GRU(x_dim + d_dim, a_dim, bidirectional=True)
        self.a0 = nn.Parameter(torch.zeros(2, 1, a_dim))

    def forward(self, x, d):
        a0 = self.a0.expand(2, x.size(1), self.a0.size(2)).contiguous()
        a, _ = self.rnn(torch.cat([x, d], dim=-1), a0)
        return {"a": a[:, :, a.size(2) // 2:]}


class VariationalPrior(pxd.Normal):
    def __init__(self, z_dim, a_dim):
        super().__init__(cond_var=["z_prev", "a"], var=["z"])

        self.fc1 = nn.Linear(z_dim + a_dim, 512)
        self.fc21 = nn.Linear(512, z_dim)
        self.fc22 = nn.Linear(512, z_dim)

    def forward(self, z_prev, a):
        h = F.relu(self.fc1(torch.cat([z_prev, a], dim=-1)))
        scale = self.fc21(h)
        loc = F.softplus(self.fc22(h))
        return {"scale": scale, "loc": loc}


class SRNN(BaseSequentialModel):
    def __init__(self, x_dim, d_dim, z_dim, a_dim, t_dim, device, u_dim=None,
                 anneal_params={}, **kwargs):

        # Input dimension
        if u_dim is None:
            u_dim = x_dim

        self.x_dim = x_dim
        self.u_dim = u_dim

        # Latent dimension
        self.d_dim = d_dim
        self.z_dim = z_dim
        self.a_dim = a_dim

        # Distributions
        self.frnn = ForwardRNN(u_dim, d_dim).to(device)
        self.prior = Prior(d_dim, z_dim).to(device)
        self.decoder = Generator(z_dim, d_dim, x_dim).to(device)
        self.brnn = BackwardRNN(x_dim, d_dim, a_dim).to(device)
        self.encoder = VariationalPrior(z_dim, a_dim).to(device)
        distributions = [self.prior, self.frnn, self.decoder, self.brnn,
                         self.encoder]

        # Loss
        ce = pxl.CrossEntropy(self.encoder, self.decoder)
        kl = pxl.KullbackLeibler(self.encoder, self.prior)
        beta = pxl.Parameter("beta")
        _loss = MonitoredIterativeLoss(
            ce, kl, beta, max_iter=t_dim, series_var=["x", "d", "a"],
            update_value={"z": "z_prev"})

        # Calculate Backward latent a_{1:T} = brnn(d_{1:T})
        _loss_batch_obs = _loss.expectation(self.brnn)

        # Calculate Forward latent d_{1:T} = frnn(x_{1:T})
        _loss_batch = TimeSeriesExpectation(
            self.frnn, _loss_batch_obs, series_var=["u"],
            update_value={"d": "d_prev"})

        # Mean for batch
        loss = _loss_batch.mean()

        super().__init__(device=device, t_dim=t_dim, loss=loss,
                         distributions=distributions, **anneal_params,
                         **kwargs)

    def _init_variable(self, minibatch_size, **kwargs):

        if "x" in kwargs:
            data = {
                "z_prev": torch.zeros(
                    minibatch_size, self.z_dim).to(self.device),
                "d_prev": torch.zeros(
                    minibatch_size, self.d_dim).to(self.device),
                "u": torch.cat(
                    [torch.zeros(1, minibatch_size, self.x_dim),
                     kwargs["x"][:-1].cpu()]).to(self.device),
            }
        else:
            data = {
                "z_prev": torch.zeros(
                    minibatch_size, self.z_dim).to(self.device),
                "d_prev": torch.zeros(
                    minibatch_size, self.d_dim).to(self.device),
                "u": torch.zeros(
                    minibatch_size, self.x_dim).to(self.device),
            }

        return data

    def _sample_one_step(self, data, **kwargs):
        # Sample x_t
        sample = (self.prior * self.frnn * self.decoder).sample(data)
        x_t = self.decoder.sample_mean({"z": sample["z"], "d": sample["d"]})

        # Update
        data["z_prev"] = sample["z"]
        data["d_prev"] = sample["d"]
        data["u"] = x_t

        return x_t[None, :], data
