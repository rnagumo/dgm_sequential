
"""STORN (Stochastic RNN)

Learning Stochastic Recurrent Networks
http://arxiv.org/abs/1411.7610
"""

import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F

import pixyz.distributions as pxd
import pixyz.losses as pxl

from .iteration_loss import KLAnnealedIterativeLoss
from .base import BaseSequentialModel


class GeneratorRNN(pxd.Deterministic):
    def __init__(self, z_dim, u_dim, h_dim):
        super().__init__(cond_var=["z", "u", "h_prev"], var=["h"])

        self.rnn_cell = nn.RNNCell(z_dim + u_dim, h_dim)

    def forward(self, z, u, h_prev):
        h = self.rnn_cell(torch.cat([z, u], dim=-1), h_prev)
        return {"h": h}


class Generator(pxd.Bernoulli):
    def __init__(self, h_dim, x_dim):
        super().__init__(cond_var=["h"], var=["x"])

        self.fc1 = nn.Linear(h_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, x_dim)

    def forward(self, h):
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        probs = torch.sigmoid(self.fc3(h))
        return {"probs": probs}


class InferenceRNN(pxd.Deterministic):
    def __init__(self, x_dim, z_dim):
        super().__init__(cond_var=["x"], var=["h_v"])

        self.rnn = nn.RNN(x_dim, z_dim * 2)
        self.h0 = nn.Parameter(torch.zeros(1, 1, z_dim * 2))

    def forward(self, x):
        h0 = self.h0.expand(1, x.size(1), self.h0.size(2)).contiguous()
        h, _ = self.rnn(x, h0)
        return {"h_v": h}


class Inference(pxd.Normal):
    def __init__(self):
        super().__init__(cond_var=["h_v"], var=["z"])

    def forward(self, h_v):
        loc = h_v[:, :h_v.size(1) // 2]
        scale = h_v[:, h_v.size(1) // 2:] ** 2
        return {"loc": loc, "scale": scale}


class STORN(BaseSequentialModel):
    def __init__(self, x_dim, h_dim, z_dim, t_dim, device, u_dim=None,
                 anneal_params={}, **kwargs):

        # Input dimension
        if u_dim is None:
            u_dim = x_dim

        self.x_dim = x_dim
        self.u_dim = u_dim

        # Latent dimension
        self.h_dim = h_dim
        self.z_dim = z_dim

        # Distributions
        self.prior = pxd.Normal(
            loc=torch.tensor(0.), scale=torch.tensor(1.), var=["z"],
            features_shape=torch.Size([z_dim])).to(device)
        self.grnn = GeneratorRNN(z_dim, u_dim, h_dim).to(device)
        self.decoder = Generator(h_dim, x_dim).to(device)
        self.irnn = InferenceRNN(x_dim, z_dim).to(device)
        self.encoder = Inference().to(device)
        distributions = [self.prior, self.grnn, self.decoder, self.irnn,
                         self.encoder]

        # Loss
        ce = pxl.CrossEntropy(self.grnn * self.encoder, self.decoder)
        kl = pxl.KullbackLeibler(self.encoder, self.prior)
        _loss = KLAnnealedIterativeLoss(
            ce, kl, max_iter=t_dim, series_var=["x", "u", "h_v"],
            update_value={"h": "h_prev"}, **anneal_params)
        loss = _loss.expectation(self.irnn).mean()

        super().__init__(device=device, t_dim=t_dim, loss=loss,
                         distributions=distributions, **kwargs)

    def _init_variable(self, minibatch_size, x=None, **kwargs):

        if x is not None:
            data = {
                "h_prev": torch.zeros(
                    minibatch_size, self.h_dim).to(self.device),
                "u": torch.cat(
                    [torch.zeros(1, minibatch_size, self.x_dim), x[:-1].cpu()]
                ).to(self.device),
            }
        else:
            data = {
                "h_prev": torch.zeros(
                    minibatch_size, self.h_dim).to(self.device),
                "u": torch.zeros(
                    minibatch_size, self.x_dim).to(self.device),
            }

        return data

    def _sample_one_step(self, data, **kwargs):
        # Sample x_t
        sample = (self.decoder * self.grnn * self.prior).sample(data)
        x_t = self.decoder.sample_mean({"h": sample["h"]})

        # Update h_t
        data["h_prev"] = sample["h"]
        data["u"] = x_t

        return x_t[None, :], data
