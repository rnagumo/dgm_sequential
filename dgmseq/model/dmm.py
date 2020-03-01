
"""DMM (Deep Markov Model)

Structured Inference Networks for Nonlinear State Space Models
http://arxiv.org/abs/1609.09869
"""

import torch
from torch import nn
from torch.nn import functional as F

import pixyz.distributions as pxd
import pixyz.losses as pxl

from ..loss.iteration_loss import MonitoredIterativeLoss
from .base import BaseSequentialModel


class GatedTrainsitionPrior(pxd.Normal):
    def __init__(self, z_dim, trans_dim):
        super().__init__(cond_var=["z_prev"], var=["z"])

        # Gating unit
        self.fc11 = nn.Linear(z_dim, trans_dim)
        self.fc12 = nn.Linear(trans_dim, z_dim)

        # Proposed mean
        self.fc21 = nn.Linear(z_dim, trans_dim)
        self.fc22 = nn.Linear(trans_dim, z_dim)

        # Trainsition
        self.fc31 = nn.Linear(z_dim, z_dim)
        self.fc32 = nn.Linear(z_dim, z_dim)

        # Initialize as identity function
        self.fc31.weight.data = torch.eye(z_dim)
        self.fc31.bias.data = torch.zeros(z_dim)

    def forward(self, z_prev):
        # Gating unit
        g_t = F.relu(self.fc11(z_prev))
        g_t = torch.sigmoid(self.fc12(g_t))

        # Proposed mean
        h_t = F.relu(self.fc21(z_prev))
        h_t = self.fc22(h_t)

        # Transition mean
        loc = (1 - g_t) * self.fc31(z_prev) + g_t * h_t
        scale = F.softplus(self.fc32(F.relu(h_t)))

        return {"loc": loc, "scale": scale}


class Generator(pxd.Bernoulli):
    def __init__(self, z_dim, hidden_dim, x_dim):
        super().__init__(cond_var=["z"], var=["x"])

        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, x_dim)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        probs = torch.sigmoid(self.fc3(h))
        return {"probs": probs}


class Inference(pxd.Normal):
    def __init__(self, z_dim, h_dim):
        super().__init__(cond_var=["h", "z_prev"], var=["z"])

        self.fc1 = nn.Linear(z_dim, h_dim)
        self.fc21 = nn.Linear(h_dim, z_dim)
        self.fc22 = nn.Linear(h_dim, z_dim)

    def forward(self, h, z_prev):
        h = 0.5 * (h + torch.tanh(self.fc1(z_prev)))
        loc = self.fc21(h)
        scale = F.softplus(self.fc22(h))
        return {"loc": loc, "scale": scale}


class RNN(pxd.Deterministic):
    def __init__(self, x_dim, h_dim):
        super().__init__(cond_var=["x"], var=["h"])

        self.rnn = nn.GRU(x_dim, h_dim, bidirectional=True)
        self.h0 = nn.Parameter(torch.zeros(2, 1, h_dim))

    def forward(self, x):
        h0 = self.h0.expand(2, x.size(1), self.h0.size(2)).contiguous()
        h, _ = self.rnn(x, h0)
        return {"h": h[:, :, h.size(2) // 2:]}


class DMM(BaseSequentialModel):
    def __init__(self, x_dim, h_dim, hidden_dim, z_dim, trans_dim, t_dim,
                 device, anneal_params={}, **kwargs):

        # Dimension
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.trans_dim = trans_dim

        # Distributions
        self.prior = GatedTrainsitionPrior(z_dim, trans_dim).to(device)
        self.decoder = Generator(z_dim, hidden_dim, x_dim).to(device)
        self.encoder = Inference(z_dim, h_dim).to(device)
        self.rnn = RNN(x_dim, h_dim).to(device)
        distributions = [self.prior, self.decoder, self.encoder, self.rnn]

        # Loss
        ce = pxl.CrossEntropy(self.encoder, self.decoder)
        kl = pxl.KullbackLeibler(self.encoder, self.prior)
        beta = pxl.Parameter("beta")
        _loss = MonitoredIterativeLoss(
            ce, kl, beta, max_iter=t_dim, series_var=["x", "h"],
            update_value={"z": "z_prev"})
        loss = _loss.expectation(self.rnn).mean()

        super().__init__(device=device, t_dim=t_dim, series_var=["x", "h"],
                         loss=loss, distributions=distributions,
                         **anneal_params, **kwargs)

    def _init_variable(self, minibatch_size, **kwargs):

        data = {
            "z_prev": torch.zeros(minibatch_size, self.z_dim).to(self.device),
        }

        return data

    def _sample_one_step(self, data, reconstruct=False, **kwargs):

        if reconstruct:
            # Sample latent from encoder
            sample = self.encoder.sample(data)
        else:
            # Sample latent from prior
            sample = self.prior.sample(data)

        # Sample z_t
        x_t = self.decoder.sample_mean({"z": sample["z"]})

        # Update z_t
        z_t = sample["z"]
        data["z_prev"] = z_t

        return x_t[None, :], z_t[None, :], data

    def _inference_batch(self, data, **kwargs):

        sample = self.rnn.sample(data)

        return sample

    def _extract_latest(self, data, **kwargs):

        res_dict = {
            "z_prev": data["z_prev"],
        }

        return res_dict
