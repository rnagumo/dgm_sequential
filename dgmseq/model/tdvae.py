
"""Temporal Difference VAE (TD-VAE)

Temporal Difference Variational Auto-Encoder
http://arxiv.org/abs/1806.03107

Pixyz implementation
https://github.com/masa-su/pixyzoo/blob/master/TD-VAE/distribution.py
"""

import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F

import pixyz.distributions as pxd
import pixyz.losses as pxl

from ..loss.iteration_loss import MonitoredIterativeLoss
from .base import BaseSequentialModel


class DBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, output_dim)
        self.fc_logsigma = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h = torch.tanh(self.fc1(x)) * torch.sigmoid(self.fc2(x))
        mu = self.fc_mu(h)
        logsigma = self.fc_logsigma(h)
        return mu, logsigma


class Flitering(pxd.Normal):
    def __init__(self, b_dim, h_dim, z_dim):
        super().__init__(cond_var=["b_t1"], var=["z_t1"], name="p_b")

        self.dblock = DBlock(b_dim, h_dim, z_dim)

    def forward(self, b_t1):
        mu, logsigma = self.dblock(b_t1)
        std = torch.exp(logsigma)
        return {"loc": mu, "scale": std}


class Transition(pxd.Normal):
    def __init__(self, z_dim, h_dim):
        super().__init__(cond_var=["z_t1"], var=["z_t2"], name="p_t")

        self.dblock = DBlock(z_dim, h_dim, z_dim)

    def forward(self, z_t1):
        mu, logsigma = self.dblock(z_t1)
        std = torch.exp(logsigma)
        return {"loc": mu, "scale": std}


class Generator(pxd.Bernoulli):
    def __init__(self, z_dim, h_dim, x_dim):
        super().__init__(cond_var=["z_t2"], var=["x_t2"], name="p_g")

        self.fc1 = nn.Linear(z_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc3 = nn.Linear(h_dim, x_dim)

    def forward(self, z_t2):
        h = F.relu(self.fc1(z_t2))
        h = F.relu(self.fc2(h))
        probs = torch.sigmoid(self.fc3(h))
        return {"probs": probs}


class Inference(pxd.Normal):
    def __init__(self, z_dim, b_dim, h_dim):
        super().__init__(cond_var=["z_t2", "b_t1", "b_t2"], var=["z_t1"],
                         name="q")

        self.dblock = DBlock(z_dim + b_dim * 2, h_dim, z_dim)

    def forward(self, z_t2, b_t1, b_t2):
        mu, logsigma = self.dblock(torch.cat((z_t2, b_t1, b_t2), dim=1))
        std = torch.exp(logsigma)
        return {"loc": mu, "scale": std}


class BeliefStateNet(pxd.Deterministic):
    def __init__(self, x_dim, rnn_dim, b_dim):
        super().__init__(cond_var=["x"], var=["b"])

        self.fc1 = nn.Linear(x_dim, rnn_dim)
        self.fc2 = nn.Linear(rnn_dim, rnn_dim)
        self.rnn = nn.LSTM(rnn_dim, b_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        b, _ = self.rnn(h)
        return {"b": b}


class SliceStep(pxd.Deterministic):
    def __init__(self):
        super().__init__(cond_var=["t", "x", "b"],
                         var=["x_t2", "b_t1", "b_t2"], name="f")

    def forward(self, t, x, b):
        slice_dict = {"x_t2": x[t], "b_t1": b[t], "b_t2": b[t + 1]}
        return slice_dict


class TDVAE(BaseSequentialModel):
    def __init__(self, x_dim, z_dim, h_dim, b_dim, rnn_dim, t_dim, device,
                 anneal_params={}, **kwargs):

        # Dimension
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.b_dim = b_dim

        # Distributions
        self.p_b1 = Flitering(b_dim, h_dim, z_dim).to(device)
        self.p_b2 = self.p_b1.replace_var(b_t1="b_t2", z_t1="z_t2")
        self.p_t = Transition(z_dim, h_dim).to(device)
        self.p_g = Generator(z_dim, h_dim, x_dim).to(device)
        self.q = Inference(z_dim, b_dim, h_dim).to(device)
        self.belief_net = BeliefStateNet(x_dim, rnn_dim, b_dim).to(device)
        self.slice_step = SliceStep()
        distributions = [self.p_b1, self.p_b2, self.p_t, self.p_g, self.q,
                         self.belief_net]

        # Loss
        ce = pxl.Expectation(
            self.q,
            -self.p_g.log_prob() - self.p_t.log_prob() + self.p_b2.log_prob())
        kl = pxl.KullbackLeibler(self.q, self.p_b2)
        beta = pxl.Parameter("beta")
        _loss = MonitoredIterativeLoss(
            ce, kl, beta, p=self.p_b2, max_iter=t_dim - 1,
            series_var=["x", "b"], slice_step=self.slice_step)
        loss = _loss.expectation(self.belief_net).mean()

        super().__init__(device=device, t_dim=t_dim - 1, loss=loss,
                         distributions=distributions, series_var=["x", "b"],
                         **anneal_params, **kwargs)

    def _init_variable(self, minibatch_size, **kwargs):

        if "x" in kwargs:
            data = {}
        else:
            data = {
                "z_t1": torch.zeros(
                    minibatch_size, self.z_dim).to(self.device),
            }

        return data

    def _sample_one_step(self, data, reconstruct=False, **kwargs):

        if reconstruct:
            # Sample latent from encoder
            sample = (self.p_t * self.p_b1 * self.slice_step).sample(data)
        else:
            # Sample latent from prior
            sample = self.p_t.sample(data)

        # Sample x_t
        x_t = self.p_g.sample_mean({"z_t2": sample["z_t2"]})

        # Update z_t
        z_t = sample["z_t2"]
        data["z_t1"] = z_t

        return x_t[None, :], z_t[None, :], data

    def _inference_batch(self, data, **kwargs):

        return self.belief_net.sample(data)

    def _extract_latest(self, data, **kwargs):

        res_dict = {
            "z_t1": data["z_t1"]
        }

        return res_dict
