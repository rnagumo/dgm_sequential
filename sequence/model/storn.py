
"""STORN (Stochastic RNN)

Learning Stochastic Recurrent Networks
http://arxiv.org/abs/1411.7610
"""

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F

import pixyz.distributions as pxd
import pixyz.losses as pxl
import pixyz.models as pxm

from .iteration_loss import KLAnnealedIterativeLoss


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


def load_storn_model(config):

    # Config params
    x_dim = config["x_dim"]
    t_dim = config["t_dim"]
    device = config["device"]
    u_dim = x_dim

    # Latent dimension
    h_dim = config["storn_params"]["h_dim"]
    z_dim = config["storn_params"]["z_dim"]

    # Distributions
    prior = pxd.Normal(
        loc=torch.tensor(0.), scale=torch.tensor(1.), var=["z"],
        features_shape=torch.Size([z_dim])).to(device)
    grnn = GeneratorRNN(z_dim, u_dim, h_dim).to(device)
    decoder = Generator(h_dim, x_dim).to(device)
    irnn = InferenceRNN(x_dim, z_dim).to(device)
    encoder = Inference().to(device)

    # Loss
    ce = pxl.CrossEntropy(grnn * encoder, decoder)
    kl = pxl.KullbackLeibler(encoder, prior)
    _loss = KLAnnealedIterativeLoss(
        ce, kl, max_iter=t_dim, series_var=["x", "u", "h_v"],
        update_value={"h": "h_prev"}, **config["anneal_params"])
    loss = _loss.expectation(irnn).mean()

    # Model
    storn = pxm.Model(
        loss, distributions=[prior, grnn, decoder, irnn, encoder],
        optimizer=optim.Adam,
        optimizer_params=config["optimizer_params"])

    return storn, (decoder, grnn, prior)


def init_storn_var(minibatch_size, config, x=None, **kwargs):

    if x is not None:
        data = {
            "h_prev": torch.zeros(
                minibatch_size, config["storn_params"]["h_dim"]
            ).to(config["device"]),
            "u": torch.cat(
                [torch.zeros(1, minibatch_size, config["x_dim"]), x[:-1].cpu()]
            ).to(config["device"]),
        }
    else:
        data = {
            "h_prev": torch.zeros(
                minibatch_size, config["storn_params"]["h_dim"]
            ).to(config["device"]),
            "u": torch.zeros(
                minibatch_size, config["x_dim"]).to(config["device"]),
        }

    return data


def get_storn_sample(sampler, data):

    # TODO: wrong method
    decoder, grnn, prior = sampler

    # Sample x_t
    sample = (decoder * grnn * prior).sample(data)
    x_t = decoder.sample_mean({"h": sample["h"]})

    # Update h_t
    data["h_prev"] = sample["h"]
    data["u"] = x_t

    return x_t[None, :], data
