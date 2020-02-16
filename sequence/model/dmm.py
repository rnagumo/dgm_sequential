
"""Deep Markov Model

"""

import torch
from torch import nn, optim
from torch.nn import functional as F

import pixyz.distributions as pxd
import pixyz.losses as pxl
import pixyz.models as pxm

from .iteration_loss import KLAnnealedIterativeLoss


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


def load_dmm_model(config):

    # Config params
    x_dim = config["x_dim"]
    t_dim = config["t_dim"]
    device = config["device"]

    # Latent dimension
    h_dim = config["dmm_params"]["h_dim"]
    hidden_dim = config["dmm_params"]["hidden_dim"]
    z_dim = config["dmm_params"]["z_dim"]
    trans_dim = config["dmm_params"]["trans_dim"]

    # Distributions
    prior = GatedTrainsitionPrior(z_dim, trans_dim).to(device)
    decoder = Generator(z_dim, hidden_dim, x_dim).to(device)
    encoder = Inference(z_dim, h_dim).to(device)
    rnn = RNN(x_dim, h_dim).to(device)

    # Loss
    ce = pxl.CrossEntropy(encoder, decoder)
    kl = pxl.KullbackLeibler(encoder, prior)
    _loss = KLAnnealedIterativeLoss(
        ce, kl, max_iter=t_dim, series_var=["x", "h"],
        update_value={"z": "z_prev"}, **config["anneal_params"])
    loss = _loss.expectation(rnn).mean()

    # Model
    dmm = pxm.Model(
        loss, distributions=[rnn, encoder, decoder, prior],
        optimizer=optim.Adam,
        optimizer_params=config["optimizer_params"])

    return dmm, (prior, decoder)


def init_dmm_var(minibatch_size, config, **kwargs):

    data = {
        "z_prev": torch.zeros(
            minibatch_size, config["dmm_params"]["z_dim"]
        ).to(config["device"]),
    }

    return data


def get_dmm_sample(sampler, data):
    prior, decoder = sampler

    # Sample x_t
    sample = (prior * decoder).sample(data)
    x_t = decoder.sample_mean({"z": sample["z"]})

    # Update z_t
    data["z_prev"] = sample["z"]

    return x_t[None, :], data
