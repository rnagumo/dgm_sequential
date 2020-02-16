
"""VRNN

A recurrent latent variable model for sequential data
http://arxiv.org/abs/1506.02216
"""

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F

import pixyz.distributions as pxd
import pixyz.losses as pxl
import pixyz.models as pxm

from .iteration_loss import KLAnnealedIterativeLoss


class Phi_x(nn.Module):
    def __init__(self, x_dim, h_dim):
        super().__init__()
        self.fc0 = nn.Linear(x_dim, h_dim)

    def forward(self, x):
        return F.relu(self.fc0(x))


class Phi_z(nn.Module):
    def __init__(self, z_dim, h_dim):
        super().__init__()
        self.fc0 = nn.Linear(z_dim, h_dim)

    def forward(self, z):
        return F.relu(self.fc0(z))


class Generator(pxd.Bernoulli):
    def __init__(self, h_dim, z_dim, x_dim, f_phi_z):
        super().__init__(cond_var=["z", "h_prev"], var=["x"])

        self.fc1 = nn.Linear(h_dim * 2, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc3 = nn.Linear(h_dim, x_dim)
        self.f_phi_z = f_phi_z

    def forward(self, z, h_prev):
        h = torch.cat([self.f_phi_z(z), h_prev], dim=1)
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        probs = torch.sigmoid(self.fc3(h))
        return {"probs": probs}


class Prior(pxd.Normal):
    def __init__(self, h_dim, z_dim):
        super().__init__(cond_var=["h_prev"], var=["z"])

        self.fc1 = nn.Linear(h_dim, h_dim)
        self.fc21 = nn.Linear(h_dim, z_dim)
        self.fc22 = nn.Linear(h_dim, z_dim)

    def forward(self, h_prev):
        h = F.relu(self.fc1(h_prev))
        loc = self.fc21(h)
        scale = F.softplus(self.fc22(h))
        return {"loc": loc, "scale": scale}


class Recurrence(pxd.Deterministic):
    def __init__(self, h_dim, f_phi_x, f_phi_z):
        super().__init__(cond_var=["x", "z", "h_prev"], var=["h"])

        self.rnn_cell = nn.GRUCell(h_dim * 2, h_dim)
        self.f_phi_x = f_phi_x
        self.f_phi_z = f_phi_z

    def forward(self, x, z, h_prev):
        h_next = self.rnn_cell(
            torch.cat([self.f_phi_z(z), self.f_phi_x(x)], dim=-1), h_prev)
        return {"h": h_next}


class Inference(pxd.Normal):
    def __init__(self, h_dim, z_dim, f_phi_x):
        super().__init__(cond_var=["x", "h_prev"], var=["z"], name="q")

        self.fc1 = nn.Linear(h_dim * 2, h_dim)
        self.fc21 = nn.Linear(h_dim, z_dim)
        self.fc22 = nn.Linear(h_dim, z_dim)
        self.f_phi_x = f_phi_x

    def forward(self, x, h_prev):
        h = torch.cat([self.f_phi_x(x), h_prev], dim=-1)
        h = F.relu(self.fc1(h))
        loc = self.fc21(h)
        scale = F.softplus(self.fc22(h))
        return {"loc": loc, "scale": scale}


def load_vrnn_model(config):

    # Config params
    x_dim = config["x_dim"]
    t_dim = config["t_dim"]
    device = config["device"]

    # Latent dimension
    h_dim = config["vrnn_params"]["h_dim"]
    z_dim = config["vrnn_params"]["z_dim"]

    # Functions
    f_phi_x = Phi_x(x_dim, h_dim).to(device)
    f_phi_z = Phi_z(z_dim, h_dim).to(device)

    # Distributions
    prior = Prior(h_dim, z_dim).to(device)
    decoder = Generator(h_dim, z_dim, x_dim, f_phi_z).to(device)
    encoder = Inference(h_dim, z_dim, f_phi_x).to(device)
    recurrence = Recurrence(h_dim, f_phi_x, f_phi_z).to(device)

    # Loss
    recon = pxl.StochasticReconstructionLoss(
        encoder * recurrence, decoder)
    kl = pxl.KullbackLeibler(encoder, prior)
    _loss = KLAnnealedIterativeLoss(
        recon, kl, max_iter=t_dim, series_var=["x"],
        update_value={"h": "h_prev"}, **config["anneal_params"])
    loss = _loss.mean()

    # Model
    vrnn = pxm.Model(
        loss, distributions=[encoder, decoder, prior, recurrence],
        optimizer=optim.Adam,
        optimizer_params=config["optimizer_params"])

    return vrnn, (prior, recurrence, decoder)


def init_vrnn_var(minibatch_size, config, **kwargs):

    data = {
        "h_prev": torch.zeros(
            minibatch_size, config["vrnn_params"]["h_dim"]
        ).to(config["device"]),
    }

    return data


def get_vrnn_sample(sampler, data):
    prior, recurrence, decoder = sampler

    # Sample x_t
    sample = (prior * decoder * recurrence).sample(data)
    x_t = decoder.sample_mean({"z": sample["z"], "h_prev": sample["h"]})

    # Update h_t
    data["h_prev"] = sample["h"]

    return x_t[None, :], data
