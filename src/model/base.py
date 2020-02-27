
"""Base class for deep sequential model"""


import tqdm

import torch
import pixyz.models as pxm
import pixyz.utils as pxu


class BaseSequentialModel(pxm.Model):
    def __init__(self, device, t_dim, series_var, anneal_epochs, min_factor,
                 max_factor, **kwargs):
        super().__init__(**kwargs)

        self.device = device
        self.t_dim = t_dim
        self.series_var = series_var

        # KL annealing parameters
        self.anneal_epochs = anneal_epochs
        self.min_factor = min_factor
        self.max_factor = max_factor

    def run(self, loader, epoch, training=True):

        # Calculate KL annealing factor
        if epoch < self.anneal_epochs:
            beta = (self.min_factor + (self.max_factor - self.min_factor)
                    * epoch / self.anneal_epochs)
        else:
            beta = self.max_factor

        # Returned values
        total_loss = 0
        total_len = 0
        results = []

        # Train with mini-batch
        for x, seq_len in tqdm.tqdm(loader):
            # Input dimension must be (timestep_size, batch_size, feature_size)
            x = x.transpose(0, 1).to(self.device)
            minibatch_size = x.size(1)

            # Prepare data
            data = {"x": x, "beta": beta}
            data.update(self._init_variable(minibatch_size, x=x))

            # Mask for sequencial data
            mask = torch.zeros(x.size(0), x.size(1)).to(self.device)
            for i, v in enumerate(seq_len):
                mask[:v, i] += 1

            # Train / test
            if training:
                _loss = self.train(data, mask=mask, results=results)
            else:
                _loss = self.test(data, mask=mask)

            # Add training results
            total_loss += _loss * minibatch_size
            total_len += seq_len.sum()

        # Return value
        loss_dict = {"loss": (total_loss / total_len).item(), "beta": beta}

        # Add monitored values
        if results:
            results = torch.tensor(results).sum(axis=0)
            loss_dict["cross_entropy"] = results[0] / total_len
            loss_dict["kl_divergence"] = results[1] / total_len

        return loss_dict

    def sample(self):
        """Sample data from initial latent variable."""

        # Get initial values
        data = self._init_variable(1)

        x = []
        z = []
        with torch.no_grad():
            for _ in range(self.t_dim):
                # Sample
                x_t, z_t, data = self._sample_one_step(data)

                # Add to data list
                x.append(x_t)
                z.append(z_t)

        # Data of size (batch_size, seq_len, input_size)
        x = torch.cat(x).transpose(0, 1).cpu()
        z = torch.cat(z).transpose(0, 1).cpu()

        # Return data of size (batch_size, channel=1, seq_len, input_size)
        return x[:, None], z[:, None]

    def reconstruct(self, x, time_step=0):
        """Inference latent variable, and reconstruct observable."""

        # Input dimension must be (timestep_size, batch_size, feature_size)
        x = x.transpose(0, 1).to(self.device)
        minibatch_size = x.size(1)

        # Prepare data
        data = {"x": x}
        data.update(self._init_variable(minibatch_size, x=x))

        # Batch inference
        data = self._inference_batch(data)

        # Extract time series values
        series_dict = pxu.get_dict_values(data, self.series_var,
                                          return_dict=True)

        # Reconstruct
        x_recon = []
        z = []
        with torch.no_grad():
            for t in range(self.t_dim):
                # Update time step
                if hasattr(self, "slice_step"):
                    # Update t value
                    data.update({"t": t})
                else:
                    # Update time series variable manually
                    data.update({k: v[t] for k, v in series_dict.items()})

                # Sample observable
                x_t, z_t, data = self._reconstruct_one_step(data)

                # Add sampled values to data list
                x_recon.append(x_t)
                z.append(z_t)

        # Predict future
        data = self._extract_latest(data)
        with torch.no_grad():
            for _ in range(time_step):
                # Sample
                x_t, z_t, data = self._sample_one_step(data)

                # Add to data list
                x_recon.append(x_t)
                z.append(z_t)

        # Data of size (batch_size, seq_len, input_size)
        x_recon = torch.cat(x_recon).transpose(0, 1).cpu()
        z = torch.cat(z).transpose(0, 1).cpu()

        # Return data of size (batch_size, channel=1, seq_len, input_size)
        return x_recon[:, None], z[:, None]

    def _init_variable(self, minibatch_size, **kwargs):
        raise NotImplementedError

    def _sample_one_step(self, data, **kwargs):
        raise NotImplementedError

    def _inference_batch(self, data, **kwargs):
        raise NotImplementedError

    def _reconstruct_one_step(self, data, **kwargs):
        raise NotImplementedError

    def _extract_latest(self, data, **kwargs):
        raise NotImplementedError
