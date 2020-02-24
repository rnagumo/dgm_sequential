
"""Base class for deep sequential model"""


import tqdm

import torch
import pixyz.models as pxm


class BaseSequentialModel(pxm.Model):
    def __init__(self, device, t_dim, **kwargs):
        super().__init__(**kwargs)

        self.device = device
        self.t_dim = t_dim

    def run(self, loader, epoch, training=True):

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
            data = {"x": x}
            data.update(self._init_variable(minibatch_size, x=x))

            # Mask for sequencial data
            mask = torch.zeros(x.size(0), x.size(1)).to(self.device)
            for i, v in enumerate(seq_len):
                mask[:v, i] += 1

            # Train / test
            if training:
                _loss = super().train(data, mask=mask, epoch=epoch,
                                      results=results)
            else:
                _loss = super().test(data, mask=mask, epoch=epoch)

            # Add training results
            total_loss += _loss * minibatch_size
            total_len += seq_len.sum()

        # Return value
        loss_dict = {"loss": (total_loss / total_len).item()}

        if results:
            results = torch.tensor(results).sum(axis=0)
            loss_dict["annealing_factor"] = results[0] / len(loader)
            loss_dict["cross_entropy"] = results[1] / total_len
            loss_dict["kl_divergence"] = results[2] / total_len

        return loss_dict

    def sample(self):

        # Get initial values
        data = self._init_variable(1)

        x = []
        with torch.no_grad():
            for _ in range(self.t_dim):
                # Sample
                x_t, data = self._sample_one_step(data)

                # Add to data list
                x.append(x_t)

            # Data of size (batch_size, seq_len, input_size)
            x = torch.cat(x).transpose(0, 1)

        # Return data of size (1, batch_size, seq_len, input_size)
        return x[:, None]

    def _init_variable(self, minibatch_size, **kwargs):
        raise NotImplementedError

    def _sample_one_step(self, data, **kwargs):
        raise NotImplementedError
