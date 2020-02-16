
import torch

from pixyz.losses import Expectation
from pixyz.utils import get_dict_values


class TimeSeriesExpectation(Expectation):

    def __init__(self, p, f, max_iter=None, series_var=None, update_value={}):
        super().__init__(p, f)

        if (max_iter is None) and (series_var is None):
            raise ValueError

        self.max_iter = max_iter
        self.series_var = series_var
        self.update_value = update_value

    @property
    def _symbol(self):
        raise NotImplementedError

    def _get_eval(self, x_dict={}, **kwargs):

        samples_dicts = [
            self._eval_time_expectation(x_dict)
            for i in range(self.sample_shape.numel())]

        loss_and_dicts = [
            self._f.eval(samples_dict, return_dict=True, **kwargs)
            for samples_dict in samples_dicts]

        losses = [loss for loss, loss_sample_dict in loss_and_dicts]

        # sum over sample_shape
        loss = torch.stack(losses).mean(dim=0)
        samples_dicts[0].update(loss_and_dicts[0][1])

        return loss, samples_dicts[0]

    def slice_step_fn(self, t, x):
        return {k: v[t] for k, v in x.items()}

    def _eval_time_expectation(self, x_dict):
        # Extract original values
        series_x_dict = get_dict_values(
            x_dict, self.series_var, return_dict=True)
        updated_x_dict = get_dict_values(
            x_dict, list(self.update_value.values()), return_dict=True)

        # Set max_iter
        if self.max_iter:
            max_iter = self.max_iter
        else:
            max_iter = len(series_x_dict[self.series_var[0]])

        # Sampled values
        sample_dict = {k: [] for k in self.update_value}

        # Time series iteration
        for t in range(max_iter):
            # Update series inputs
            x_dict.update(self.slice_step_fn(t, series_x_dict))

            # 1-time step sample
            samples = self.p.sample(x_dict, reparam=True, return_all=True)

            # Add samples to list
            for k in self.update_value:
                sample_dict[k].append(samples[k][None, :])

            # Update
            for key, value in self.update_value.items():
                x_dict.update({value: samples[key]})

        # Concatenate sampled values
        for key, value in sample_dict.items():
            sample_dict[key] = torch.cat(value)

        # Add sampled values to x_dict
        x_dict.update(sample_dict)

        # Restore original values
        x_dict.update(series_x_dict)
        x_dict.update(updated_x_dict)

        return x_dict
