
""""Iterative loss with KL annealing"""

import pixyz.losses as pxl
import pixyz.utils as pxu


class KLAnnealedIterativeLoss(pxl.IterativeLoss):
    """Iterative loss with KL annealing"""

    def __init__(self, cross_entropy, kl_divergence, annealing_epochs,
                 min_factor, max_factor=1, **kwargs):
        super().__init__(cross_entropy + kl_divergence, **kwargs)

        self.cross_entropy = cross_entropy
        self.kl_divergence = kl_divergence
        self.annealing_epochs = annealing_epochs
        self.min_factor = min_factor
        self.max_factor = max_factor

    @property
    def _symbol(self):
        raise NotImplementedError

    def _get_eval(self, x_dict, **kwargs):
        series_x_dict = pxu.get_dict_values(
            x_dict, self.series_var, return_dict=True)
        updated_x_dict = pxu.get_dict_values(x_dict, list(
            self.update_value.values()), return_dict=True)

        # set max_iter
        if self.max_iter:
            max_iter = self.max_iter
        else:
            max_iter = len(series_x_dict[self.series_var[0]])

        # Mask for sequence
        if "mask" in kwargs.keys():
            mask = kwargs["mask"].float()
        else:
            mask = None

        # KL annealing factor
        if "epoch" in kwargs and kwargs["epoch"] < self.annealing_epochs:
            annealing_factor = (self.min_factor
                                + (self.max_factor - self.min_factor)
                                * kwargs["epoch"] / self.annealing_epochs)
        else:
            annealing_factor = 1.0

        _ce_loss_sum = 0
        _kl_loss_sum = 0

        for t in range(max_iter):
            if self.slice_step:
                x_dict.update({self.timestep_var[0]: t})
            else:
                # update series inputs & use slice_step_fn
                x_dict.update(self.slice_step_fn(t, series_x_dict))

            # evaluate
            _ce_loss, samples_1 = self.cross_entropy.eval(
                x_dict, return_dict=True)
            _kl_loss, samples_2 = self.kl_divergence.eval(
                x_dict, return_dict=True)

            x_dict.update(samples_1)
            x_dict.update(samples_2)

            if mask is not None:
                _ce_loss *= mask[t]
                _kl_loss *= mask[t]

            _ce_loss_sum += _ce_loss
            _kl_loss_sum += _kl_loss

            # update
            for key, value in self.update_value.items():
                x_dict.update({value: x_dict[key]})

        loss = _ce_loss_sum + annealing_factor * _kl_loss_sum

        if "results" in kwargs:
            kwargs["results"].append([
                annealing_factor, _ce_loss_sum.sum().item(),
                _kl_loss_sum.sum().item()])

        # Restore original values
        x_dict.update(series_x_dict)
        x_dict.update(updated_x_dict)

        return loss, x_dict
