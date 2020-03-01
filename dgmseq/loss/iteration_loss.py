
""""Iterative loss with monitoring CE and KL"""

import pixyz.losses as pxl
import pixyz.utils as pxu


class MonitoredIterativeLoss(pxl.IterativeLoss):
    """Iterative loss with monitoring CE and KL"""

    def __init__(self, cross_entropy, kl_divergence, beta, p=None, **kwargs):

        if p is None:
            _step_loss = cross_entropy + beta * kl_divergence
        else:
            _step_loss = (cross_entropy + beta * kl_divergence).expectation(p)

        super().__init__(_step_loss, **kwargs)

        self.cross_entropy = cross_entropy
        self.kl_divergence = kl_divergence
        self.beta = beta
        self.p = p

    def _get_eval(self, x_dict, **kwargs):
        series_x_dict = pxu.get_dict_values(
            x_dict, self.series_var, return_dict=True)
        updated_x_dict = pxu.get_dict_values(x_dict, list(
            self.update_value.values()), return_dict=True)

        # Set max_iter
        if self.max_iter:
            max_iter = self.max_iter
        else:
            max_iter = len(series_x_dict[self.series_var[0]])

        # Mask for sequence
        if "mask" in kwargs.keys():
            mask = kwargs["mask"].float()
        else:
            mask = None

        _ce_loss_sum = 0
        _kl_loss_sum = 0

        for t in range(max_iter):
            if self.slice_step:
                # Update time step variable, and sample data from slice_step
                x_dict.update({self.timestep_var[0]: t})
                x_dict = self.slice_step.sample(x_dict)
            else:
                # Update series inputs & use slice_step_fn
                x_dict.update(self.slice_step_fn(t, series_x_dict))

            # If p exists, sample variable from p
            if self.p:
                x_dict = self.p.sample(x_dict)

            # Evaluate
            _ce_loss, samples_1 = self.cross_entropy.eval(
                x_dict, return_dict=True)
            _kl_loss, samples_2 = self.kl_divergence.eval(
                x_dict, return_dict=True)

            # Update variable
            x_dict.update(samples_1)
            x_dict.update(samples_2)

            # Set mask
            if mask is not None:
                _ce_loss *= mask[t]
                _kl_loss *= mask[t]

            # Accumulate losses
            _ce_loss_sum += _ce_loss
            _kl_loss_sum += _kl_loss

            # Update
            for key, value in self.update_value.items():
                x_dict.update({value: x_dict[key]})

        # Calculate ELBO loss
        beta = self.beta.eval(x_dict)
        loss = _ce_loss_sum + beta * _kl_loss_sum

        # Save losses to results list
        if "results" in kwargs:
            kwargs["results"].append([_ce_loss_sum.sum().item(),
                                      _kl_loss_sum.sum().item()])

        # Restore original values
        x_dict.update(series_x_dict)
        x_dict.update(updated_x_dict)

        return loss, x_dict
