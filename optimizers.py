import numpy
from mxnet.ndarray import (zeros, clip, sqrt, square)
from mxnet.ndarray import (sgd_update, sgd_mom_update,
                       mp_sgd_update, mp_sgd_mom_update,
                       multi_sgd_update, multi_sgd_mom_update,
                       multi_mp_sgd_update, multi_mp_sgd_mom_update, adam_update)
from mxnet.optimizer import Optimizer, register

import math
__all__ = ['SGD']


def _flatten_list(nested_list):
    return [item for sublist in nested_list for item in sublist]

@register
class SGDGCC(Optimizer):
    """The SGD optimizer with momentum and weight decay.

    If the storage types of grad is ``row_sparse`` and ``lazy_update`` is True, \
    **lazy updates** are applied by::

        for row in grad.indices:
            rescaled_grad[row] = clip(rescale_grad * grad[row] + wd * weight[row], clip_gradient)
            state[row] = momentum[row] * state[row] + lr * rescaled_grad[row]
            weight[row] = weight[row] - state[row]

    The sparse update only updates the momentum for the weights whose row_sparse
    gradient indices appear in the current batch, rather than updating it for all
    indices. Compared with the original update, it can provide large
    improvements in model training throughput for some applications. However, it
    provides slightly different semantics than the original update, and
    may lead to different empirical results.

    In the case when ``update_on_kvstore`` is set to False (either globally via
    MXNET_UPDATE_ON_KVSTORE=0 environment variable or as a parameter in
    :class:`~mxnet.gluon.Trainer`) SGD optimizer can perform aggregated update
    of parameters, which may lead to improved performance. The aggregation size
    is controlled by ``aggregate_num`` and defaults to 4.

    Otherwise, **standard updates** are applied by::

        rescaled_grad = clip(rescale_grad * grad, clip_gradient)) + wd * weight
        state = momentum * state + lr * rescaled_grad
        weight = weight - state

    For details of the update algorithm see
    :class:`~mxnet.ndarray.sgd_update` and :class:`~mxnet.ndarray.sgd_mom_update`.

    This optimizer accepts the following parameters in addition to those accepted
    by :class:`.Optimizer`.

    Parameters
    ----------
    learning_rate : float, default 0.1
        The initial learning rate. If None, the optimization will use the
        learning rate from ``lr_scheduler``. If not None, it will overwrite
        the learning rate in ``lr_scheduler``. If None and ``lr_scheduler``
        is also None, then it will be set to 0.01 by default.
    momentum : float, default 0.
        The momentum value.
    lazy_update : bool, default False
        Default is False. If True, lazy updates are applied \
        if the storage types of weight and grad are both ``row_sparse``.
    multi_precision: bool, default False
        Flag to control the internal precision of the optimizer.
        False: results in using the same precision as the weights (default),
        True: makes internal 32-bit copy of the weights and applies gradients
        in 32-bit precision even if actual weights used in the model have lower precision.
        Turning this on can improve convergence and accuracy when training with float16.
    aggregate_num : int, default 1
        Number of weights to be aggregated in a list.
        They are passed to the optimizer for a single optimization step.
    use_fused_step : bool, default True
        Whether or not to use fused kernels for optimizer.
        When use_fused_step=False, step is called,
        otherwise, fused_step is called.
    """
    def __init__(self, learning_rate=0.1, momentum=0.0, lazy_update=False,
                 multi_precision=False, use_fused_step=True, aggregate_num=1, **kwargs):
        super(SGDGCC, self).__init__(learning_rate=learning_rate,
                                  multi_precision=multi_precision,
                                  aggregate_num=aggregate_num,
                                  use_fused_step=use_fused_step,
                                  **kwargs)
        if not self.use_fused_step:
            assert not lazy_update, \
                'When use_fused_step is set to False, lazy_update has to be turned off.'
        if lazy_update:
            assert not multi_precision, \
                'When lazy_update is set to True, multi_precision has be turned off.'
        self.momentum = momentum
        self.lazy_update = lazy_update

    def create_state(self, index, weight):
        momentum = None
        if self.momentum != 0.0:
            stype = weight.stype if self.lazy_update else 'default'
            momentum = zeros(weight.shape, weight.context, dtype=weight.dtype, stype=stype)
        return momentum

    def step(self, indices, weights, grads, states):
        """Perform an optimization step using gradients and states.

        Parameters
        ----------
        indices : list of int
            List of unique indices of the parameters into the individual learning rates
            and weight decays. Learning rates and weight decay may be set via `set_lr_mult()`
            and `set_wd_mult()`, respectively.
        weights : list of NDArray
            List of parameters to be updated.
        grads : list of NDArray
            List of gradients of the objective with respect to this parameter.
        states : List of any obj
            List of state returned by `create_state()`.
        """
        for index, weight, grad, state in zip(indices, weights, grads, states):
            self._update_count(index)
            lr = self._get_lr(index)
            wd = self._get_wd(index)
            if len(list(grad.shape_array()))>3:
                   grad -= grad.mean(axis=0, exclude=True, keepdims=True)
            # preprocess grad
            grad *= self.rescale_grad
            if self.clip_gradient is not None:
                grad = clip(grad, -self.clip_gradient, self.clip_gradient)
            
            grad += wd * weight
            
            # update mom
            mom = state
            if mom is not None:
                mom[:] *= self.momentum
                mom[:] -= lr * grad
            else:
                mom = -lr * grad

            # update weight
            weight[:] += mom

    def fused_step(self, indices, weights, grads, states):
        """Perform a fused optimization step using gradients and states.
        Fused kernel is used for update.

        Parameters
        ----------
        indices : list of int
            List of unique indices of the parameters into the individual learning rates
            and weight decays. Learning rates and weight decay may be set via `set_lr_mult()`
            and `set_wd_mult()`, respectively.
        weights : list of NDArray
            List of parameters to be updated.
        grads : list of NDArray
            List of gradients of the objective with respect to this parameter.
        states : List of any obj
            List of state returned by `create_state()`.
        """
        # When either weight or gradient is sparse, aggregate is False.
        aggregate = self.aggregate_num > 1
        for weight, grad in zip(weights, grads):
            if len(list(grad.shape_array()))>3:
                   grad -= grad.mean(axis=0, exclude=True, keepdims=True)
            aggregate = (aggregate and
                         weight.stype == 'default' and
                         grad.stype == 'default')
        self._update_count(indices)
        lrs = self._get_lrs(indices)
        wds = self._get_wds(indices)

        kwargs = {'rescale_grad': self.rescale_grad}
        if self.momentum > 0:
            kwargs['momentum'] = self.momentum
        if self.clip_gradient:
            kwargs['clip_gradient'] = self.clip_gradient

        if aggregate:
            # update `aggregate_num` number of weights in a single kernel.
            # this does not support sparse weight or gradient.
            multi_precision = self.multi_precision and weights[0].dtype == numpy.float16
            if not multi_precision:
                if self.momentum > 0:
                    multi_sgd_mom_update(*_flatten_list(zip(weights, grads, states)), out=weights,
                                         num_weights=len(weights), lrs=lrs, wds=wds, **kwargs)
                else:
                    multi_sgd_update(*_flatten_list(zip(weights, grads)), out=weights,
                                     num_weights=len(weights), lrs=lrs, wds=wds, **kwargs)
            else:
                states = list(zip(*states))
                weights32, moms = states
                if self.momentum > 0:
                    multi_mp_sgd_mom_update(*_flatten_list(zip(weights, grads,
                                                               moms, weights32)),
                                            out=weights, num_weights=len(weights),
                                            lrs=lrs, wds=wds, **kwargs)
                else:
                    multi_mp_sgd_update(*_flatten_list(zip(weights, grads,
                                                           weights32)),
                                        out=weights, num_weights=len(weights),
                                        lrs=lrs, wds=wds, **kwargs)
        else:
            for weight, grad, state, lr, wd in zip(weights, grads, states, lrs, wds):
                multi_precision = self.multi_precision and weight.dtype == numpy.float16
                if not multi_precision:
                    mom = state
                    if mom is not None:
                        sgd_mom_update(weight, grad, mom, out=weight,
                                       lazy_update=self.lazy_update, lr=lr, wd=wd, **kwargs)
                    else:
                        sgd_update(weight, grad, out=weight, lazy_update=self.lazy_update,
                                   lr=lr, wd=wd, **kwargs)
                else:
                    # weight32 is a float32 copy of weight.
                    # in the kernel, we firstly update weight32,
                    # and then cast the result to float16 and save it to weight.
                    weight32, mom = state
                    if mom is not None:
                        mp_sgd_mom_update(weight, grad, mom, weight32, out=weight,
                                          lr=lr, wd=wd, **kwargs)
                    else:
                        mp_sgd_update(weight, grad, weight32, out=weight,
                                      lr=lr, wd=wd, **kwargs)

    def update_multi_precision(self, indices, weights, grads, states):
        """Override update_multi_precision.
        """
        if self.use_fused_step:
            self.update(indices, weights, grads, states)
        else:
            super(SGD, self).update_multi_precision(indices, weights, grads, states)

@register
class AdamGCC(Optimizer):
    """The Adam optimizer.

    This class implements the optimizer described in *Adam: A Method for
    Stochastic Optimization*, available at http://arxiv.org/abs/1412.6980.

    If the storage types of grad is ``row_sparse``, and ``lazy_update`` is True, \
    **lazy updates** at step t are applied by::

        for row in grad.indices:
            rescaled_grad[row] = clip(grad[row] * rescale_grad, clip_gradient) + wd * weight[row]
            m[row] = beta1 * m[row] + (1 - beta1) * rescaled_grad[row]
            v[row] = beta2 * v[row] + (1 - beta2) * (rescaled_grad[row]**2)
            lr = learning_rate * sqrt(1 - beta2**t) / (1 - beta1**t)
            w[row] = w[row] - lr * m[row] / (sqrt(v[row]) + epsilon)

    The lazy update only updates the mean and var for the weights whose row_sparse
    gradient indices appear in the current batch, rather than updating it for all indices.
    Compared with the original update, it can provide large improvements in model training
    throughput for some applications. However, it provides slightly different semantics than
    the original update, and may lead to different empirical results.

    Otherwise, **standard updates** at step t are applied by::

        rescaled_grad = clip(grad * rescale_grad, clip_gradient) + wd * weight
        m = beta1 * m + (1 - beta1) * rescaled_grad
        v = beta2 * v + (1 - beta2) * (rescaled_grad**2)
        lr = learning_rate * sqrt(1 - beta2**t) / (1 - beta1**t)
        w = w - lr * m / (sqrt(v) + epsilon)

    This optimizer accepts the following parameters in addition to those accepted
    by :class:`.Optimizer`.

    For details of the update algorithm, see :class:`~mxnet.ndarray.adam_update`.

    Parameters
    ----------
    learning_rate : float, default 0.001
        The initial learning rate. If None, the optimization will use the
        learning rate from ``lr_scheduler``. If not None, it will overwrite
        the learning rate in ``lr_scheduler``. If None and ``lr_scheduler``
        is also None, then it will be set to 0.01 by default.
    beta1 : float, default 0.9
        Exponential decay rate for the first moment estimates.
    beta2 : float, default 0.999
        Exponential decay rate for the second moment estimates.
    epsilon : float, default 1e-8
        Small value to avoid division by 0.
    lazy_update : bool, default False
       Default is False. If True, lazy updates are applied \
       if the storage types of weight and grad are both ``row_sparse``.
    use_fused_step : bool, default True
        Whether or not to use fused kernels for optimizer.
        When use_fused_step=False, step is called,
        otherwise, fused_step is called.
    """
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
                 lazy_update=False, use_fused_step=False, **kwargs):
        super(AdamGCC, self).__init__(use_fused_step=use_fused_step,
                                   learning_rate=learning_rate,
                                   **kwargs)
        if not self.use_fused_step:
            assert not lazy_update,\
                'When use_fused_step is set to False, lazy_update has to be turned off.'
        self.lazy_update = lazy_update
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.lazy_update = lazy_update

    def create_state(self, index, weight):
        stype = weight.stype if self.lazy_update else 'default'
        return (zeros(weight.shape, weight.context, dtype=weight.dtype,
                      stype=stype),  # mean
                zeros(weight.shape, weight.context, dtype=weight.dtype,
                      stype=stype))  # variance


    def step(self, indices, weights, grads, states):
        """Perform an optimization step using gradients and states.

        Parameters
        ----------
        indices : list of int
            List of unique indices of the parameters into the individual learning rates
            and weight decays. Learning rates and weight decay may be set via `set_lr_mult()`
            and `set_wd_mult()`, respectively.
        weights : list of NDArray
            List of parameters to be updated.
        grads : list of NDArray
            List of gradients of the objective with respect to this parameter.
        states : List of any obj
            List of state returned by `create_state()`.
        """
        for index, weight, grad, state in zip(indices, weights, grads, states):
            self._update_count(index)
            lr = self._get_lr(index)
            wd = self._get_wd(index)
            t = self._index_update_count[index]
            if len(list(grad.shape_array()))>3:
                grad -= grad.mean(axis=0, exclude=True, keepdims=True)
            # preprocess grad
            grad *= self.rescale_grad
            if self.clip_gradient is not None:
                grad = clip(grad, - self.clip_gradient, self.clip_gradient)
            grad += wd * weight

            coef1 = 1. - self.beta1**t
            coef2 = 1. - self.beta2**t
            lr *= math.sqrt(coef2) / coef1

            # update mean and var
            mean, var = state
            mean[:] *= self.beta1
            mean[:] += (1. - self.beta1) * grad
            var[:] *= self.beta2
            var[:] += (1. - self.beta2) * square(grad)

            # update weight
            d = mean / (sqrt(var) + self.epsilon)
            weight[:] -= lr * d


    def fused_step(self, indices, weights, grads, states):
        """Perform a fused optimization step using gradients and states.
        Fused kernel is used for update.

        Parameters
        ----------
        indices : list of int
            List of unique indices of the parameters into the individual learning rates
            and weight decays. Learning rates and weight decay may be set via `set_lr_mult()`
            and `set_wd_mult()`, respectively.
        weights : list of NDArray
            List of parameters to be updated.
        grads : list of NDArray
            List of gradients of the objective with respect to this parameter.
        states : List of any obj
            List of state returned by `create_state()`.
        """
        for index, weight, grad, state in zip(indices, weights, grads, states):
            if len(list(grad.shape_array()))>3:
                grad -= grad.mean(axis=0, exclude=True, keepdims=True)
            self._update_count(index)
            lr = self._get_lr(index)
            wd = self._get_wd(index)
            t = self._index_update_count[index]

            coef1 = 1. - self.beta1**t
            coef2 = 1. - self.beta2**t

            lr *= math.sqrt(coef2)/coef1

            kwargs = {'beta1': self.beta1, 'beta2': self.beta2, 'epsilon': self.epsilon,
                      'rescale_grad': self.rescale_grad}
            if self.clip_gradient:
                kwargs['clip_gradient'] = self.clip_gradient

            mean, var = state

            # update weight with fused kernel
            adam_update(weight, grad, mean, var, out=weight,
                        lazy_update=self.lazy_update, lr=lr, wd=wd, **kwargs)

@register
class NAGGCC(Optimizer):
    """Nesterov accelerated gradient.

    This optimizer updates each weight by::

        grad = clip(grad * rescale_grad, clip_gradient) + wd * weight
        state = momentum * state + lr * grad
        weight = weight - (momentum * state + lr * grad)

    Parameters
    ----------
    learning_rate : float, default 0.1
        The initial learning rate. If None, the optimization will use the
        learning rate from ``lr_scheduler``. If not None, it will overwrite
        the learning rate in ``lr_scheduler``. If None and ``lr_scheduler``
        is also None, then it will be set to 0.01 by default.
    momentum : float, default 0.9
       The momentum value.
    multi_precision: bool, default False
        Flag to control the internal precision of the optimizer.
        False: results in using the same precision as the weights (default),
        True: makes internal 32-bit copy of the weights and applies gradients
        in 32-bit precision even if actual weights used in the model have lower precision.
        Turning this on can improve convergence and accuracy when training with float16.
    use_fused_step : bool, default True
        Whether or not to use fused kernels for optimizer.
        When use_fused_step=False, step is called,
        otherwise, fused_step is called.
    """
    def __init__(self, learning_rate=0.1, momentum=0.9, multi_precision=False,
                 use_fused_step=True, **kwargs):
        super(NAGGCC, self).__init__(learning_rate=learning_rate,
                                  multi_precision=multi_precision,
                                  use_fused_step=use_fused_step,
                                  **kwargs)
        self.momentum = momentum

    def create_state(self, index, weight):
        momentum = None
        if self.momentum != 0.0:
            momentum = zeros(weight.shape, weight.context, dtype=weight.dtype)
        return momentum

    def step(self, indices, weights, grads, states):
        """Perform an optimization step using gradients and states.

        Parameters
        ----------
        indices : list of int
            List of unique indices of the parameters into the individual learning rates
            and weight decays. Learning rates and weight decay may be set via `set_lr_mult()`
            and `set_wd_mult()`, respectively.
        weights : list of NDArray
            List of parameters to be updated.
        grads : list of NDArray
            List of gradients of the objective with respect to this parameter.
        states : List of any obj
            List of state returned by `create_state()`.
        """
        for index, weight, grad, state in zip(indices, weights, grads, states):
            if len(list(grad.shape_array()))>3:
                grad -= grad.mean(axis=0, exclude=True, keepdims=True)
            self._update_count(index)
            lr = self._get_lr(index)
            wd = self._get_wd(index)

            # preprocess grad
            grad *= self.rescale_grad
            if self.clip_gradient is not None:
                grad = clip(grad, -self.clip_gradient, self.clip_gradient)
            grad += wd * weight

            # update mom
            mom = state
            if mom is not None:
                mom[:] *= self.momentum
                mom[:] -= lr * grad
                d = self.momentum * mom - lr * grad
            else:
                d = -lr * grad

            # update weight
            weight[:] += d

    def fused_step(self, indices, weights, grads, states):
        """Perform a fused optimization step using gradients and states.
        Fused kernel is used for update.

        Parameters
        ----------
        indices : list of int
            List of unique indices of the parameters into the individual learning rates
            and weight decays. Learning rates and weight decay may be set via `set_lr_mult()`
            and `set_wd_mult()`, respectively.
        weights : list of NDArray
            List of parameters to be updated.
        grads : list of NDArray
            List of gradients of the objective with respect to this parameter.
        states : List of any obj
            List of state returned by `create_state()`.
        """
        for index, weight, grad, state in zip(indices, weights, grads, states):
            if len(list(grad.shape_array()))>3:
                grad -= grad.mean(axis=0, exclude=True, keepdims=True)
            self._update_count(index)
            lr = self._get_lr(index)
            wd = self._get_wd(index)

            kwargs = {'rescale_grad': self.rescale_grad}
            if self.momentum > 0:
                kwargs['momentum'] = self.momentum
            if self.clip_gradient:
                kwargs['clip_gradient'] = self.clip_gradient

            multi_precision = self.multi_precision and weight.dtype == numpy.float16

            if not multi_precision:
                mom = state
                if mom is not None:
                    nag_mom_update(weight, grad, mom, out=weight, lr=lr, wd=wd, **kwargs)
                else:
                    sgd_update(weight, grad, out=weight, lr=lr, wd=wd, **kwargs)
            else:
                weight32, mom = state
                if mom is not None:
                    mp_nag_mom_update(weight, grad, mom, weight32, out=weight,
                                      lr=lr, wd=wd, **kwargs)
                else:
                    mp_sgd_update(weight, grad, weight32, out=weight,
                                  lr=lr, wd=wd, **kwargs)

    def update_multi_precision(self, indices, weights, grads, states):
        """Override update_multi_precision.
        """
        if self.use_fused_step:
            self.update(indices, weights, grads, states)
        else:
            super(NAG, self).update_multi_precision(indices, weights, grads, states)