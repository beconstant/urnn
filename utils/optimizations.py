import theano
import lasagne
import theano.tensor as T
import numpy as np

from collections import OrderedDict


def clipped_gradients(gradients, gradient_clipping):
    clipped_grads = [T.clip(g, -gradient_clipping, gradient_clipping)
                     for g in gradients]
    return clipped_grads


def gradient_descent(learning_rate, parameters, gradients):
    updates = [(p, p - learning_rate * g) for p, g in zip(parameters, gradients)]
    return updates


def gradient_descent_momentum(learning_rate, momentum, parameters, gradients):
    velocities = [theano.shared(np.zeros_like(p.get_value(),
                                              dtype=theano.config.floatX)) for p in parameters]

    updates1 = [(vel, momentum * vel - learning_rate * g)
                for vel, g in zip(velocities, gradients)]
    updates2 = [(p, p + vel) for p, vel in zip(parameters, velocities)]
    updates = updates1 + updates2
    return updates


def rms_prop(learning_rate, parameters, gradients):
    rmsprop = [theano.shared(1e-3 * np.ones_like(p.get_value())) for p in parameters]
    new_rmsprop = [0.9 * vel + 0.1 * (g ** 2) for vel, g in zip(rmsprop, gradients)]

    updates1 = list(zip(rmsprop, new_rmsprop))
    updates2 = [(p, p - learning_rate * g / T.sqrt(rms)) for
                p, g, rms in zip(parameters, gradients, new_rmsprop)]
    updates = updates1 + updates2
    return updates, rmsprop


def custom_sgd(loss_or_grads, params, learning_rate, manifolds=None):
    """Stochastic Gradient Descent (SGD) updates

    Generates update expressions of the form:

    * ``param := param - learning_rate * gradient``

    Parameters
    ----------
    loss_or_grads : symbolic expression or list of expressions
        A scalar loss expression, or a list of gradient expressions
    params : list of shared variables
        The variables to generate update expressions for
    learning_rate : float or symbolic scalar
        The learning rate controlling the size of update steps
    manifolds : dict
        Dictionary that contains manifolds for manifold parameters

    Returns
    -------
    OrderedDict
        A dictionary mapping each parameter to its update expression
    """

    def filter_func(manifold_name, inverse=False):
        def inner_filter_func(param_grad_tuple):
            filter_result = (hasattr(param_grad_tuple[0], 'name') and manifold_name in param_grad_tuple[0].name)
            return not filter_result if inverse else filter_result

        return inner_filter_func

    manifolds = manifolds if manifolds else {}
    grads = lasagne.updates.get_or_compute_grads(loss_or_grads, params)
    updates = OrderedDict()

    manifolds_params_stack = []
    manifolds_grads_stack = []

    if isinstance(manifolds, dict) and manifolds:
        for manifold_name in manifolds:
            # filter parameters and gradients for specific manifold
            man_params_tuple, man_grads_tuple = zip(*filter(filter_func(manifold_name), zip(params, grads)))

            man_params_tuple = {manifold_name: tuple(man_params_tuple)}
            man_grads_tuple = {manifold_name: tuple(man_grads_tuple)}

            if len(man_params_tuple[manifold_name]) == len(params):
                params, grads = [], []
            else:
                params, grads = zip(*filter(filter_func(manifold_name, inverse=True), zip(params, grads)))
            manifolds_params_stack.append(man_params_tuple)
            manifolds_grads_stack.append(man_grads_tuple)
            params = list(params)
            grads = list(grads)

    params = manifolds_params_stack + params
    grads = manifolds_grads_stack + grads

    for param, grad in zip(params, grads):
        if isinstance(param, dict):
            manifold_name = list(param.keys())[0]
            manifold = manifolds[manifold_name]
            if hasattr(manifold, "from_partial"):
                grad_from_partial = manifold.from_partial(param[manifold_name], grad[manifold_name])
                grad_step = manifold.lincomb(param[manifold_name], grad_from_partial, -learning_rate)
                param_updates = manifold.retr(param[manifold_name], grad_step)
                for p, upd in zip(param[manifold_name], param_updates):
                    updates[p] = upd
            else:
                param_tuple = param[manifold_name]
                grad_tuple = grad[manifold_name]
                if len(param_tuple) == 1:
                    param_tuple, grad_tuple = param_tuple[0], grad_tuple[0]
                grad_step = manifold.lincomb(param_tuple, manifold.proj(param_tuple, grad_tuple), -learning_rate)
                param_updates = manifold.retr(param_tuple, grad_step)
                updates[param_tuple] = param_updates
        else:
            updates[param] = param - learning_rate * grad
    return updates


def sgd(loss_or_grads, params, learning_rate):
    """Stochastic Gradient Descent (SGD) updates
    Generates update expressions of the form:
    * ``param := param - learning_rate * gradient``
    Parameters
    ----------
    loss_or_grads : symbolic expression or list of expressions
        A scalar loss expression, or a list of gradient expressions
    params : list of shared variables
        The variables to generate update expressions for
    learning_rate : float or symbolic scalar
        The learning rate controlling the size of update steps
    Returns
    -------
    OrderedDict
        A dictionary mapping each parameter to its update expression
    """
    grads = lasagne.updates.get_or_compute_grads(loss_or_grads, params)
    updates = OrderedDict()

    for param, grad in zip(params, grads):
        updates[param] = param - learning_rate * grad

    return updates