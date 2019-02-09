import tensorflow as tf
import numpy as np


class SantaSSSOptimizer(tf.train.Optimizer):

    def __init__(self, eta, burnin, annealing_fn,
                 sigma=0.999, alpha=1000, epsilon=1e-8,
                 use_locking=False, name="SantaSSSOptimizer"):

        super().__init__(use_locking, name)

        self.eta = eta
        self.sigma = sigma
        self.alpha = alpha
        self.epsilon = epsilon
        self.burnin = burnin
        self.annealing_fn = annealing_fn

    def _create_slots(self, var_list):

        for var in var_list:
            self._get_or_make_slot_with_initializer(
                var, tf.zeros_initializer(),
                var.shape, var.dtype, "v", self._name
            )
            self._get_or_make_slot_with_initializer(
                var, tf.zeros_initializer(),
                var.shape, var.dtype, "g", self._name
            )
            self._get_or_make_slot_with_initializer(
                var, tf.constant_initializer(np.sqrt(self.eta) * self.alpha),
                var.shape, var.dtype, "a", self._name
            )
            self._get_or_make_slot_with_initializer(
                var, tf.random_normal_initializer(stddev=np.sqrt(self.eta)),
                var.shape, var.dtype, "u", self._name
            )

    def _prepare(self):

        self.eta = tf.convert_to_tensor(self.eta, name="eta")
        self.sigma = tf.convert_to_tensor(self.sigma, name="sigma")
        self.epsilon = tf.convert_to_tensor(self.epsilon, name="epsilon")
        self.burnin = tf.convert_to_tensor(self.burnin, name="burnin")

    def _apply_dense(self, grad, var):

        t = tf.train.get_global_step()

        v = self.get_slot(var, "v")
        g = self.get_slot(var, "g")
        a = self.get_slot(var, "a")
        u = self.get_slot(var, "u")

        eta = tf.cast(self.eta, var.dtype.base_dtype)
        sigma = tf.cast(self.sigma, var.dtype.base_dtype)
        epsilon = tf.cast(self.epsilon, var.dtype.base_dtype)
        burnin = tf.cast(self.burnin, t.dtype.base_dtype)

        b = self.annealing_fn(tf.cast(t, var.dtype.base_dtype))
        z = tf.random_normal(shape=var.shape, dtype=var.dtype.base_dtype)

        def _update(exploration):

            v_ = sigma * v + (1 - sigma) * grad * grad
            g_ = 1 / tf.sqrt(epsilon + tf.sqrt(v_))

            var_ = var + g_ * u / 2

            if exploration:
                a_ = a + (u * u - eta / b) / 2
                u_ = tf.exp(- a_ / 2) * u
                u_ = u_ - eta * g_ * grad
                u_ = u_ + tf.sqrt(2 * eta / b * g) * z
                u_ = u_ + eta / b * (1 - g / g_) / u
                u_ = tf.exp(- a_ / 2) * u_
                a_ = a_ + (u_ * u_ - eta / b) / 2
            else:
                a_ = a
                u_ = tf.exp(- a_ / 2) * u
                u_ = u_ - eta * g_ * grad
                u_ = tf.exp(- a_ / 2) * u_

            var_ = var_ + g_ * u_ / 2

            return var_, v_, g_, a_, u_

        var_, v_, g_, a_, u_ = tf.cond(
            pred=tf.less(t, burnin),
            true_fn=lambda: _update(True),
            false_fn=lambda: _update(False)
        )

        return tf.group(*[
            var.assign(var_),
            v.assign(v_),
            g.assign(g_),
            a.assign(a_),
            u.assign(u_),
        ])
