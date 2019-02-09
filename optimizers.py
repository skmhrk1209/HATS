import tensorflow as tf
import numpy as np


class SantaOptimizer(tf.train.Optimizer):

    def __init__(self, eta=4e-6, gamma=0.5, sigma=0.95, const=1000, epsilon=1e-8,
                 burnin=10000, use_locking=False, name="SantaOptimizer"):

        super().__init__(use_locking, name)

        self.eta = eta
        self.gamma = gamma
        self.sigma = sigma
        self.const = const
        self.epsilon = epsilon
        self.burnin = burnin

        self.eta_t = None
        self.gamma_t = None
        self.sigma_t = None
        self.epsilon_t = None
        self.burnin_t = None

    def _create_slots(self, var_list):

        for var in var_list:

            self._get_or_make_slot_with_initializer(
                var, tf.zeros_initializer(),
                var.shape, var.dtype, "v", self._name
            )
            self._get_or_make_slot_with_initializer(
                var, tf.constant_initializer(1 / np.sqrt(self.epsilon)),
                var.shape, var.dtype, "g", self._name
            )
            self._get_or_make_slot_with_initializer(
                var, tf.constant_initializer(np.sqrt(self.eta) * self.const),
                var.shape, var.dtype, "a", self._name
            )
            self._get_or_make_slot_with_initializer(
                var, tf.random_normal_initializer(stddev=np.sqrt(self.eta)),
                var.shape, var.dtype, "u", self._name
            )

    def _prepare(self):

        self.eta_t = tf.convert_to_tensor(self.eta, name="eta")
        self.gamma_t = tf.convert_to_tensor(self.gamma, name="gamma")
        self.sigma_t = tf.convert_to_tensor(self.sigma, name="sigma")
        self.epsilon_t = tf.convert_to_tensor(self.epsilon, name="epsilon")
        self.burnin_t = tf.convert_to_tensor(self.burnin, name="burnin")

    def _apply_dense(self, grad, var):

        t = tf.train.get_global_step()

        v = self.get_slot(var, "v")
        g = self.get_slot(var, "g")
        a = self.get_slot(var, "a")
        u = self.get_slot(var, "u")

        eta = tf.cast(self.eta_t, var.dtype)
        gamma = tf.cast(self.gamma_t, var.dtype)
        sigma = tf.cast(self.sigma_t, var.dtype)
        epsilon = tf.cast(self.epsilon_t, var.dtype)
        burnin = tf.cast(self.burnin_t, t.dtype)

        def _update(exploration):

            beta = tf.cast(t + 1, var.dtype) ** gamma
            zeta = tf.random_normal(var.shape)

            v_ = sigma * v + (1 - sigma) * grad * grad
            g_ = 1 / tf.sqrt(epsilon + tf.sqrt(v_))

            var_ = var + g_ * u / 2

            if exploration:
                a_ = a + (u * u - eta / beta) / 2
                u_ = tf.exp(- a_ / 2) * u
                u_ = u_ - eta * g_ * grad
                #u_ = u_ + tf.sqrt(2 * eta / beta * g) * zeta
                #u_ = u_ + eta / beta * (1 - g / g_) / u
                u_ = u_ + (2 * eta ** 1.5 / beta * g) ** 0.5 * zeta
                u_ = tf.exp(- a_ / 2) * u_
                a_ = a_ + (u_ * u_ - eta / beta) / 2
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
