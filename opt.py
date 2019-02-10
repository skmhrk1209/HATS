"""Eve for TensorFlow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_ops
from tensorflow.python.util.tf_export import tf_export
from tensorflow.python.training.optimizer import *
import tensorflow as tf


class OptimizableVariable(object):
    """Interface for abstracting over variables in the optimizers."""

    @abc.abstractmethod
    def target(self):
        """Returns the optimization target for this variable."""
        raise NotImplementedError("Calling an abstract method.")

    @abc.abstractmethod
    def update_op(self, optimizer, grad):
        """Returns the update ops for updating the variable."""
        raise NotImplementedError("Calling an abstract method.")


class RefVariableProcessor(OptimizableVariable):
    """Processor for Variable."""

    def __init__(self, var):
        self.var = var

    def __str__(self):
        return "<_RefVariableProcessor(%s)>" % self.var

    def target(self):
        return self.var._ref()

    def update_op(self, optimizer, loss, grad, global_step):
        if isinstance(grad, ops.Tensor):
            update_op = optimizer._apply_dense(loss, grad, self.var, global_step)
            if self.var.constraint is not None:
                with ops.control_dependencies([update_op]):
                    return self.var.assign(self.var.constraint(self.var))
            else:
                return update_op
        else:
            assert isinstance(grad, ops.IndexedSlices), ("Gradient ", grad, " is neither a tensor nor IndexedSlices.")
            if self.var.constraint is not None:
                raise RuntimeError("Cannot use a constraint function on a sparse variable.")
            return optimizer._apply_sparse_duplicate_indices(grad, self.var)


def get_processor(var):
    """The processor of var"""
    if context.executing_eagerly():
        if isinstance(var, ops.Tensor):
            return _TensorProcessor(var)
        else:
            return _DenseResourceVariableProcessor(var)
    if isinstance(var, resource_variable_ops.ResourceVariable) and not var._in_graph_mode:
        # True if and only if `v` was initialized eagerly.
        return _DenseResourceVariableProcessor(var)
    if var.op.type == "VarHandleOp":
        return _DenseResourceVariableProcessor(var)
    if isinstance(var, variables.Variable):
        return RefVariableProcessor(var)
    if isinstance(var, ops.Tensor):
        return _TensorProcessor(var)
    raise NotImplementedError("Trying to optimize unsupported type ", var)


class EveOptimizer(optimizer.Optimizer):
    """Optimizer that implements the Eve algorithm.
    See [Kingma et al., 2014](https://arxiv.org/pdf/1611.01505.pdf)
    """

    # Values for gate_gradients.
    GATE_NONE = 0
    GATE_OP = 1
    GATE_GRAPH = 2

    def __init__(self, alpha1=1e-3, beta1=0.9, beta2=0.999, beta3=0.999,
                 clip_value=10, epsilon=1e-8, use_locking=False, name="Eve"):
        """Construct a new Eve optimizer.
        Args:
          alpha1: A Tensor or a floating point value.  
            The learning rate.
          beta1: A float value or a constant float tensor.
            The exponential decay rate for the 1st moment estimates.
          beta2: A float value or a constant float tensor.
            The exponential decay rate for the 2nd moment estimates.
          beta3: A float value or a constant float tensor.
            The exponential decay rate for computing relative change.
          epsilon: A float value or a constant float tensor.
            A small constant for numerical stability. 
          use_locking: If True use locks for update operations.
          name: Optional name for the operations created when applying gradients.
            Defaults to "Eve".
        @compatibility(eager)
        When eager execution is enabled, `alpha1`, `beta1`, `beta2`, `beta3`, `clip_value`, 
        and `epsilon` can each be a callable that takes no arguments and returns the
        actual value to use. This can be useful for changing these values across
        different invocations of optimizer functions.
        @end_compatibility
        """
        super(EveOptimizer, self).__init__(use_locking, name)
        self.alpha1 = alpha1
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
        self.clip_value = clip_value
        self.epsilon = epsilon

    def _create_slots(self, var_list):
        # Create the beta1 and beta2 accumulators on the same device as the first
        # variable. Sort the var_list to make sure this device is consistent across
        # workers (these need to go on the same PS, otherwise some updates are
        # silently ignored).
        first_var = min(var_list, key=lambda x: x.name)
        self._create_non_slot_variable(
            initial_value=self.beta1,
            name="beta1_power",
            colocate_with=first_var
        )
        self._create_non_slot_variable(
            initial_value=self.beta2,
            name="beta2_power",
            colocate_with=first_var
        )
        self._create_non_slot_variable(
            initial_value=0.0,
            name="prev_loss",
            colocate_with=first_var
        )
        # Create slots for the first and second moments.
        for var in var_list:
            self._zeros_slot(var, "m", self._name)
            self._zeros_slot(var, "v", self._name)
            self._zeros_slot(var, "d", self._name)

    def _prepare(self):
        self.alpha1 = ops.convert_to_tensor(
            value=self._call_if_callable(self.alpha1),
            name="alpha1"
        )
        self.beta1 = ops.convert_to_tensor(
            value=self._call_if_callable(self.beta1),
            name="beta1"
        )
        self.beta2 = ops.convert_to_tensor(
            value=self._call_if_callable(self.beta2),
            name="beta2"
        )
        self.beta3 = ops.convert_to_tensor(
            value=self._call_if_callable(self.beta3),
            name="beta3"
        )
        self.clip_value = ops.convert_to_tensor(
            value=self._call_if_callable(self.clip_value),
            name="clip_value"
        )
        self.epsilon = ops.convert_to_tensor(
            value=self._call_if_callable(self.epsilon),
            name="epsilon"
        )

    def minimize(self, loss, global_step=None, var_list=None,
                 gate_gradients=GATE_OP, aggregation_method=None,
                 colocate_gradients_with_ops=False, name=None, grad_loss=None):
        """Add operations to minimize `loss` by updating `var_list`.
        This method simply combines calls `compute_gradients()` and
        `apply_gradients()`. If you want to process the gradient before applying
        them call `compute_gradients()` and `apply_gradients()` explicitly instead
        of using this function.
        Args:
        loss: A `Tensor` containing the value to minimize.
        global_step: Optional `Variable` to increment by one after the
            variables have been updated.
        var_list: Optional list or tuple of `Variable` objects to update to
            minimize `loss`.  Defaults to the list of variables collected in
            the graph under the key `GraphKeys.TRAINABLE_VARIABLES`.
        gate_gradients: How to gate the computation of gradients.  Can be
            `GATE_NONE`, `GATE_OP`, or  `GATE_GRAPH`.
        aggregation_method: Specifies the method used to combine gradient terms.
            Valid values are defined in the class `AggregationMethod`.
        colocate_gradients_with_ops: If True, try colocating gradients with
            the corresponding op.
        name: Optional name for the returned operation.
        grad_loss: Optional. A `Tensor` holding the gradient computed for `loss`.
        Returns:
        An Operation that updates the variables in `var_list`.  If `global_step`
        was not `None`, that operation also increments `global_step`.
        Raises:
        ValueError: If some of the variables are not `Variable` objects.
        @compatibility(eager)
        When eager execution is enabled, `loss` should be a Python function that
        takes no arguments and computes the value to be minimized. Minimization (and
        gradient computation) is done with respect to the elements of `var_list` if
        not None, else with respect to any trainable variables created during the
        execution of the `loss` function. `gate_gradients`, `aggregation_method`,
        `colocate_gradients_with_ops` and `grad_loss` are ignored when eager
        execution is enabled.
        @end_compatibility
        """
        grads_and_vars = self.compute_gradients(
            loss=loss,
            var_list=var_list,
            gate_gradients=gate_gradients,
            aggregation_method=aggregation_method,
            colocate_gradients_with_ops=colocate_gradients_with_ops,
            grad_loss=grad_loss
        )

        if not [var for grad, var in grads_and_vars if grad is not None]:
            raise ValueError(
                "No gradients provided for any variable, check your graph for ops"
                " that do not support gradients, between variables %s and loss %s." %
                ([str(var) for _, var in grads_and_vars], loss)
            )

        return self.apply_gradients(
            loss=loss,
            grads_and_vars=grads_and_vars,
            global_step=global_step,
            name=name
        )

    def apply_gradients(self, loss, grads_and_vars, global_step=None, name=None):
        """Apply gradients to variables.
        This is the second part of `minimize()`. It returns an `Operation` that
        applies gradients.
        Args:
        grads_and_vars: List of (gradient, variable) pairs as returned by
            `compute_gradients()`.
        global_step: Optional `Variable` to increment by one after the
            variables have been updated.
        name: Optional name for the returned operation.  Default to the
            name passed to the `Optimizer` constructor.
        Returns:
        An `Operation` that applies the specified gradients. If `global_step`
        was not None, that operation also increments `global_step`.
        Raises:
        TypeError: If `grads_and_vars` is malformed.
        ValueError: If none of the variables have gradients.
        RuntimeError: If you should use `_distributed_apply()` instead.
        """
        # This is a default implementation of apply_gradients() that can be shared
        # by most optimizers.  It relies on the subclass implementing the following
        # methods: _create_slots(), _prepare(), _apply_dense(), and _apply_sparse().

        # Handle DistributionStrategy case.
        if distribution_strategy_context.get_cross_tower_context():
            raise RuntimeError("Use `_distributed_apply()` instead of `apply_gradients()` in a cross-tower context.")
        # TODO(isaprykin): Get rid of `has_distribution_strategy()` check by
        # always calling _distributed_apply(), using the default distribution
        # as needed.
        if distribution_strategy_context.has_distribution_strategy():
            grads_and_vars = get_filtered_grad_fn(lambda: grads_and_vars)()
            return distribution_strategy_context.get_tower_context().merge_call(
                self._distributed_apply, grads_and_vars, global_step, name
            )

        # No DistributionStrategy case.
        grads_and_vars = tuple(grads_and_vars)  # Make sure repeat iteration works.
        if not grads_and_vars:
            raise ValueError("No variables provided.")
        converted_grads_and_vars = []
        for grad, var in grads_and_vars:
            if grad is not None:
                try:
                    # Convert the grad to Tensor or IndexedSlices if necessary.
                    grad = ops.convert_to_tensor_or_indexed_slices(grad)
                except TypeError:
                    raise TypeError("Gradient must be convertible to a Tensor or IndexedSlices, or None: %s" % grad)
                if not isinstance(grad, (ops.Tensor, ops.IndexedSlices)):
                    raise TypeError("Gradient must be a Tensor, IndexedSlices, or None: %s" % grad)
            processor = get_processor(var)
            converted_grads_and_vars.append((grad, var, processor))

        converted_grads_and_vars = tuple(converted_grads_and_vars)
        var_list = [var for grad, var, _ in converted_grads_and_vars if grad is not None]
        if not var_list:
            raise ValueError("No gradients provided for any variable: %s." % ([str(var) for _, var, _ in converted_grads_and_vars],))
        with ops.init_scope():
            self._create_slots(var_list)
        update_ops = []
        with ops.name_scope(name, self._name) as name:
            self._prepare()
            for grad, var, processor in converted_grads_and_vars:
                if grad is None:
                    continue
                # We colocate all ops created in _apply_dense or _apply_sparse
                # on the same device as the variable.
                # TODO(apassos): figure out how to get the variable name here.
                if context.executing_eagerly() or isinstance(var, resource_variable_ops.ResourceVariable) and not var._in_graph_mode:
                    scope_name = ""
                else:
                    scope_name = var.op.name
                with ops.name_scope("update_" + scope_name), ops.colocate_with(var):
                    update_ops.append(processor.update_op(self, loss, grad, global_step))
            if global_step is None:
                apply_updates = self._finish(update_ops, loss, name)
            else:
                with ops.control_dependencies([self._finish(update_ops, loss, "update")]):
                    with ops.colocate_with(global_step):
                        if isinstance(global_step, resource_variable_ops.ResourceVariable):
                            # TODO(apassos): the implicit read in assign_add is slow; consider
                            # making it less so.
                            apply_updates = resource_variable_ops.assign_add_variable_op(
                                resource=global_step.handle,
                                value=ops.convert_to_tensor(
                                    value=1,
                                    dtype=global_step.dtype
                                ),
                                name=name
                            )
                        else:
                            apply_updates = state_ops.assign_add(
                                ref=global_step,
                                value=1,
                                name=name
                            )

            if not context.executing_eagerly():
                if isinstance(apply_updates, ops.Tensor):
                    apply_updates = apply_updates.op
                train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
                if apply_updates not in train_op:
                    train_op.append(apply_updates)

            return apply_updates

    def _apply_dense(self, loss, grad, var, global_step):

        with ops.init_scope():
            graph = None if context.executing_eagerly() else ops.get_default_graph()
            beta1_power = self._get_non_slot_variable("beta1_power", graph=graph)
            beta2_power = self._get_non_slot_variable("beta2_power", graph=graph)
            prev_loss = self._get_non_slot_variable("prev_loss", graph=graph)

        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")
        d = self.get_slot(var, "d")

        beta1_power = math_ops.cast(beta1_power, var.dtype.base_dtype)
        beta2_power = math_ops.cast(beta2_power, var.dtype.base_dtype)
        alpha1 = math_ops.cast(self.alpha1, var.dtype.base_dtype)
        beta1 = math_ops.cast(self.beta1, var.dtype.base_dtype)
        beta2 = math_ops.cast(self.beta2, var.dtype.base_dtype)
        beta3 = math_ops.cast(self.beta3, var.dtype.base_dtype)
        clip_value = math_ops.cast(self.clip_value, var.dtype.base_dtype)
        epsilon = math_ops.cast(self.epsilon, var.dtype.base_dtype)

        m = m.assign(beta1 * m + (1 - beta1) * grad)
        m_hat = m / (1 - beta1_power)

        v = v.assign(beta2 * v + (1 - beta2) * grad ** 2)
        v_hat = v / (1 - beta2_power)

        d = d.assign(tf.cond(
            pred=tf.greater(global_step, tf.zeros_like(global_step)),
            true_fn=lambda: (beta3 * d + (1 - beta3) * tf.clip_by_value(
                t=tf.abs(loss - prev_loss) / tf.minimum(loss, prev_loss),
                clip_value_min=1 / clip_value,
                clip_value_max=clip_value
            )),
            false_fn=lambda: tf.ones_like(d)
        ))

        var = var.assign_sub(alpha1 / d * m_hat / (v_hat ** 0.5 + epsilon))

        return var

    def _finish(self, update_ops, loss, name_scope):
        # Update the power accumulators.
        with ops.control_dependencies(update_ops):
            with ops.init_scope():
                graph = None if context.executing_eagerly() else ops.get_default_graph()
                beta1_power = self._get_non_slot_variable("beta1_power", graph=graph)
                beta2_power = self._get_non_slot_variable("beta2_power", graph=graph)
                prev_loss = self._get_non_slot_variable("prev_loss", graph=graph)
            with ops.colocate_with(beta1_power):
                update_beta1_power = beta1_power.assign(
                    value=beta1_power * self.beta1,
                    use_locking=self._use_locking
                )
                update_beta2_power = beta2_power.assign(
                    value=beta2_power * self.beta2,
                    use_locking=self._use_locking
                )
                update_prev_loss = prev_loss.assign_add(
                    delta=loss,
                    use_locking=self._use_locking
                )
        return control_flow_ops.group(
            *(update_ops + [update_beta1_power, update_beta2_power, update_prev_loss]),
            name=name_scope
        )
