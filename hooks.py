import tensorflow as tf


class ValidationHook(tf.train.SessionRunHook):

    def __init__(self, estimator, input_fn, every_n_secs=None, every_n_steps=None, **kwargs):

        self.timer = tf.train.SecondOrStepTimer(every_n_secs, every_n_steps)
        self.estimator = estimator
        self.input_fn = input_fn
        self.kwargs = kwargs

    def begin(self):
        """Called once before using the session.

        When called, the default graph is the one that will be launched in the
        session.  The hook can modify the graph by adding new operations to it.
        After the `begin()` call the graph will be finalized and the other callbacks
        can not modify the graph anymore. Second call of `begin()` on the same
        graph, should not change the graph.
        """

        self.timer.reset()
        self.global_step = tf.train.get_global_step()

    def before_run(self, run_context):
        """Called before each call to run().

        You can return from this call a `SessionRunArgs` object indicating ops or
        tensors to add to the upcoming `run()` call.  These ops/tensors will be run
        together with the ops/tensors originally passed to the original run() call.
        The run args you return can also contain feeds to be added to the run()
        call.

        The `run_context` argument is a `SessionRunContext` that provides
        information about the upcoming `run()` call: the originally requested
        op/tensors, the TensorFlow Session.

        At this point graph is finalized and you can not add ops.

        Args:
        run_context: A `SessionRunContext` object.

        Returns:
        None or a `SessionRunArgs` object.
        """

        return tf.train.SessionRunArgs(self.global_step)

    def after_run(self, run_context, run_values):
        """Called after each call to run().

        The `run_values` argument contains results of requested ops/tensors by
        `before_run()`.

        The `run_context` argument is the same one send to `before_run` call.
        `run_context.request_stop()` can be called to stop the iteration.

        If `session.run()` raises any exceptions then `after_run()` is not called.

        Args:
        run_context: A `SessionRunContext` object.
        run_values: A `SessionRunValues` object.
        """

        global_step = run_values.result
        if self.timer.should_trigger_for_step(global_step):
            print(self.estimator.evaluate(self.input_fn, **self.kwargs))
            self.timer.update_last_triggered_step(global_step)
