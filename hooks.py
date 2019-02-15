import tensorflow as tf


class ValidationMonitorHook(tf.train.SessionRunHook):
    """ Hook to extend calls to MonitoredSession.run(). """

    def __init__(self, estimator, input_fn, every_n_secs=None, every_n_steps=None, **kwargs):

        self.timer = tf.train.SecondOrStepTimer(every_n_secs, every_n_steps)
        self.estimator = estimator
        self.input_fn = input_fn
        self.kwargs = kwargs

    def begin(self):
        """ Called once before using the session.

        When called, the default graph is the one that will be launched in the
        session.  The hook can modify the graph by adding new operations to it.
        After the `begin()` call the graph will be finalized and the other callbacks
        can not modify the graph anymore. Second call of `begin()` on the same
        graph, should not change the graph.
        """
        self.timer.reset()
        self.global_step = tf.train.get_global_step()

    def after_create_session(self, session, coord):
        """ Called when new TensorFlow session is created.

        This is called to signal the hooks that a new session has been created. This
        has two essential differences with the situation in which `begin` is called:

        * When this is called, the graph is finalized and ops can no longer be added
            to the graph.
        * This method will also be called as a result of recovering a wrapped
            session, not only at the beginning of the overall session.

        Args:
          session: A TensorFlow Session that has been created.
          coord: A Coordinator object which keeps track of all threads.
        """
        pass

    def before_run(self, run_context):
        """ Called before each call to run().

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
        """ Called after each call to run().

        The `run_values` argument contains results of requested ops/tensors by
        `before_run()`.

        The `run_context` argument is the same one send to `before_run` call.
        `run_context.request_stop()` can be called to stop the iteration.

        If `session.run()` raises any exceptions then `after_run()` is not called.

        Args:
          run_context: A `SessionRunContext` object.
          run_values: A `SessionRunValues` object.
        """
        global_step = run_values.results

        if self.timer.should_trigger_for_step(global_step):

            eval_result = self.estimator.evaluate(self.input_fn, **self.kwargs)

            print("==================================================")
            tf.logging.info("validation result")
            tf.logging.info(eval_result)
            print("==================================================")

            self.timer.update_last_triggered_step(global_step)

    def end(self, session):
        """ Called at the end of session.

        The `session` argument can be used in case the hook wants to run final ops,
        such as saving a last checkpoint.

        If `session.run()` raises exception other than OutOfRangeError or
        StopIteration then `end()` is not called.
        Note the difference between `end()` and `after_run()` behavior when
        `session.run()` raises OutOfRangeError or StopIteration. In that case
        `end()` is called but `after_run()` is not called.

        Args:
          session: A TensorFlow Session that will be soon closed.
        """
        pass


class LearningRateDecayHook(tf.train.SessionRunHook):
    """ Hook to extend calls to MonitoredSession.run(). """

    def __init__(self, estimator, input_fn, learning_rate_name, decay_rate, decay_steps,
                 every_n_secs=None, every_n_steps=None, **kwargs):

        self.timer = tf.train.SecondOrStepTimer(every_n_secs, every_n_steps)
        self.learning_rate_name = learning_rate_name
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.estimator = estimator
        self.input_fn = input_fn
        self.kwargs = kwargs

        self.min_loss = None
        self.min_step = None

    def begin(self):
        """ Called once before using the session.

        When called, the default graph is the one that will be launched in the
        session.  The hook can modify the graph by adding new operations to it.
        After the `begin()` call the graph will be finalized and the other callbacks
        can not modify the graph anymore. Second call of `begin()` on the same
        graph, should not change the graph.
        """
        self.timer.reset()
        self.global_step = tf.train.get_global_step()

        tf.get_variable_scope().reuse_variables()
        self.learning_rate = tf.get_variable(name=self.learning_rate_name, dtype=tf.float64)
        self.decayed_learning_rate = tf.placeholder(dtype=tf.float64, shape=[])
        self.assign_op = self.learning_rate.assign(self.decayed_learning_rate)

    def after_create_session(self, session, coord):
        """ Called when new TensorFlow session is created.

        This is called to signal the hooks that a new session has been created. This
        has two essential differences with the situation in which `begin` is called:

        * When this is called, the graph is finalized and ops can no longer be added
            to the graph.
        * This method will also be called as a result of recovering a wrapped
            session, not only at the beginning of the overall session.

        Args:
          session: A TensorFlow Session that has been created.
          coord: A Coordinator object which keeps track of all threads.
        """
        pass

    def before_run(self, run_context):
        """ Called before each call to run().

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
        return tf.train.SessionRunArgs([self.global_step, self.learning_rate])

    def after_run(self, run_context, run_values):
        """ Called after each call to run().

        The `run_values` argument contains results of requested ops/tensors by
        `before_run()`.

        The `run_context` argument is the same one send to `before_run` call.
        `run_context.request_stop()` can be called to stop the iteration.

        If `session.run()` raises any exceptions then `after_run()` is not called.

        Args:
          run_context: A `SessionRunContext` object.
          run_values: A `SessionRunValues` object.
        """
        global_step, learning_rate = run_values.results

        if self.timer.should_trigger_for_step(global_step):

            eval_result = self.estimator.evaluate(self.input_fn, **self.kwargs)

            print("==================================================")
            tf.logging.info("validation result")
            tf.logging.info(eval_result)
            print("==================================================")

            if (self.min_loss is None) or (eval_result["loss"] < self.min_loss):

                self.min_loss = eval_result["loss"]
                self.min_step = global_step

            if (global_step - self.min_step) >= self.decay_steps:

                print("==================================================")
                tf.logging.info("loss didn't decrease in {} steps".format(self.decay_steps))
                tf.logging.info("decay learning rate (decay rate: {})".format(self.decay_rate))
                print("==================================================")

                run_context.session.run(
                    fetches=[self.assign_op],
                    feed_dict={self.decayed_learning_rate: learning_rate * self.decay_rate}
                )

            self.timer.update_last_triggered_step(global_step)

    def end(self, session):
        """ Called at the end of session.

        The `session` argument can be used in case the hook wants to run final ops,
        such as saving a last checkpoint.

        If `session.run()` raises exception other than OutOfRangeError or
        StopIteration then `end()` is not called.
        Note the difference between `end()` and `after_run()` behavior when
        `session.run()` raises OutOfRangeError or StopIteration. In that case
        `end()` is called but `after_run()` is not called.

        Args:
          session: A TensorFlow Session that will be soon closed.
        """
        pass
